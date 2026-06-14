"""TorchRL integration for the Sushi Go! environment.

The PettingZoo env exposes structured observations because the encoder path needs
sequential leaves such as `hand_history` and `opponent_tableaus`. The baseline
MLP/DQN path still wants one flat vector per agent, so this module also owns the
deterministic TensorDict flattener used by actor/critic/DQN builders.
"""

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import RewardSum, TransformedEnv, check_env_specs
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor, ValueOperator

from SushiGo_env.sushi_go_env import OBS_COMPONENTS, SushiGoParallelEnv, N_TYPES

# Grouped TensorDict keys produced by PettingZooWrapper.
GROUP = "players"
# Structured state leaves, each shaped as [n_agents, ...] after grouping.
OBS_KEYS = tuple((GROUP, "observation", key) for key in OBS_COMPONENTS)
# Compatibility key produced by `make_observation_flattener()`.
OBS_KEY = (GROUP, "flat_observation")
# Scalar per-agent PettingZoo masks become [n_agents, 1] after TorchRL grouping.
PLAYER_MASK_KEY = (GROUP, "observation", "player_mask")
MASK_KEY = (GROUP, "action_mask")
ACTION_KEY = (GROUP, "action")
LOGITS_KEY = (GROUP, "logits")
VALUE_KEY = (GROUP, "state_value")


class StructuredObservationFlattener(nn.Module):
    """Flatten Sushi Go structured observation components into one per-agent vector.

    Components arrive with a shared leading shape ending in the agent axis, for
    example `[T, B, n_agents, history_len, N_TYPES]`. `component_ndims` tells the
    module how many trailing dimensions belong to that component so it can flatten
    only feature dimensions and preserve all leading batch/agent dimensions.
    """

    def __init__(self, component_ndims):
        super().__init__()
        self.component_ndims = tuple(component_ndims)

    def forward(self, *components):
        """Return `[..., n_agents, flat_obs_dim]` from ordered structured leaves."""
        flat_components = [
            component.flatten(start_dim=-ndim)
            for component, ndim in zip(components, self.component_ndims)
        ]
        return torch.cat(flat_components, dim=-1)


def make_observation_flattener(out_key=OBS_KEY):
    """Build the TensorDictModule that creates the flat observation key.

    The order must match `OBS_COMPONENTS` and `SushiGoParallelEnv.flatten_observation`;
    changing it would silently change the baseline model's input semantics.
    """
    component_ndims = (1, 2, 1, 2, 1, 1)
    return TensorDictModule(
        StructuredObservationFlattener(component_ndims),
        in_keys=list(OBS_KEYS),
        out_keys=[out_key],
    )


def flat_observation_dim(env):
    """Compute the flattened per-agent observation width from TorchRL specs."""
    return sum(int(torch.tensor(env.observation_spec[key].shape[1:]).prod()) for key in OBS_KEYS)


def make_torchrl_env(n_players=3, min_n_players=None, max_n_players=None, history_len=None,
                     include_opponent_tableaus=True, reward_scale=0.1, device="cpu"):
    """Build a Sushi Go env wrapped for TorchRL with dense max-player grouping.

    Fixed games pass `n_players`. Stochastic games pass `n_players=None` and
    `min_n_players`/`max_n_players`; the TorchRL group size is the max count, while
    the env's `player_mask` marks which dense slots are active in each episode.
    """
    if n_players is not None and (min_n_players is not None or max_n_players is not None):
        raise ValueError("Use either n_players or min_n_players/max_n_players, not both.")
    
    if n_players is None and (min_n_players is None or max_n_players is None):
        raise ValueError("Must specify either n_players or min_n_players/max_n_players.")
    
    group_n_players = n_players if n_players is not None else max_n_players

    base = SushiGoParallelEnv(
        n_players=n_players,
        min_n_players=min_n_players,
        max_n_players=max_n_players,
        history_len=history_len,
        include_opponent_tableaus=include_opponent_tableaus,
        reward_scale=reward_scale,
    )
    env = PettingZooWrapper(
        base,
        use_mask=True,             # exposes ("players", "action_mask")
        categorical_actions=True,  # integer actions, not one-hot
        group_map={GROUP: [f"player_{i}" for i in range(group_n_players)]},
        device=device,
    )
    # Tracks the per-seat episode return at ("next", "players", "episode_reward").
    env = TransformedEnv(
        env, RewardSum(in_keys=[(GROUP, "reward")], out_keys=[(GROUP, "episode_reward")])
    )
    return env


def build_actor(n_players, obs_dim, num_cells=128, depth=2, device="cpu"):
    """Masked multi-agent policy: structured obs -> flat obs -> logits -> action."""
    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=N_TYPES,      # one logit per card type
        n_agents=n_players,
        centralised=False,            # decentralised actor (acts on its own obs)
        share_params=True,            # one shared policy across all seats (self-play)
        depth=depth,
        num_cells=num_cells,
        activation_class=torch.nn.Tanh,
        device=device,
    )
    module = TensorDictModule(net, in_keys=[OBS_KEY], out_keys=[LOGITS_KEY])
    actor = ProbabilisticActor(
        module=module,
        in_keys={"logits": LOGITS_KEY, "mask": MASK_KEY},
        out_keys=[ACTION_KEY],
        distribution_class=MaskedCategorical,   # forbids drafting absent cards
        return_log_prob=True,
        log_prob_key=(GROUP, "sample_log_prob"),
    )
    return TensorDictSequential(make_observation_flattener(), actor)


def build_critic(n_players, obs_dim, num_cells=128, depth=2,
                 centralised=False, device="cpu"):
    """Multi-agent value network with the same structured-observation flattening."""
    net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=n_players,
        centralised=centralised,
        share_params=True,
        depth=depth,
        num_cells=num_cells,
        activation_class=torch.nn.Tanh,
        device=device,
    )
    return TensorDictSequential(
        make_observation_flattener(),
        ValueOperator(net, in_keys=[OBS_KEY], out_keys=[VALUE_KEY]),
    )


# test
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    for n in (2, 3, 4):
        env = make_torchrl_env(n_players=n)
        check_env_specs(env)
        obs_dim = flat_observation_dim(env)

        actor = build_actor(n, obs_dim)
        critic = build_critic(n, obs_dim)

        td = env.reset()
        actor(td)              # populate logits + action + log-prob
        critic(td)             # populate state value
        td = env.step(td)

        # A short random-policy rollout exercises reset/step/termination.
        rollout = env.rollout(max_steps=120, policy=actor)

        print(f"n_players={n}: obs_dim={obs_dim} | "
              f"action shape={td[ACTION_KEY].shape} | "
              f"value shape={td[VALUE_KEY].shape} | "
              f"rollout length={rollout.batch_size}")
    print("\nEnvironment, actor and critic are correctly wired for TorchRL.")
    print("Attach your MARL training loop (collector + PPO/MAPPO loss) to these modules.")
