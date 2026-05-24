"""
Independent multi-agent DQN  (MADQN / IQL)  for the Sushi Go! environment, TorchRL.


"Independent Q-Learning" for a competitive, symmetric, simultaneous-move game.
Every seat is controlled by ONE shared Q-network (parameter sharing = self-play).
Each seat picks the action that maximises its own Q-value; there is no centralised
critic and no value decomposition (VDN/QMIX would assume a cooperative shared
reward — wrong for Sushi Go). This is exactly "MA independent learning", value
based. For the policy-gradient analogue (IPPO) reuse the actor/critic in
`torchrl_integration.py` with a PPO loss instead.

HOW THE PIECES MAP TO DQN
=========================
  Q-network          : MultiAgentMLP, obs -> 12 Q-values (one per card type).
  Action selection   : QValueModule does a *masked* argmax — illegal cards (not in
                       hand) are excluded via the env's action_mask.
  Exploration        : EGreedyModule, epsilon annealed over training; random picks
                       are also restricted to legal cards by the mask.
  Target network     : DQNLoss(delay_value=True) keeps a slow copy for stable
                       bootstrap targets; SoftUpdate nudges it each step.
  Replay buffer      : transitions stored and sampled i.i.d. (off-policy).
  Loss               : DQNLoss with a TD(0) target.

"""
import argparse
import warnings

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.envs import check_env_specs, ParallelEnv
from torchrl.modules import EGreedyModule, MultiAgentMLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators

from SushiGo_env.sushi_go_env import N_TYPES
from SushiGo_env.torchrl_integration import make_torchrl_env, OBS_KEY, MASK_KEY, GROUP, ACTION_KEY

warnings.filterwarnings("ignore")

# Grouped tensordict keys the Q-pipeline reads / writes.
ACTION_VALUE_KEY = (GROUP, "action_value")          # the 12 Q-values
CHOSEN_VALUE_KEY = (GROUP, "chosen_action_value")   # Q of the action actually taken

NUM_WORKERS = 4 


def build_qvalue_actor(n_players, obs_dim, num_cells=128, depth=2, device="cpu"):
    """Shared-parameter multi-agent Q-network + masked argmax head."""
    q_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=N_TYPES,      # one Q-value per card type
        n_agents=n_players,
        centralised=False,            # each seat sees only its own observation
        share_params=True,           # ONE network across all seats (self-play)
        depth=depth,
        num_cells=num_cells,
        activation_class=nn.Tanh,
        device=device,
    )
    q_module = TensorDictModule(q_net, in_keys=[OBS_KEY], out_keys=[ACTION_VALUE_KEY])

    # QValueModule turns Q-values into a (masked) greedy action. The action_mask_key
    # makes it exclude cards not in hand BEFORE the argmax — for both the acting
    # policy and, inside DQNLoss, the bootstrap target.
    qvalue_module = QValueModule(
        action_space="categorical", #  means it outputs an integer index
        action_value_key=ACTION_VALUE_KEY,
        action_mask_key=MASK_KEY,
        out_keys=[ACTION_KEY, ACTION_VALUE_KEY, CHOSEN_VALUE_KEY],
    )
    return TensorDictSequential(q_module, qvalue_module)



def train(args):
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    if args.smoke:  # tiny run — only checks the pipeline executes end to end
        args.iterations, args.frames_per_batch = 3, 600
        args.buffer_size, args.batch_size, args.updates_per_batch = 2000, 256, 8

    total_frames = args.frames_per_batch * args.iterations
    print(f"device={device}  n_players={args.n_players}  total_frames={total_frames}")

    # environment  
    env = make_torchrl_env(
        n_players=args.n_players, reward_scale=args.reward_scale, device=device)
    check_env_specs(env)
    obs_dim = env.observation_spec[OBS_KEY].shape[-1]

    # Q-network  
    qvalue_actor = build_qvalue_actor(args.n_players, obs_dim, device=device)
    qvalue_actor(env.reset())  # warm up lazy parameters with a real observation
    # It must happen before you build the optimizer or the loss, because those need real parameters to attach to.

    # exploration: epsilon-greedy on top of the greedy Q-policy 
    explore = EGreedyModule(
        spec=env.action_spec,
        eps_init=1.0,
        eps_end=0.05,
        annealing_num_steps=total_frames // 2,
        action_key=ACTION_KEY,
        action_mask_key=MASK_KEY,         # random exploration also stays legal
    )
    collector_policy = TensorDictSequential(qvalue_actor, explore)

    # DQN loss + target network 
    # computes the TD error: the gap between Q(s,a) and the bootstrap target r + γ·max Q(s',·).
    # delay_value=True is the target network
    loss_module = DQNLoss(qvalue_actor, action_space="categorical", delay_value=True)
    
    # Define the corrected keys
    loss_module.set_keys(
        action_value=ACTION_VALUE_KEY,
        action=ACTION_KEY,
        value=CHOSEN_VALUE_KEY,
        reward=(GROUP, "reward"),
        done=(GROUP, "done"),
        terminated=(GROUP, "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=args.gamma)
    # make_value_estimator(TD0) sets the bootstrap target to one-step TD: r + γ·max Q(s',·)
    
    target_updater = SoftUpdate(loss_module, eps=args.target_eps)
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # data collection + replay buffer
    def env_factory():
        return make_torchrl_env(
            n_players=args.n_players, reward_scale=args.reward_scale, device=device)

    collector = MultiSyncDataCollector(
        create_env_fn=[env_factory] * NUM_WORKERS,
        policy=collector_policy,
        frames_per_batch=args.frames_per_batch,   # split across workers automatically
        total_frames=total_frames,
        device=device,
        storing_device="cpu",
    )
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=device),
        batch_size=args.batch_size,
    )

    # training loop
    for it, batch in enumerate(collector):
        replay.extend(batch.reshape(-1))  # flatten time dim; agent dim stays nested

        last_loss = None
        for _ in range(args.updates_per_batch):
            sample = replay.sample()
            loss_vals = loss_module(sample)
            loss = loss_vals["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm) # caps the gradient magnitude so a rare huge TD error can't blow up the weights
            optim.step()
            optim.zero_grad()
            target_updater.step()        # slow target-network update
            last_loss = loss.item()

        explore.step(args.frames_per_batch)   # anneal epsilon
        collector.update_policy_weights_()

        # Logging: mean per-turn reward, and per-seat episode return where games ended.
        mean_r = batch.get(("next", GROUP, "reward")).mean().item()
        done = batch.get(("next", GROUP, "done"))
        ep_ret = batch.get(("next", GROUP, "episode_reward"))
        finished = ep_ret[done]
        eps_now = explore.eps.item() if hasattr(explore.eps, "item") else float(explore.eps)
        msg = (f"iter {it:3d} | loss={last_loss:.4f} | eps={eps_now:.3f} "
               f"| mean turn reward={mean_r:+.3f}")
        if finished.numel() > 0:
            msg += f" | mean episode return/seat={finished.float().mean().item():+.2f}"
        print(msg)

    if not args.smoke:
        torch.save(qvalue_actor.state_dict(), args.save_path)
        print(f"saved Q-network -> {args.save_path}")
    collector.shutdown()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-players", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--frames-per-batch", type=int, default=5000)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--updates-per-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-eps", type=float, default=0.995)  # SoftUpdate mix factor
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny wiring-check run")
    p.add_argument("--save-path", type=str, default="sushi_go_qnet_2_players.pt")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())