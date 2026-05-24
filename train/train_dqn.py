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
  Q-network          : structured obs -> deterministic flat vector ->
                       MultiAgentMLP -> 12 Q-values (one per card type).
  Action selection   : QValueModule does a *masked* argmax — illegal cards (not in
                       hand) are excluded via the env's action_mask.
  Exploration        : EGreedyModule, epsilon annealed over training; random picks
                       are also restricted to legal cards by the mask.
  Target network     : DQNLoss(delay_value=True) keeps a slow copy for stable
                       bootstrap targets; SoftUpdate nudges it each step.
  Replay buffer      : transitions stored and sampled i.i.d. (off-policy).
  Loss               : DQNLoss with a TD(0) target.

VARIABLE PLAYER COUNTS
======================
The environment keeps a dense player axis of `max_n_players`, but each reset can
sample fewer active seats. The per-slot `player_mask` is not an action mask; it
marks which dense slots are real players in the sampled episode. DQN loss and
training metrics use this mask so padded inactive seats do not contribute zero
loss/reward and bias learning.

"""
import argparse
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.envs import check_env_specs
from torchrl.modules import EGreedyModule, MultiAgentMLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators

from SushiGo_env.sushi_go_env import N_TYPES
from SushiGo_env.encoder_adapter import build_encoder_qvalue_actor
from SushiGo_env.torchrl_integration import (
    make_observation_flattener,
    make_torchrl_env,
    flat_observation_dim,
    OBS_KEY,
    MASK_KEY,
    PLAYER_MASK_KEY,
    GROUP,
    ACTION_KEY,
)

warnings.filterwarnings("ignore")

# Grouped tensordict keys the Q-pipeline reads / writes.
ACTION_VALUE_KEY = (GROUP, "action_value")          # the 12 Q-values
CHOSEN_VALUE_KEY = (GROUP, "chosen_action_value")   # Q of the action actually taken

NUM_WORKERS = 4 


@contextmanager
def maybe_profiler(args, device):
    """Optionally wrap training in torch.profiler with TensorBoard trace export."""
    if not args.profile:
        yield None
        return

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sort_by = args.profile_sort or ("self_cuda_time_total" if device == "cuda" else "self_cpu_time_total")

    print(
        "profiling enabled: "
        f"wait={args.profile_wait} warmup={args.profile_warmup} "
        f"active={args.profile_active} repeat={args.profile_repeat}"
    )
    on_trace_ready = None
    if args.profile_export_trace:
        trace_dir = Path(args.profile_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(str(trace_dir))
        print(f"exporting TensorBoard traces to: {trace_dir}")
        print(f"view traces with: tensorboard --logdir {trace_dir}")
    else:
        print("TensorBoard trace export disabled; use --profile-export-trace to write trace files.")

    prof_schedule = schedule(
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        repeat=args.profile_repeat,
        skip_first=args.profile_skip_first,
    )
    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=args.profile_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.profile_stack,
    ) as prof:
        yield prof

    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.profile_row_limit))


def profile_stop_steps(args):
    """Number of training-loop iterations needed to finish the profiler schedule."""
    if not args.profile or args.profile_repeat == 0:
        return None
    return args.profile_skip_first + args.profile_repeat * (
        args.profile_wait + args.profile_warmup + args.profile_active
    )


def build_qvalue_selector():
    """Build the masked argmax head shared by baseline and encoder DQN paths."""
    # QValueModule turns Q-values into a (masked) greedy action. The action_mask_key
    # makes it exclude cards not in hand BEFORE the argmax — for both the acting
    # policy and, inside DQNLoss, the bootstrap target.
    return QValueModule(
        action_space="categorical", #  means it outputs an integer index
        action_value_key=ACTION_VALUE_KEY,
        action_mask_key=MASK_KEY,
        out_keys=[ACTION_KEY, ACTION_VALUE_KEY, CHOSEN_VALUE_KEY],
    )


def build_mlp_qvalue_actor(n_players, obs_dim, num_cells=128, depth=2, device="cpu"):
    """Baseline shared-parameter Q-network over flattened structured observations."""
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
    return TensorDictSequential(make_observation_flattener(), q_module, build_qvalue_selector())


def build_qvalue_actor(env, n_players, obs_dim, args, device="cpu"):
    """Build the selected DQN architecture.

    Baseline mode flattens structured observations before an MLP. Encoder mode
    keeps sequential leaves structured, encodes them with the vendored transformer
    module, then applies a small Q-head.
    """
    if args.use_encoder:
        return build_encoder_qvalue_actor(
            env,
            n_agents=n_players,
            qvalue_module=build_qvalue_selector(),
            encoder_output_dim=args.encoder_output_dim,
            q_head_cells=args.encoder_q_cells,
            q_head_depth=args.encoder_q_depth,
            device=device,
        )
    return build_mlp_qvalue_actor(
        n_players,
        obs_dim,
        num_cells=args.mlp_cells,
        depth=args.mlp_depth,
        device=device,
    )


def resolve_player_config(args):
    """Resolve CLI player-count flags into env and model configuration.

    Returns `(n_players, min_n_players, max_n_players, model_n_players)`.
    `n_players` is used for fixed-count compatibility. For stochastic games,
    `model_n_players` is the dense max size because `MultiAgentMLP` has a static
    agent axis.
    """
    using_range = args.min_n_players is not None or args.max_n_players is not None
    if args.n_players is not None and using_range:
        raise ValueError("Use either --n-players or --min-n-players/--max-n-players, not both.")
    if args.n_players is not None:
        return args.n_players, None, None, args.n_players

    if not using_range:
        return 2, None, None, 2

    min_n_players = 2 if args.min_n_players is None else args.min_n_players
    max_n_players = min_n_players if args.max_n_players is None else args.max_n_players
    if not 2 <= min_n_players <= max_n_players <= 4:
        raise ValueError("player count bounds must satisfy 2 <= min <= max <= 4")
    return None, min_n_players, max_n_players, max_n_players


def masked_mean(value, mask):
    """Mean over active dense player slots only.

    TorchRL's DQNLoss has no inactive-player mask key, so the loss is requested
    with `reduction="none"` and reduced here with the current observation's
    `player_mask`.
    """
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)



def train(args):
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    n_players, min_n_players, max_n_players, model_n_players = resolve_player_config(args)

    if args.smoke:  # tiny run — only checks the pipeline executes end to end
        args.iterations, args.frames_per_batch = 3, 600
        args.buffer_size, args.batch_size, args.updates_per_batch = 2000, 256, 8

    total_frames = args.frames_per_batch * args.iterations
    player_msg = (
        f"n_players={n_players}" if n_players is not None
        else f"n_players=[{min_n_players}, {max_n_players}]"
    )
    model_msg = "encoder+DQN" if args.use_encoder else "MLP+DQN"
    print(
        f"device={device}  {player_msg}  model={model_msg}  "
        f"workers={args.num_workers}  total_frames={total_frames}"
    )

    # environment  
    env = make_torchrl_env(
        n_players=n_players,
        min_n_players=min_n_players,
        max_n_players=max_n_players,
        reward_scale=args.reward_scale,
        device=device,
    )
    check_env_specs(env)
    obs_dim = flat_observation_dim(env)

    # Q-network. In stochastic mode this is sized for max_n_players; inactive
    # slots are still forwarded, but their losses are masked out below.
    qvalue_actor = build_qvalue_actor(env, model_n_players, obs_dim, args, device=device)
    if args.compile:
        print("compiling Q-network with torch.compile()...")  # speeds up training but adds overhead, so optional
        qvalue_actor = torch.compile(qvalue_actor)
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
    # delay_value=True is the target network. reduction="none" preserves the
    # [batch, n_agents] loss tensor so padded inactive slots can be ignored.
    loss_module = DQNLoss(
        qvalue_actor,
        action_space="categorical",
        delay_value=True,
        reduction="none",
    )
    
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
            n_players=n_players,
            min_n_players=min_n_players,
            max_n_players=max_n_players,
            reward_scale=args.reward_scale,
            device=device,
        )

    if args.num_workers == 1:
        collector = SyncDataCollector(
            create_env_fn=env_factory,
            policy=collector_policy,
            frames_per_batch=args.frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device="cpu",
        )
    else:
        collector = MultiSyncDataCollector(
            create_env_fn=[env_factory] * args.num_workers,
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
    collector_iter = iter(collector)
    profile_steps = profile_stop_steps(args)
    with maybe_profiler(args, device) as prof:
        for it in range(args.iterations):
            with record_function("train/collect_batch"):
                try:
                    batch = next(collector_iter)
                except StopIteration:
                    break

            with record_function("train/replay_extend"):
                replay.extend(batch.reshape(-1))  # flatten time dim; agent dim stays nested

            last_loss = None
            for _ in range(args.updates_per_batch):
                with record_function("train/update"):
                    with record_function("train/replay_sample"):
                        sample = replay.sample()
                    with record_function("train/loss_forward"):
                        loss_vals = loss_module(sample)
                        # The sampled action/reward belong to the current observation, so use
                        # the current player_mask, not next.player_mask, for loss reduction.
                        active = sample.get(PLAYER_MASK_KEY).squeeze(-1).bool()
                        loss = masked_mean(loss_vals["loss"], active)
                    with record_function("train/backward"):
                        loss.backward()
                    with record_function("train/clip_grad"):
                        nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm) # caps the gradient magnitude so a rare huge TD error can't blow up the weights
                    with record_function("train/optim_step"):
                        optim.step()
                    with record_function("train/optim_zero_grad"):
                        optim.zero_grad()
                    with record_function("train/target_update"):
                        target_updater.step()        # slow target-network update
                    last_loss = loss.item()

            with record_function("train/post_update_policy"):
                explore.step(args.frames_per_batch)   # anneal epsilon
                collector.update_policy_weights_()

            with record_function("train/log_metrics"):
                # Logging uses next.player_mask because these rewards/episode returns live
                # under "next". This prevents inactive padded seats from diluting metrics.
                active_next = batch.get(("next", *PLAYER_MASK_KEY)).squeeze(-1).bool()
                reward = batch.get(("next", GROUP, "reward")).squeeze(-1)
                mean_r = reward[active_next].mean().item()
                done = batch.get(("next", GROUP, "done")).squeeze(-1).bool()
                ep_ret = batch.get(("next", GROUP, "episode_reward")).squeeze(-1)
                finished = ep_ret[done & active_next]
                eps_now = explore.eps.item() if hasattr(explore.eps, "item") else float(explore.eps)
                msg = (f"iter {it:3d} | loss={last_loss:.4f} | eps={eps_now:.3f} "
                       f"| mean turn reward={mean_r:+.3f}")
                if finished.numel() > 0:
                    msg += f" | mean episode return/seat={finished.float().mean().item():+.2f}"
                print(msg)

            if prof is not None:
                prof.step()
                if profile_steps is not None and it + 1 >= profile_steps:
                    print(f"profiler schedule completed after {it + 1} iterations; stopping training loop.")
                    break

    if not args.smoke and not args.profile:
        torch.save(qvalue_actor.state_dict(), args.save_path)
        print(f"saved Q-network -> {args.save_path}")
    collector.shutdown()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-players", type=int, default=None, choices=[2, 3, 4])
    p.add_argument("--min-n-players", type=int, default=None, choices=[2, 3, 4])
    p.add_argument("--max-n-players", type=int, default=None, choices=[2, 3, 4])
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
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="environment collectors; use 1 for single-process profiling")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the Q-network for faster training")
    p.add_argument("--smoke", action="store_true", help="tiny wiring-check run")
    p.add_argument("--use-encoder", action="store_true", help="use transformer encoder before the DQN Q-head")
    p.add_argument("--profile", action="store_true", help="run torch.profiler around the training loop")
    p.add_argument("--profile-dir", type=str, default="runs/profiler/dqn", help="TensorBoard trace output directory")
    p.add_argument("--profile-export-trace", action="store_true", help="write TensorBoard/Chrome trace files; can be very large")
    p.add_argument("--profile-skip-first", type=int, default=0, help="profiler steps to skip before scheduling")
    p.add_argument("--profile-wait", type=int, default=1, help="profiler schedule wait steps")
    p.add_argument("--profile-warmup", type=int, default=1, help="profiler schedule warmup steps")
    p.add_argument("--profile-active", type=int, default=3, help="profiler schedule active steps")
    p.add_argument("--profile-repeat", type=int, default=1, help="profiler schedule repeats")
    p.add_argument("--profile-shapes", action="store_true", help="record tensor shapes in profiler events")
    p.add_argument("--profile-memory", action="store_true", help="record tensor memory usage in profiler events")
    p.add_argument("--profile-stack", action="store_true", help="record Python stacks; useful but expensive")
    p.add_argument("--profile-sort", type=str, default=None, help="sort column for the printed profiler summary")
    p.add_argument("--profile-row-limit", type=int, default=25, help="number of rows in the printed profiler summary")
    p.add_argument("--mlp-cells", type=int, default=128)
    p.add_argument("--mlp-depth", type=int, default=2)
    p.add_argument("--encoder-output-dim", type=int, default=128)
    p.add_argument("--encoder-q-cells", type=int, default=128)
    p.add_argument("--encoder-q-depth", type=int, default=1)
    p.add_argument("--save-path", type=str, default="sushi_go_qnet_2_players.pt")
    args = p.parse_args()
    if args.num_workers < 1:
        p.error("--num-workers must be >= 1")
    return args


if __name__ == "__main__":
    train(get_args())
