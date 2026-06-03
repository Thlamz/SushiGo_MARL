"""Sushi Go specific adapter for the vendored flexible encoder.

The vendored encoder is intentionally generic: it expects named sequential and
flat tensors plus an optional dense agent mask. This module binds that generic
interface to the structured TorchRL keys emitted by `SushiGoParallelEnv`.
"""

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP

from SushiGo_env.encoder import (
    FlatEncoderConfig,
    FlatEncoderInput,
    MultiAgentFlexModule,
    SequentialEncoderConfig,
    SequentialEncoderInput,
)
from SushiGo_env.sushi_go_env import N_TYPES
from SushiGo_env.torchrl_integration import (
    ACTION_KEY,
    GROUP,
    MASK_KEY,
    PLAYER_MASK_KEY,
)

ENCODER_EMBEDDING_KEY = (GROUP, "encoder_embedding")


def build_sushigo_encoder(
    env,
    n_agents: int,
    output_dim: int,
    device="cpu",
    *,
    sequential_embed_dim: int = 64,
    sequential_head_dim: int = 16,
    sequential_num_heads: int = 4,
    sequential_ff_dim: int = 128,
    sequential_depth: int = 2,
    sequential_dropout: float = 0.0,
    flat_embed_dim: int = 64,
    flat_depth: int = 1,
    flat_num_cells: int = 128,
    mix_layer_depth: int = 1,
    mix_layer_num_cells: int = 128,
) -> TensorDictModule:
    """Build a TensorDictModule that encodes structured Sushi Go observations.

    Sequential leaves (`hand_history`, `opponent_tableaus`) keep their sequence
    axis and are processed by transformer heads. Flat leaves are processed by MLP
    heads. The scalar per-agent `player_mask` is routed to the encoder's `mask`
    argument so inactive dense player slots do not affect sequential attention.
    """

    sequential_config = SequentialEncoderConfig(
        embed_dim=sequential_embed_dim,
        head_dim=sequential_head_dim,
        num_heads=sequential_num_heads,
        ff_dim=sequential_ff_dim,
        depth=sequential_depth,
        dropout=sequential_dropout,
        max_num_agents=n_agents,
        agentic_encoding=True,
    )
    flat_config = FlatEncoderConfig(
        embed_dim=flat_embed_dim,
        depth=flat_depth,
        num_cells=flat_num_cells,
        activation_class=nn.Tanh,
    )

    sequential_inputs = [
        SequentialEncoderInput(
            key="hand_history",
            input_size=env.observation_spec[(GROUP, "observation", "hand_history")].shape[-1],
        ),
        SequentialEncoderInput(
            key="opponent_tableaus",
            input_size=env.observation_spec[(GROUP, "observation", "opponent_tableaus")].shape[-1],
        ),
    ]
    flat_inputs = [
        FlatEncoderInput(
            key="current_hand",
            input_size=env.observation_spec[(GROUP, "observation", "current_hand")].shape[-1],
        ),
        FlatEncoderInput(
            key="own_tableau",
            input_size=env.observation_spec[(GROUP, "observation", "own_tableau")].shape[-1],
        ),
        FlatEncoderInput(
            key="cards_played",
            input_size=env.observation_spec[(GROUP, "observation", "cards_played")].shape[-1],
        ),
        FlatEncoderInput(
            key="game_scalars",
            input_size=env.observation_spec[(GROUP, "observation", "game_scalars")].shape[-1],
        ),
    ]

    encoder = MultiAgentFlexModule(
        sequential_config=sequential_config,
        sequential_inputs=sequential_inputs,
        flat_inputs=flat_inputs,
        flat_config=flat_config,
        mix_layer_depth=mix_layer_depth,
        mix_layer_num_cells=mix_layer_num_cells,
        mix_activation_class=nn.Tanh,
        output_dim=output_dim,
        n_agents=n_agents,
        centralized=False,
        share_params=True,
        device=device,
    )

    in_keys = {
        "hand_history": (GROUP, "observation", "hand_history"),
        "opponent_tableaus": (GROUP, "observation", "opponent_tableaus"),
        "current_hand": (GROUP, "observation", "current_hand"),
        "own_tableau": (GROUP, "observation", "own_tableau"),
        "cards_played": (GROUP, "observation", "cards_played"),
        "game_scalars": (GROUP, "observation", "game_scalars"),
        # TorchRL groups scalar PettingZoo player_mask values as [n_agents, 1].
        # The encoder expects [n_agents], so this is squeezed by MaskSqueezer below.
        "mask": (GROUP, "player_mask"),
    }
    return TensorDictModule(
        encoder,
        in_keys=in_keys,
        out_keys=[ENCODER_EMBEDDING_KEY],
        out_to_in_map=True,
    )


class MaskSqueezer(nn.Module):
    """Move and squeeze player_mask into the shape expected by MultiAgentFlexModule."""

    def forward(self, player_mask: torch.Tensor) -> torch.Tensor:
        return player_mask.squeeze(-1).bool()


def build_encoder_qvalue_actor(
    env,
    n_agents: int,
    qvalue_module,
    *,
    encoder_output_dim: int = 128,
    sequential_embed_dim: int = 64,
    sequential_head_dim: int = 16,
    sequential_num_heads: int = 4,
    sequential_ff_dim: int = 128,
    sequential_depth: int = 2,
    sequential_dropout: float = 0.0,
    flat_embed_dim: int = 64,
    flat_depth: int = 1,
    flat_num_cells: int = 128,
    mix_layer_depth: int = 1,
    mix_layer_num_cells: int = 128,
    q_head_cells: int = 128,
    q_head_depth: int = 1,
    device="cpu",
) -> TensorDictSequential:
    """Build encoder -> Q-head -> masked argmax for DQN training."""

    mask_module = TensorDictModule(
        MaskSqueezer(),
        in_keys=[PLAYER_MASK_KEY],
        out_keys=[(GROUP, "player_mask")],
    )
    encoder_module = build_sushigo_encoder(
        env,
        n_agents=n_agents,
        output_dim=encoder_output_dim,
        device=device,
        sequential_embed_dim=sequential_embed_dim,
        sequential_head_dim=sequential_head_dim,
        sequential_num_heads=sequential_num_heads,
        sequential_ff_dim=sequential_ff_dim,
        sequential_depth=sequential_depth,
        sequential_dropout=sequential_dropout,
        flat_embed_dim=flat_embed_dim,
        flat_depth=flat_depth,
        flat_num_cells=flat_num_cells,
        mix_layer_depth=mix_layer_depth,
        mix_layer_num_cells=mix_layer_num_cells,
    )
    q_head = MultiAgentMLP(
        n_agent_inputs=encoder_output_dim,
        n_agent_outputs=N_TYPES,
        n_agents=n_agents,
        centralised=False,
        share_params=True,
        depth=q_head_depth,
        num_cells=q_head_cells,
        activation_class=nn.Tanh,
        device=device,
    )
    q_module = TensorDictModule(
        q_head,
        in_keys=[ENCODER_EMBEDDING_KEY],
        out_keys=[(GROUP, "action_value")],
    )
    return TensorDictSequential(mask_module, encoder_module, q_module, qvalue_module)
