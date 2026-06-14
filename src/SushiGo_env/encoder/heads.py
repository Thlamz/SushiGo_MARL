import torch
from torch import nn
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING

from .configs import SequentialEncoderConfig, SequentialEncoderInput, FlatEncoderConfig, FlatEncoderInput


class SequentialEncoder(nn.Module):
    """
    A Transformer-based encoder that processes sequential observations. It receives data of shape
    (*B, seq_len, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input sequence.
    """
    def __init__(
        self,
        input: SequentialEncoderInput,
        config: SequentialEncoderConfig,
        device: DEVICE_TYPING | None = None,
    ):
        """Initialize the sequential head.

        Args:
            config: Configuration for the sequential head.
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config
        self.input = input

        # Linear layer to project input features to the embedding dimension
        self.obs_encoder = nn.Linear(input.input_size, config.embed_dim, device=device)
        # Embedding layer for agent indices
        self.agent_embedder = nn.Embedding(
            num_embeddings=config.max_num_agents,
            embedding_dim=config.embed_dim,
            device=device,
        ) if config.agentic_encoding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            device=device,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
            enable_nested_tensor=True,
        )

    def forward(self, x: torch.Tensor, agent_idx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequential observation for one agent.

        Args:
            x: Input tensor of shape (*B, seq_len, input_dim).
            agent_idx: Input tensor of shape (*B, 1) containing the agent index for each batch entry. Will be used for
                agentic encoding if it's enabled.
            mask: Optional tensor of shape (*B, seq_len) indicating valid timesteps. Not currently used.

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """

        leading_batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        x_flat = x.reshape(-1, seq_len, x.shape[-1])

        # Observations where all features are -1 are padding and should be ignored by the
        # attention mechanism. Create a mask that marks these timesteps so the attention
        # layers do not attend to them.
        padded_input_mask = torch.all(x_flat == -1, dim=-1)

        # Combine with the externally provided agent mask
        if mask is not None:
            padded_input_mask |= ~mask.reshape(-1, seq_len)

        embed_output = self.obs_encoder(x_flat)

        # If agentic encoding is enabled, add the agent embedding to every step in the sequence.
        if self.agent_embedder is not None:
            # If the agent_idx shape matches the leading batch dimensions of x, we can directly embed it. 
            # Otherwise, we need to reshape it to match the flattened batch dimensions of x_flat.
            if agent_idx.shape[:-1] == x_flat.shape[:-1]:
                agent_embeddings = self.agent_embedder(agent_idx.squeeze(-1).to(torch.long))
            else:
                agent_embeddings = self.agent_embedder(agent_idx.reshape(-1, 1).to(torch.long)).squeeze(dim=-2)
                agent_embeddings = agent_embeddings.unsqueeze(-2)
            embed_output += agent_embeddings

        transformer_padding_mask = padded_input_mask.clone()
        fully_padded_rows = transformer_padding_mask.all(dim=-1)
        transformer_padding_mask[..., 0] &= ~fully_padded_rows

        seq_output = self.transformer(embed_output, src_key_padding_mask=transformer_padding_mask)

        # Aggregate only valid timesteps so masked entries do not influence the output.
        valid_timestep_mask = ~padded_input_mask
        valid_timestep_mask = valid_timestep_mask.unsqueeze(-1).to(seq_output.dtype)
        seq_output = (seq_output * valid_timestep_mask).sum(dim=-2) / valid_timestep_mask.sum(dim=-2).clamp_min(1.0)

        return seq_output.reshape(*leading_batch_shape, seq_output.shape[-1])


class FlatEncoder(nn.Module):
    """
    An MLP-based head for processing a single flat observation key for one agent. The input is a tensor of shape
    (*B, input_dim) where *B denotes zero or more leading batch dimensions. The output is a tensor of shape
    (*B, embed_dim) representing an intermediate representation of the input.
    """
    def __init__(
        self,
        input: FlatEncoderInput,
        config: FlatEncoderConfig,
        input_dim: int | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        """Initialize the flat head.

        Args:
            config: Configuration for the flat head.
            input_dim: Optional override for input size (useful when inputs are concatenated externally).
            device: Device to place the modules on. Defaults to CPU when ``None``.
        """
        super().__init__()
        self.config = config

        resolved_input_dim = input_dim if input_dim is not None else input.input_size
        self.mlp = MLP(
            in_features=resolved_input_dim,
            out_features=config.embed_dim,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=config.activation_class,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a flat observation for one agent.

        Args:
            x: Input tensor of shape (*B, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (*B, embed_dim).
        """
        return self.mlp(x)
