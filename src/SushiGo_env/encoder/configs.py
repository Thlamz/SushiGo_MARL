from dataclasses import dataclass
from torch import nn


@dataclass
class SequentialEncoderConfig:
    """
    Configuration for a SequentialEncoder
    """
    embed_dim: int
    """Dimensionality of the output embedding produced by this head."""
    head_dim: int
    """Dimensionality of the output of each attention head."""
    num_heads: int
    """Number of attention heads in the Transformer blocks."""
    ff_dim: int
    """Dimensionality of the feedforward layers in the Transformer blocks."""
    depth: int
    """Number of Transformer blocks in the head."""
    dropout: float
    """Dropout rate used in the Transformer blocks."""
    max_num_agents: int
    """Maximum number of agents expected in the environment. Necessary for generating agentic embeddings."""
    agentic_encoding: bool
    """When ``True``, adds an agent embedding to the input based on the agent index."""


@dataclass
class SequentialEncoderInput:
    """
    Represents an input that should be processed by a SequentialEncoder.
    """
    key: str
    """The key this encoder processes."""
    input_size: int
    """Dimensionality of a single input feature in the sequence"""


@dataclass
class FlatEncoderConfig:
    """
    Configuration for a FlatEncoder
    """
    embed_dim: int
    """Dimensionality of the output embedding produced by this head."""
    depth: int
    """Number of hidden layers in the MLP."""
    num_cells: int
    """Number of cells per hidden layer in the MLP."""
    activation_class: type[nn.Module]
    """Activation function class used between MLP layers."""


@dataclass
class FlatEncoderInput:
    """
    Represents an input that should be processed by a FlatEncoder.
    """
    key: str
    """The key this encoder processes."""
    input_size: int
    """Dimensionality of the flat input feature."""