import torch
from torch import nn
from torchrl.modules import MLP
from torchrl.data.utils import DEVICE_TYPING

from .configs import (
    SequentialEncoderConfig,
    SequentialEncoderInput,
    FlatEncoderConfig,
    FlatEncoderInput,
)
from .heads import SequentialEncoder, FlatEncoder
from .blocks import CentralizedAgentBlock, SharedAgentBlock, PerAgentBlock


class _BaseFlexModule(nn.Module):
    def __init__(
        self,
        sequential_config: SequentialEncoderConfig,
        sequential_inputs: list[SequentialEncoderInput],
        flat_inputs: list[FlatEncoderInput],
        flat_config: FlatEncoderConfig,
        mix_layer_depth: int,
        mix_layer_num_cells: int,
        mix_activation_class: type[nn.Module] | None,
        output_dim: int,
        n_agents: int,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        self.sequential_inputs = sequential_inputs
        self.sequential_config = sequential_config
        self.flat_inputs = flat_inputs
        self.flat_config = flat_config
        self.output_dim = output_dim
        self.mix_layer_depth = mix_layer_depth
        self.mix_layer_num_cells = mix_layer_num_cells
        self.mix_activation_class = mix_activation_class if mix_activation_class is not None else nn.Tanh
        self.n_agents = n_agents
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def _pre_forward_check(self, inputs):
        """Validate the structure, shape and device of the provided observation tensors."""
        for config in self.sequential_inputs:
            if config.key not in inputs:
                raise KeyError(f"Sequential key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.input_size:
                raise ValueError(
                    f"Sequential input '{config.key}' last dimension must be {config.input_size}, got {inputs[config.key].shape[-1]}."
                )

        for config in self.flat_inputs:
            if config.key not in inputs:
                raise KeyError(f"Flat key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.input_size:
                raise ValueError(
                    f"Flat input '{config.key}' last dimension must be {config.input_size}, got {inputs[config.key].shape[-1]}."
                )

        input_keys = set(inputs.keys())
        expected_keys = {config.key for config in self.sequential_inputs + self.flat_inputs}
        if input_keys != expected_keys:
            raise KeyError(
                f"Input keys do not match expected keys. Expected: {expected_keys}, got: {input_keys}."
            )

        batch_dim = next(iter(inputs.values())).shape[0]
        for key, tensor in inputs.items():
            if tensor.shape[0] != batch_dim:
                raise ValueError(
                    f"All input tensors must have the same batch size. Tensor '{key}' has batch size {tensor.shape[0]}, expected {batch_dim}."
                )

    def _build_sequence_head(self, seq_input: SequentialEncoderInput) -> SequentialEncoder:
        return SequentialEncoder(seq_input, self.sequential_config, device=self.device)

    def _build_flat_head(self, flat_input: FlatEncoderInput, input_dim: int | None = None) -> FlatEncoder:
        return FlatEncoder(flat_input, self.flat_config, input_dim=input_dim, device=self.device)

    def _build_mix_layer(self, mix_input_dim: int) -> nn.Module:
        return MLP(
            in_features=mix_input_dim,
            out_features=self.output_dim,
            depth=self.mix_layer_depth,
            num_cells=self.mix_layer_num_cells,
            activation_class=self.mix_activation_class,
            device=self.device,
        )


class _CentralizedFlexModule(_BaseFlexModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        seq_heads: dict[str, nn.Module] = {}
        for seq_input in self.sequential_inputs:
            seq_heads[seq_input.key] = self._build_sequence_head(seq_input)

        for flat_input in self.flat_inputs:
            seq_heads[flat_input.key] = self._build_sequence_head(
                SequentialEncoderInput(input_size=flat_input.input_size, key=flat_input.key)
            )

        mix_input_dim = self.sequential_config.embed_dim * (len(self.sequential_inputs) + len(self.flat_inputs))
        mix_layer = self._build_mix_layer(mix_input_dim)
        self.block = CentralizedAgentBlock(seq_heads, mix_layer, self.n_agents)

    def forward(self, observation: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        for flat_input in self.flat_inputs:
            key = flat_input.key
            obs_tensor = observation[key]
            observation[key] = obs_tensor.unsqueeze(-2)

        output = self.block(observation, mask)
        return output.unsqueeze(-2).expand(*output.shape[:-1], self.n_agents, output.shape[-1])


class _SharedFlexModule(_BaseFlexModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        seq_heads: dict[str, nn.Module] = {}
        for seq_input in self.sequential_inputs:
            seq_heads[seq_input.key] = self._build_sequence_head(seq_input)

        flat_heads: dict[str, nn.Module] = {}
        for flat_input in self.flat_inputs:
            flat_heads[flat_input.key] = self._build_flat_head(flat_input)

        mix_input_dim = (
            self.sequential_config.embed_dim * len(self.sequential_inputs)
            + self.flat_config.embed_dim * len(self.flat_inputs)
        )
        mix_layer = self._build_mix_layer(mix_input_dim)
        self.block = SharedAgentBlock(seq_heads, flat_heads, mix_layer, self.n_agents)

    def forward(self, observation: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(observation, mask)


class _PerAgentFlexModule(_BaseFlexModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        mix_input_dim = (
            self.sequential_config.embed_dim * len(self.sequential_inputs)
            + self.flat_config.embed_dim * len(self.flat_inputs)
        )

        self.agent_blocks = nn.ModuleList()
        for _ in range(self.n_agents):
            seq_heads: dict[str, nn.Module] = {}
            for seq_input in self.sequential_inputs:
                seq_heads[seq_input.key] = self._build_sequence_head(seq_input)

            flat_heads: dict[str, nn.Module] = {}
            for flat_input in self.flat_inputs:
                flat_heads[flat_input.key] = self._build_flat_head(flat_input)

            mix_layer = self._build_mix_layer(mix_input_dim)
            self.agent_blocks.append(PerAgentBlock(seq_heads, flat_heads, mix_layer))

    def forward(self, observation: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for agent_idx, block in enumerate(self.agent_blocks):
            outputs.append(block(observation, agent_idx, mask))
        return torch.stack(outputs, dim=-2)


class MultiAgentFlexModule(nn.Module):
    """
    Flexible multi-agent encoder that dispatches to specialized modules based on centralized/share_params.
    """
    def __init__(
        self,
        sequential_config: SequentialEncoderConfig,
        sequential_inputs: list[SequentialEncoderInput],
        flat_inputs: list[FlatEncoderInput],
        flat_config: FlatEncoderConfig,
        mix_layer_depth: int,
        mix_layer_num_cells: int,
        mix_activation_class: type[nn.Module] | None,
        output_dim: int,
        n_agents: int,
        centralized: bool | None = None,
        share_params: bool | None = None,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        self.sequential_inputs = sequential_inputs
        self.flat_inputs = flat_inputs
        self.centralized = centralized if centralized is not None else False
        self.share_params = share_params

        if self.centralized:
            self.share_params = False

        base_kwargs = dict(
            sequential_config=sequential_config,
            sequential_inputs=sequential_inputs,
            flat_inputs=flat_inputs,
            flat_config=flat_config,
            mix_layer_depth=mix_layer_depth,
            mix_layer_num_cells=mix_layer_num_cells,
            mix_activation_class=mix_activation_class,
            output_dim=output_dim,
            n_agents=n_agents,
            device=device,
        )

        if self.centralized:
            self.impl = _CentralizedFlexModule(**base_kwargs)
        elif self.share_params:
            self.impl = _SharedFlexModule(**base_kwargs)
        else:
            self.impl = _PerAgentFlexModule(**base_kwargs)

    def _pre_forward_check(self, inputs):
        for config in self.sequential_inputs:
            if config.key not in inputs:
                raise KeyError(f"Sequential key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.input_size:
                raise ValueError(
                    f"Sequential input '{config.key}' last dimension must be {config.input_size}, got {inputs[config.key].shape[-1]}."
                )

        for config in self.flat_inputs:
            if config.key not in inputs:
                raise KeyError(f"Flat key '{config.key}' not found in inputs.")
            if inputs[config.key].shape[-1] != config.input_size:
                raise ValueError(
                    f"Flat input '{config.key}' last dimension must be {config.input_size}, got {inputs[config.key].shape[-1]}."
                )

        input_keys = set(inputs.keys())
        expected_keys = {config.key for config in self.sequential_inputs + self.flat_inputs}
        if input_keys != expected_keys:
            raise KeyError(
                f"Input keys do not match expected keys. Expected: {expected_keys}, got: {input_keys}."
            )

        batch_dim = next(iter(inputs.values())).shape[0]
        for key, tensor in inputs.items():
            if tensor.shape[0] != batch_dim:
                raise ValueError(
                    f"All input tensors must have the same batch size. Tensor '{key}' has batch size {tensor.shape[0]}, expected {batch_dim}."
                )

    def forward(self, mask: torch.Tensor | None = None, **observation: torch.Tensor) -> torch.Tensor:
        self._pre_forward_check(observation)
        return self.impl(observation, mask)
