import torch
from torch import nn


class CentralizedAgentBlock(nn.Module):
    """Processes centralized observations for all agents in a single block."""
    def __init__(self, seq_heads: dict[str, nn.Module], mix_layer: nn.Module, n_agents: int):
        super().__init__()
        self.seq_heads = nn.ModuleDict(seq_heads)
        self.mix_layer = mix_layer
        self.n_agents = n_agents

    def forward(self, observation: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        head_outputs: list[torch.Tensor] = []

        for key, head in self.seq_heads.items():
            seq_input = observation[key]
            mask_expanded: torch.Tensor | None = None

            agent_idx_tensor = torch.arange(self.n_agents, device=seq_input.device).repeat_interleave(seq_input.shape[-2])
            agent_idx_tensor = agent_idx_tensor.unsqueeze(0).expand(seq_input.shape[0], -1).unsqueeze(-1)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, seq_input.shape[-2]).flatten(-2)

            seq_input = seq_input.flatten(start_dim=-3, end_dim=-2)
            seq_output = head(seq_input, agent_idx_tensor, mask_expanded)
            head_outputs.append(seq_output)

        agent_input = torch.cat(head_outputs, dim=-1)
        return self.mix_layer(agent_input)


class SharedAgentBlock(nn.Module):
    """Processes per-agent observations with shared parameters across agents."""
    def __init__(self, seq_heads: dict[str, nn.Module], flat_heads: dict[str, nn.Module], mix_layer: nn.Module, n_agents: int):
        super().__init__()
        self.seq_heads = nn.ModuleDict(seq_heads)
        self.flat_heads = nn.ModuleDict(flat_heads)
        self.mix_layer = mix_layer
        self.n_agents = n_agents

    def forward(self, observation: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        head_outputs: list[torch.Tensor] = []

        for key, head in self.seq_heads.items():
            seq_input = observation[key]
            original_shape = seq_input.shape
            mask_expanded: torch.Tensor | None = None

            seq_input = seq_input.flatten(start_dim=0, end_dim=-3)
            agent_idx_repeats = seq_input.shape[0] // self.n_agents
            agent_idx_tensor = torch.arange(self.n_agents, device=seq_input.device).repeat(agent_idx_repeats)
            agent_idx_tensor = agent_idx_tensor.unsqueeze(-1)

            if mask is not None:
                mask_expanded = mask.reshape(-1, 1).expand(-1, seq_input.shape[-2])

            seq_output = head(seq_input, agent_idx_tensor, mask_expanded)

            leading_batch_dims = original_shape[:-3]
            seq_output = seq_output.view(*leading_batch_dims, self.n_agents, seq_output.shape[-1])
            head_outputs.append(seq_output)

        for key, head in self.flat_heads.items():
            flat_input = observation[key]
            flat_output = head(flat_input)
            head_outputs.append(flat_output)

        agent_input = torch.cat(head_outputs, dim=-1)
        return self.mix_layer(agent_input)


class PerAgentBlock(nn.Module):
    """Processes per-agent observations with independent parameters per agent."""
    def __init__(self, seq_heads: dict[str, nn.Module], flat_heads: dict[str, nn.Module], mix_layer: nn.Module):
        super().__init__()
        self.seq_heads = nn.ModuleDict(seq_heads)
        self.flat_heads = nn.ModuleDict(flat_heads)
        self.mix_layer = mix_layer

    def forward(self, observation: dict[str, torch.Tensor], agent_idx: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        head_outputs: list[torch.Tensor] = []

        for key, head in self.seq_heads.items():
            seq_input = observation[key]
            mask_expanded: torch.Tensor | None = None

            seq_input = seq_input.select(dim=-3, index=agent_idx)
            agent_idx_tensor = torch.full_like(seq_input[..., :1, 0:1], agent_idx)

            if mask is not None:
                mask_agent = mask.select(dim=-1, index=agent_idx)
                mask_expanded = mask_agent.unsqueeze(-1).expand(*mask_agent.shape, seq_input.shape[-2])

            seq_output = head(seq_input, agent_idx_tensor, mask_expanded)
            head_outputs.append(seq_output)

        for key, head in self.flat_heads.items():
            flat_input = observation[key]
            flat_input = flat_input.select(dim=-2, index=agent_idx)
            flat_output = head(flat_input)
            head_outputs.append(flat_output)

        agent_input = torch.cat(head_outputs, dim=-1)
        return self.mix_layer(agent_input)