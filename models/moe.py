from __future__ import annotations

import torch
from torch import nn

from .common import MultiHeadSelfAttention, RMSNorm, TransformerConfig


class Expert(nn.Module):
    """MoE 专家网络。"""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """MoE 路由与专家计算层。"""
    def __init__(self, d_model: int, num_experts: int, top_k: int, dropout: float):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, dropout) for _ in range(num_experts)])
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.softmax(self.router(x), dim=-1)
        topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[..., k]
            weight = topk_vals[..., k].unsqueeze(-1)
            expert_out = torch.zeros_like(x)
            for expert_id, expert in enumerate(self.experts):
                mask = idx == expert_id
                if mask.any():
                    expert_out[mask] = expert(x[mask])
            output += weight * expert_out

        batch_tokens = x.shape[0] * x.shape[1]
        dispatch = torch.zeros_like(gates)
        for k in range(self.top_k):
            dispatch.scatter_(-1, topk_idx[..., k : k + 1], 1.0)
        mean_gates = gates.mean(dim=(0, 1))
        mean_dispatch = dispatch.mean(dim=(0, 1))
        aux_loss = torch.sum(mean_gates * mean_dispatch) * self.experts.__len__()
        aux_loss = aux_loss / max(batch_tokens, 1)
        return self.dropout(output), aux_loss


class MoEBlock(nn.Module):
    """带 MoE 的 Transformer Block。"""
    def __init__(self, d_model: int, n_heads: int, dropout: float, num_experts: int, top_k: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, top_k, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), mask)
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss


class MoETransformerLM(nn.Module):
    """MoE 版本的 Decoder-only LM。"""
    def __init__(self, config: TransformerConfig, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [MoEBlock(config.d_model, config.n_heads, config.dropout, num_experts, top_k) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = self._causal_mask(seq_len, input_ids.device)
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = block(x, mask)
            aux_losses.append(aux_loss)
        x = self.norm(x)
        total_aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=x.device)
        return self.lm_head(x), total_aux
