from __future__ import annotations

import torch
from torch import nn

from .common import FeedForward, MultiHeadSelfAttention, RMSNorm, SwiGLU, TransformerConfig


class TransformerBlock(nn.Module):
    """GPT 风格 Transformer Block。"""
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rmsnorm: bool):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class LlamaBlock(nn.Module):
    """LLaMA 风格 Block（RMSNorm + SwiGLU）。"""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer LM。"""
    def __init__(self, config: TransformerConfig, use_rmsnorm: bool = False, llama_style: bool = False):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        block_cls = LlamaBlock if llama_style else TransformerBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(config.d_model, config.n_heads, config.dropout, use_rmsnorm)
                if not llama_style
                else block_cls(config.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model) if (use_rmsnorm or llama_style) else nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        mask = self._causal_mask(seq_len, input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.lm_head(x)
