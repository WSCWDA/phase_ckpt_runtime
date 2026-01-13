from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


@dataclass
class TransformerConfig:
    """Transformer 配置参数。"""
    vocab_size: int = 32000
    max_seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class RMSNorm(nn.Module):
    """RMSNorm 层（LLaMA 风格）。"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力。"""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """前馈网络（FFN）。"""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLU(nn.Module):
    """SwiGLU 激活前馈层。"""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * 4)
        self.w2 = nn.Linear(d_model, d_model * 4)
        self.w3 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x)))


@dataclass
class DLRMConfig:
    """DLRM 配置参数。"""
    num_dense: int = 8
    num_sparse: int = 8
    vocab_size: int = 1000
    embed_dim: int = 32
    bottom_mlp: Tuple[int, ...] = (64, 32)
    top_mlp: Tuple[int, ...] = (64, 32, 1)
