from __future__ import annotations

from typing import List

import torch
from torch import nn

from .common import DLRMConfig


class DLRM(nn.Module):
    """简化版 DLRM（bottom MLP + dot interaction + top MLP）。"""
    def __init__(self, config: DLRMConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.embed_dim) for _ in range(config.num_sparse)]
        )
        self.bottom_mlp = self._make_mlp([config.num_dense] + list(config.bottom_mlp))
        num_features = 1 + config.num_sparse
        interaction_dim = num_features * (num_features - 1) // 2 + config.bottom_mlp[-1]
        self.top_mlp = self._make_mlp([interaction_dim] + list(config.top_mlp))

    def _make_mlp(self, dims: List[int]) -> nn.Sequential:
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _interaction(self, features: torch.Tensor) -> torch.Tensor:
        """计算特征两两 dot interaction。"""
        batch_size, num_features, dim = features.shape
        interactions = torch.bmm(features, features.transpose(1, 2))
        tri_indices = torch.triu_indices(num_features, num_features, offset=1, device=features.device)
        interacted = interactions[:, tri_indices[0], tri_indices[1]]
        return interacted

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        dense_out = self.bottom_mlp(dense)
        sparse_embeds = [emb(sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        features = torch.stack([dense_out] + sparse_embeds, dim=1)
        interactions = self._interaction(features)
        concat = torch.cat([dense_out, interactions], dim=-1)
        return self.top_mlp(concat).squeeze(-1)
