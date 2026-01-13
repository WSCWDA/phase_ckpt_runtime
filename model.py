import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class MiniTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        attn_mask = self._causal_mask(seq_len, input_ids.device)
        x = self.encoder(x, mask=attn_mask)
        logits = self.lm_head(x)
        return logits
