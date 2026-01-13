from typing import Tuple

import torch


def generate_lm_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """生成语言模型随机 token 批次。"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, targets


def generate_dlrm_batch(
    batch_size: int,
    num_dense: int,
    num_sparse: int,
    vocab_size: int,
    device: torch.device,
):
    """生成 DLRM 的稠密/稀疏特征与标签。"""
    dense = torch.randn(batch_size, num_dense, device=device)
    sparse = torch.randint(0, vocab_size, (batch_size, num_sparse), device=device)
    targets = torch.randint(0, 2, (batch_size,), device=device).float()
    return dense, sparse, targets


def generate_cv_batch(batch_size: int, num_classes: int, image_size: int, device: torch.device):
    """生成视觉模型随机图像与分类标签。"""
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    return images, targets
