import random
from typing import Any, Dict

import torch


def capture_rng_state() -> Dict[str, Any]:
    """抓取 RNG 状态（torch + python + 可选 CUDA）。"""
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def estimate_state_bytes(state: Any) -> int:
    """粗略估算 state 占用字节数。"""
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(estimate_state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(estimate_state_bytes(v) for v in state)
    return 0
