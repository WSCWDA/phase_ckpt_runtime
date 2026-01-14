import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemWriter, async_save, save


@dataclass
class PendingCheckpoint:
    """异步 checkpoint 的待完成任务。"""
    step: int
    start_time: float
    bytes_estimate: int
    future: Any
    path: str


class CheckpointManager:
    """旧版 checkpoint 管理器（保留备用）。"""
    def __init__(
        self,
        base_dir: str,
        mode: str,
        io_delay: float = 0.0,
        phase_inference: Optional[Any] = None,
    ):
        self.base_dir = base_dir
        self.mode = mode
        self.io_delay = io_delay
        self.phase_inference = phase_inference
        self.pending: List[PendingCheckpoint] = []
        self._initialized_pg = False
        self.num_issued = 0
        self.num_completed = 0
        self.last_persisted_step = 0
        self.last_completed_latency_s = 0.0
        self.max_ckpt_latency_s = 0.0
        os.makedirs(self.base_dir, exist_ok=True)

    def _ensure_distributed(self):
        """初始化分布式进程组（单进程也可）。"""
        if dist.is_initialized():
            return
        if torch.cuda.is_available():
            backend = "cpu:gloo,cuda:nccl"
        else:
            backend = "gloo"
        init_file = tempfile.NamedTemporaryFile(delete=False)
        init_method = f"file://{init_file.name}"
        dist.init_process_group(backend=backend, rank=0, world_size=1, init_method=init_method)
        self._initialized_pg = True

    def _checkpoint_path(self, step: int) -> str:
        return os.path.join(self.base_dir, f"step_{step:06d}")

    def trigger(self, state: Dict[str, Any], step: int, bytes_estimate: int) -> Dict[str, Any]:
        """触发一次 checkpoint 保存。"""
        path = self._checkpoint_path(step)
        metadata = {
            "ckpt_bytes": bytes_estimate,
            "ckpt_latency": 0.0,
        }
        mode = self.mode
        if self.mode == "phase" and self.phase_inference is not None:
            mode = "async" if self.phase_inference.should_enable_async_ckpt() else "sync"

        if mode == "sync":
            start = time.time()
            writer = FileSystemWriter(path)
            save(state, storage_writer=writer)
            if self.io_delay > 0:
                time.sleep(self.io_delay)
            metadata["ckpt_latency"] = time.time() - start
            self.num_issued += 1
            self.num_completed += 1
            self.last_persisted_step = max(self.last_persisted_step, step)
            self.last_completed_latency_s = metadata["ckpt_latency"]
            self.max_ckpt_latency_s = max(self.max_ckpt_latency_s, metadata["ckpt_latency"])
            return metadata

        if mode != "async":
            raise ValueError(f"Unsupported checkpoint mode: {mode}")

        self._ensure_distributed()
        start = time.time()
        writer = FileSystemWriter(path)
        future = async_save(state, storage_writer=writer)
        self.num_issued += 1
        pending = PendingCheckpoint(
            step=step,
            start_time=start,
            bytes_estimate=bytes_estimate,
            future=future,
            path=path,
        )
        self.pending.append(pending)
        return metadata

    def poll_completed(self) -> List[Dict[str, Any]]:
        """轮询已完成的异步保存。"""
        completed: List[Dict[str, Any]] = []
        remaining: List[PendingCheckpoint] = []
        for item in self.pending:
            done = False
            try:
                done = item.future.done()
            except AttributeError:
                try:
                    item.future.wait()
                    done = True
                except Exception:
                    done = False

            if done:
                try:
                    item.future.wait()
                except AttributeError:
                    pass
                if self.io_delay > 0:
                    time.sleep(self.io_delay)
                latency = time.time() - item.start_time
                self.num_completed += 1
                self.last_persisted_step = max(self.last_persisted_step, item.step)
                self.last_completed_latency_s = latency
                self.max_ckpt_latency_s = max(self.max_ckpt_latency_s, latency)
                completed.append(
                    {
                        "step": item.step,
                        "ckpt_latency": latency,
                        "ckpt_bytes": item.bytes_estimate,
                        "path": item.path,
                    }
                )
            else:
                remaining.append(item)
        self.pending = remaining
        return completed

    def finalize(self) -> None:
        """等待所有异步保存完成并关闭进程组。"""
        while self.pending:
            self.poll_completed()
        if self._initialized_pg and dist.is_initialized():
            dist.destroy_process_group()

    def queue_depth(self) -> int:
        return len(self.pending)

    def staleness(self, current_step: int) -> int:
        if not self.pending:
            return 0
        oldest = min(item.step for item in self.pending)
        return current_step - oldest
