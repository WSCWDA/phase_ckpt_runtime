from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class ObsSample:
    """单步观测样本（只做采集与聚合）。"""

    train_step: int
    elapsed_s: float
    step_time_s: float
    grad_norm: Optional[float]
    grad_nz_ratio: Optional[float]
    ckpt_completed_latency_s: Optional[float]
    queue_depth: int
    num_ckpt_issued: int
    num_ckpt_completed: int
    last_persisted_step: int
    staleness_steps: int


class ObservationBuffer:
    """滑动窗口观测缓冲区，用于统计窗口均值/最大值。"""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.records: Deque[ObsSample] = deque(maxlen=window_size)

    def update(self, sample: ObsSample) -> None:
        self.records.append(sample)

    def _percentile(self, values: list[float], pct: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round((pct / 100.0) * (len(ordered) - 1)))
        return ordered[min(max(idx, 0), len(ordered) - 1)]

    def stats(self) -> Dict[str, Optional[float]]:
        """返回窗口统计量。"""

        if not self.records:
            return {
                "step_time_mean": None,
                "step_time_p95": None,
                "step_time_p99": None,
                "grad_norm_mean": None,
                "grad_nz_ratio_mean": None,
                "ckpt_latency_mean": None,
                "ckpt_latency_p95": None,
                "ckpt_latency_p99": None,
                "queue_depth_mean": None,
                "queue_depth_max": None,
                "ckpt_completion_rate_per_s": None,
                "staleness_mean": None,
                "staleness_max": None,
            }
        step_times = [r.step_time_s for r in self.records]
        grad_norms = [r.grad_norm for r in self.records if r.grad_norm is not None]
        grad_nz = [r.grad_nz_ratio for r in self.records if r.grad_nz_ratio is not None]
        ckpt_lat = [r.ckpt_completed_latency_s for r in self.records if r.ckpt_completed_latency_s is not None]
        queue_depths = [r.queue_depth for r in self.records]
        staleness = [r.staleness_steps for r in self.records]

        step_time_mean = sum(step_times) / len(step_times)
        step_time_p95 = self._percentile(step_times, 95)
        step_time_p99 = self._percentile(step_times, 99)
        grad_norm_mean = sum(grad_norms) / len(grad_norms) if grad_norms else None
        grad_nz_ratio_mean = sum(grad_nz) / len(grad_nz) if grad_nz else None
        ckpt_latency_mean = sum(ckpt_lat) / len(ckpt_lat) if ckpt_lat else None
        ckpt_latency_p95 = self._percentile(ckpt_lat, 95)
        ckpt_latency_p99 = self._percentile(ckpt_lat, 99)
        queue_depth_mean = sum(queue_depths) / len(queue_depths) if queue_depths else None
        queue_depth_max = max(queue_depths) if queue_depths else None
        staleness_mean = sum(staleness) / len(staleness) if staleness else None
        staleness_max = max(staleness) if staleness else None

        completion_rate = None
        first = self.records[0]
        last = self.records[-1]
        elapsed_delta = max(last.elapsed_s - first.elapsed_s, 0.0)
        completed_delta = last.num_ckpt_completed - first.num_ckpt_completed
        if elapsed_delta > 0:
            completion_rate = completed_delta / elapsed_delta

        return {
            "step_time_mean": step_time_mean,
            "step_time_p95": step_time_p95,
            "step_time_p99": step_time_p99,
            "grad_norm_mean": grad_norm_mean,
            "grad_nz_ratio_mean": grad_nz_ratio_mean,
            "ckpt_latency_mean": ckpt_latency_mean,
            "ckpt_latency_p95": ckpt_latency_p95,
            "ckpt_latency_p99": ckpt_latency_p99,
            "queue_depth_mean": queue_depth_mean,
            "queue_depth_max": queue_depth_max,
            "ckpt_completion_rate_per_s": completion_rate,
            "staleness_mean": staleness_mean,
            "staleness_max": staleness_max,
        }


class AsyncObservationWorker:
    """异步观测处理器：后台更新窗口统计（不做 phase 决策）。"""

    def __init__(self, obs: ObservationBuffer, maxsize: int = 256):
        self.obs = obs
        self.queue: queue.Queue[Optional[ObsSample]] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._latest_stats: Dict[str, Optional[float]] = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, sample: ObsSample) -> None:
        """提交观测样本（队列满时阻塞）。"""

        self.queue.put(sample)

    def _worker(self) -> None:
        """后台线程：更新窗口并计算统计量。"""

        while True:
            sample = self.queue.get()
            if sample is None:
                self.queue.task_done()
                break
            self.obs.update(sample)
            stats = self.obs.stats()
            with self._lock:
                self._latest_stats = stats
            self.queue.task_done()

    def latest_stats(self) -> Dict[str, Optional[float]]:
        """返回最新的观测统计量。"""

        with self._lock:
            return dict(self._latest_stats)

    def close(self) -> None:
        """等待队列完成并关闭线程。"""

        self.queue.join()
        self.queue.put(None)
        self._thread.join()
