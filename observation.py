from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import torch
from torch import profiler


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


@dataclass
class ProfilerObservationConfig:
    """Profiler 观测配置，控制采样频率与窗口统计。"""

    enabled: bool = True
    window_size: int = 50
    schedule_wait: int = 1
    schedule_warmup: int = 1
    schedule_active: int = 2
    schedule_repeat: int = 0


class ProfilerObservation:
    """基于 PyTorch Profiler 的观测模块，用于采样运行时统计。"""

    def __init__(self, cfg: ProfilerObservationConfig):
        self.cfg = cfg
        self._records: Deque[Dict[str, Optional[float]]] = deque(maxlen=cfg.window_size)
        self._last_step_start: Optional[float] = None
        self._latest_trace: Dict[str, Optional[float]] = {}
        self._profiler: Optional[profiler.profile] = None
        if cfg.enabled:
            activities = [profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(profiler.ProfilerActivity.CUDA)
            schedule = profiler.schedule(
                wait=cfg.schedule_wait,
                warmup=cfg.schedule_warmup,
                active=cfg.schedule_active,
                repeat=cfg.schedule_repeat,
            )
            self._profiler = profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=self.on_profiler_trace,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self._profiler.__enter__()

    def _percentile(self, values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round((pct / 100.0) * (len(ordered) - 1)))
        return ordered[min(max(idx, 0), len(ordered) - 1)]

    def _variance(self, values: List[float]) -> Optional[float]:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    def _trend(self, values: Iterable[float], tolerance: float = 0.0) -> str:
        values = list(values)
        if len(values) < 2:
            return "stable"
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        if all(diff >= -tolerance for diff in diffs) and any(diff > tolerance for diff in diffs):
            return "increasing"
        if all(diff <= tolerance for diff in diffs) and any(diff < -tolerance for diff in diffs):
            return "decreasing"
        return "stable"

    def _extract(self, key: str) -> List[float]:
        return [item[key] for item in self._records if item.get(key) is not None]

    def step_begin(self) -> None:
        """记录训练步开始时间。"""
        if not self.cfg.enabled:
            return
        self._last_step_start = time.perf_counter()

    def step_end(self) -> None:
        """结束训练步并推进 profiler 调度。"""
        if not self.cfg.enabled:
            return
        if self._last_step_start is not None:
            wall_time = time.perf_counter() - self._last_step_start
            self._latest_trace["step_time"] = wall_time
        if self._profiler is not None:
            self._profiler.step()

    def on_profiler_trace(self, prof: profiler.profile) -> None:
        """处理 profiler trace，提取 checkpoint 与训练时间统计。"""
        if not self.cfg.enabled:
            return
        events = {evt.key: evt for evt in prof.key_averages()}
        def _cpu_us(key: str) -> Optional[float]:
            evt = events.get(key)
            return evt.cpu_time_total if evt is not None else None

        def _cuda_us(key: str) -> Optional[float]:
            evt = events.get(key)
            return evt.cuda_time_total if evt is not None else None

        step_cpu = _cpu_us("train_step")
        step_cuda = _cuda_us("train_step")
        optimizer_cpu = _cpu_us("optimizer_step")
        serialize_cpu = _cpu_us("checkpoint_serialize")
        write_cpu = _cpu_us("checkpoint_write")

        step_time = step_cpu / 1e6 if step_cpu is not None else None
        compute_time = step_cuda / 1e6 if step_cuda is not None else None
        optimizer_time = optimizer_cpu / 1e6 if optimizer_cpu is not None else None
        checkpoint_serialize_time = serialize_cpu / 1e6 if serialize_cpu is not None else None
        checkpoint_write_time = write_cpu / 1e6 if write_cpu is not None else None
        checkpoint_total = None
        if checkpoint_serialize_time is not None or checkpoint_write_time is not None:
            checkpoint_total = (checkpoint_serialize_time or 0.0) + (checkpoint_write_time or 0.0)
        cpu_overhead = None
        if step_time is not None and compute_time is not None:
            cpu_overhead = max(step_time - compute_time, 0.0)

        overlap_ratio = None
        if step_time is not None and compute_time is not None and checkpoint_write_time is not None:
            overlapped = max(compute_time + checkpoint_write_time - step_time, 0.0)
            overlap_ratio = overlapped / checkpoint_write_time if checkpoint_write_time else 0.0

        snapshot = {
            "step_time": step_time or self._latest_trace.get("step_time"),
            "compute_time": compute_time,
            "optimizer_time": optimizer_time,
            "cpu_overhead_time": cpu_overhead,
            "checkpoint_serialize_time": checkpoint_serialize_time,
            "checkpoint_write_time": checkpoint_write_time,
            "checkpoint_total_time": checkpoint_total,
            "checkpoint_overlap_ratio": overlap_ratio,
        }
        self._latest_trace = snapshot
        self._records.append(snapshot)

    def window_stats(self) -> Dict[str, Optional[float]]:
        """返回窗口统计量，供决策层使用。"""
        if not self._records:
            return {}
        step_times = self._extract("step_time")
        compute_times = self._extract("compute_time")
        ckpt_write = self._extract("checkpoint_write_time")
        ckpt_total = self._extract("checkpoint_total_time")
        ratios = []
        for step_time, compute_time in zip(step_times, compute_times):
            if step_time:
                ratios.append(compute_time / step_time if compute_time is not None else 0.0)
        stats = {
            "step_time_mean": sum(step_times) / len(step_times) if step_times else None,
            "step_time_var": self._variance(step_times),
            "step_time_p50": self._percentile(step_times, 50),
            "step_time_p95": self._percentile(step_times, 95),
            "step_time_p99": self._percentile(step_times, 99),
            "compute_time_mean": sum(compute_times) / len(compute_times) if compute_times else None,
            "compute_time_p95": self._percentile(compute_times, 95),
            "compute_time_p99": self._percentile(compute_times, 99),
            "checkpoint_write_p95": self._percentile(ckpt_write, 95),
            "checkpoint_write_p99": self._percentile(ckpt_write, 99),
            "checkpoint_total_mean": sum(ckpt_total) / len(ckpt_total) if ckpt_total else None,
            "compute_step_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
            "step_time_trend": self._trend(step_times),
            "compute_ratio_trend": self._trend(ratios),
        }
        if stats["step_time_mean"]:
            stats["progress_rate_steps_per_s"] = 1.0 / stats["step_time_mean"]
        else:
            stats["progress_rate_steps_per_s"] = None
        return stats

    def snapshot(self) -> Dict[str, Optional[float]]:
        """返回最新一批 trace 的快照与窗口统计。"""
        snapshot = dict(self._latest_trace)
        snapshot.update(self.window_stats())
        return snapshot

    def close(self) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
