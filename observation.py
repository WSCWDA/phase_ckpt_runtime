from __future__ import annotations

"""Observation subsystem.

Lightweight monitoring is preferred to avoid perturbing training throughput.
PyTorch Profiler is optional and used only for intermittent calibration, not as
the primary signal source. Observations are intentionally stale: policy and
phase decisions use windowed history rather than current-step precision.
"""

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Protocol

import torch
from torch import profiler

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import dcgm_agent
    import dcgm_fields
    import dcgm_structs
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dcgm_agent = None
    dcgm_fields = None
    dcgm_structs = None


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
class ObservationEvent:
    """训练线程发出的轻量事件（可丢弃）。"""

    step_id: int
    event_time_s: float
    step_time_s: float
    ckpt_write_time_s: Optional[float] = None
    queue_depth: Optional[int] = None
    last_persisted_step: Optional[int] = None
    staleness_steps: Optional[int] = None


class ObservationBackend(Protocol):
    """Observation backend interface."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def poll(self) -> Dict[str, Optional[float]]: ...


@dataclass
class ProfilerObservationConfig:
    """Profiler 观测配置，控制采样频率与窗口统计。"""

    enabled: bool = True
    aggregate_stats: bool = False
    window_size: int = 50
    resource_poll_interval_s: float = 1.0
    dcgm_enabled: bool = False
    dcgm_poll_interval_s: float = 2.0
    dcgm_window_size: int = 60
    schedule_wait: int = 1
    schedule_warmup: int = 1
    schedule_active: int = 2
    schedule_repeat: int = 0


class LightweightRuntimeBackend:
    """轻量观测后端：只消费训练事件并更新滑窗统计。"""

    def __init__(self, window_size: int):
        self._records: Deque[Dict[str, Optional[float]]] = deque(maxlen=window_size)
        self._latest: Dict[str, Optional[float]] = {}

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def update(self, event: ObservationEvent) -> None:
        record = {
            "event_time_s": event.event_time_s,
            "step_time": event.step_time_s,
            "checkpoint_write_time": event.ckpt_write_time_s,
            "queue_depth": event.queue_depth,
            "last_persisted_step": event.last_persisted_step,
            "staleness_steps": event.staleness_steps,
        }
        self._latest = record
        self._records.append(record)

    def poll(self) -> Dict[str, Optional[float]]:
        return dict(self._latest)

    def window_records(self) -> Deque[Dict[str, Optional[float]]]:
        return self._records


class ResourceMonitorBackend:
    """资源监控后端：低频轮询 CPU/GPU/IO 资源。"""

    def __init__(self, poll_interval_s: float):
        self._poll_interval_s = poll_interval_s
        self._last_poll = 0.0
        self._latest: Dict[str, Optional[float]] = {}

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def poll(self) -> Dict[str, Optional[float]]:
        now = time.time()
        if now - self._last_poll < self._poll_interval_s:
            return {}
        self._last_poll = now
        cpu_mem = None
        if psutil is not None:
            cpu_mem = psutil.virtual_memory().percent
        gpu_mem = None
        gpu_util = None
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.memory_allocated() / max(torch.cuda.max_memory_allocated(), 1)
            except RuntimeError:
                gpu_mem = None
        self._latest = {
            "cpu_mem_percent": cpu_mem,
            "gpu_mem_ratio": gpu_mem,
            "gpu_util_percent": gpu_util,
        }
        return dict(self._latest)


class DCGMObservationBackend:
    """DCGM 观测后端（环境级指标）。

    DCGM 提供低开销的硬件级遥测，不涉及模型语义，因此仅用于守护与趋势判断。
    采用粗粒度轮询即可，避免训练线程阻塞。
    """

    def __init__(self, poll_interval_s: float, window_size: int):
        self._poll_interval_s = poll_interval_s
        self._records: Deque[Dict[str, Optional[float]]] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._enabled = False
        self._stop_event = threading.Event()
        self._handle = None
        self._group_id = None
        self._last_counters: Dict[str, float] = {}
        self._last_poll = 0.0

    def start(self) -> None:
        if dcgm_agent is None or dcgm_fields is None or dcgm_structs is None:
            return
        try:
            dcgm_agent.dcgmInit()
            self._handle = dcgm_agent.dcgmStartEmbedded(dcgm_structs.DCGM_OPERATION_MODE_AUTO)
            self._group_id = dcgm_agent.dcgmGroupCreate(self._handle, dcgm_structs.DCGM_GROUP_EMPTY, "obs")
            gpu_ids = dcgm_agent.dcgmGetAllSupportedDevices(self._handle)
            if gpu_ids:
                dcgm_agent.dcgmGroupAddDevice(self._handle, self._group_id, gpu_ids[0])
            self._enabled = True
            self._thread.start()
        except Exception:
            self._enabled = False

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop_event.set()
        self._thread.join()
        try:
            if self._group_id is not None:
                dcgm_agent.dcgmGroupDestroy(self._handle, self._group_id)
            if self._handle is not None:
                dcgm_agent.dcgmShutdown()
        except Exception:
            pass

    def poll(self) -> Dict[str, Optional[float]]:
        with self._lock:
            if not self._records:
                return {}
            latest = dict(self._records[-1])
        return latest

    def window_stats(self) -> Dict[str, Optional[float]]:
        with self._lock:
            records = list(self._records)
        if not records:
            return {}
        stats: Dict[str, Optional[float]] = {}
        for key in records[-1].keys():
            values = [r[key] for r in records if r.get(key) is not None]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = None
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            stats[f"dcgm_{key}_mean"] = mean
            stats[f"dcgm_{key}_var"] = var
        return stats

    def _worker(self) -> None:
        fields = [
            dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
            dcgm_fields.DCGM_FI_DEV_SM_ACTIVE,
            dcgm_fields.DCGM_FI_DEV_TENSOR_ACTIVE,
            dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
            dcgm_fields.DCGM_FI_DEV_FB_USED,
            dcgm_fields.DCGM_FI_DEV_FB_FREE,
            dcgm_fields.DCGM_FI_DEV_PCIE_TX_BYTES,
            dcgm_fields.DCGM_FI_DEV_PCIE_RX_BYTES,
            dcgm_fields.DCGM_FI_DEV_NVLINK_TX_BYTES,
            dcgm_fields.DCGM_FI_DEV_NVLINK_RX_BYTES,
            dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
            dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
        ]
        while not self._stop_event.is_set():
            now = time.time()
            if now - self._last_poll < self._poll_interval_s:
                time.sleep(0.1)
                continue
            self._last_poll = now
            try:
                values = dcgm_agent.dcgmGetLatestValuesForFields(
                    self._handle,
                    self._group_id,
                    fields,
                )
            except Exception:
                continue
            record: Dict[str, Optional[float]] = {}
            for v in values:
                if v.status != 0:
                    continue
                record[str(v.fieldId)] = float(v.value)
            # Derive bandwidth from counters.
            for key in (str(dcgm_fields.DCGM_FI_DEV_PCIE_TX_BYTES), str(dcgm_fields.DCGM_FI_DEV_PCIE_RX_BYTES)):
                if key in record:
                    last = self._last_counters.get(key)
                    if last is not None:
                        record[f"{key}_bw"] = (record[key] - last) / max(self._poll_interval_s, 1e-6)
                    self._last_counters[key] = record[key]
            with self._lock:
                self._records.append(record)


class ProfilerBackend:
    """可选 profiler 后端，仅用于校准/调试。"""

    def __init__(self, cfg: ProfilerObservationConfig, enqueue):
        self._cfg = cfg
        self._enqueue = enqueue
        self._profiler: Optional[profiler.profile] = None

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        activities = [profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(profiler.ProfilerActivity.CUDA)
        schedule = profiler.schedule(
            wait=self._cfg.schedule_wait,
            warmup=self._cfg.schedule_warmup,
            active=self._cfg.schedule_active,
            repeat=self._cfg.schedule_repeat,
        )
        self._profiler = profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=self._on_profiler_trace,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self._profiler.__enter__()

    def stop(self) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def poll(self) -> Dict[str, Optional[float]]:
        return {}

    def _on_profiler_trace(self, prof: profiler.profile) -> None:
        if not self._cfg.enabled:
            return
        events = {evt.key: evt for evt in prof.key_averages()}

        def _cpu_us(key: str) -> Optional[float]:
            evt = events.get(key)
            return evt.cpu_time_total if evt is not None else None

        def _cuda_us(key: str) -> Optional[float]:
            evt = events.get(key)
            if evt is None:
                return None
            cuda_total = getattr(evt, "cuda_time_total", None)
            if cuda_total is None:
                cuda_total = getattr(evt, "self_cuda_time_total", None)
            return cuda_total

        step_cpu = _cpu_us("train_step")
        step_cuda = _cuda_us("train_step")
        write_cpu = _cpu_us("checkpoint_write")

        step_time = step_cpu / 1e6 if step_cpu is not None else None
        compute_time = step_cuda / 1e6 if step_cuda is not None else None
        checkpoint_write_time = write_cpu / 1e6 if write_cpu is not None else None

        overlap_ratio = None
        if step_time is not None and compute_time is not None and checkpoint_write_time is not None:
            overlapped = max(compute_time + checkpoint_write_time - step_time, 0.0)
            overlap_ratio = overlapped / checkpoint_write_time if checkpoint_write_time else 0.0

        snapshot = {
            "step_time": step_time,
            "compute_time": compute_time,
            "checkpoint_write_time": checkpoint_write_time,
            "checkpoint_overlap_ratio": overlap_ratio,
        }
        self._enqueue(snapshot)


class ObservationWorker:
    """异步观测工作线程，负责统计聚合与资源采样。"""

    def __init__(self, cfg: ProfilerObservationConfig):
        self.cfg = cfg
        self._stop_sentinel = object()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=cfg.window_size * 4)
        self._lock = threading.Lock()
        self._runtime_backend = LightweightRuntimeBackend(cfg.window_size)
        self._resource_backend = ResourceMonitorBackend(cfg.resource_poll_interval_s)
        self._dcgm_backend = DCGMObservationBackend(cfg.dcgm_poll_interval_s, cfg.dcgm_window_size)
        self._latest: Dict[str, Optional[float]] = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._profiler_backend = ProfilerBackend(cfg, self._enqueue_profiler_snapshot)
        self._thread.start()

    def emit(self, event: ObservationEvent) -> None:
        """训练路径的非阻塞事件投递，队列满时直接丢弃。"""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            return

    def step_begin(self) -> None:
        """兼容接口：训练路径调用时不执行任何阻塞操作。"""
        return

    def step_end(self) -> None:
        """推进 profiler 状态机（O(1)，不做统计聚合）。"""
        self._profiler_backend.step()

    def _worker(self) -> None:
        """后台线程：消费事件、推进 profiler 并更新统计。"""
        self._runtime_backend.start()
        self._resource_backend.start()
        self._profiler_backend.start()
        if self.cfg.dcgm_enabled:
            self._dcgm_backend.start()
        while True:
            try:
                item = self._queue.get(timeout=self.cfg.resource_poll_interval_s)
            except queue.Empty:
                item = None
            if item is None:
                resource_stats = self._resource_backend.poll()
                if resource_stats:
                    with self._lock:
                        self._latest.update(resource_stats)
                if self.cfg.dcgm_enabled:
                    dcgm_stats = self._dcgm_backend.poll()
                    if dcgm_stats:
                        with self._lock:
                            self._latest.update(dcgm_stats)
                continue
            if item is self._stop_sentinel:
                break
            event = item
            self._runtime_backend.update(event)
            with self._lock:
                self._latest.update(self._runtime_backend.poll())
            self._queue.task_done()
        self._runtime_backend.stop()
        self._resource_backend.stop()
        self._profiler_backend.stop()
        if self.cfg.dcgm_enabled:
            self._dcgm_backend.stop()

    def _enqueue_profiler_snapshot(self, snapshot: Dict[str, Optional[float]]) -> None:
        if not self.cfg.aggregate_stats:
            return
        with self._lock:
            self._latest.update(snapshot)
            self._runtime_backend.window_records().append(snapshot)

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
        return [item[key] for item in self._runtime_backend.window_records() if item.get(key) is not None]

    def get_window_stats(self) -> Dict[str, Optional[float]]:
        """返回窗口统计（可能滞后）。"""
        if not self.cfg.aggregate_stats:
            return {}
        with self._lock:
            records = self._runtime_backend.window_records()
            if not records:
                return {}
            step_times = self._extract("step_time")
            compute_times = self._extract("compute_time")
            ckpt_write = self._extract("checkpoint_write_time")
            queue_depths = self._extract("queue_depth")
            staleness = self._extract("staleness_steps")
            last_persisted = self._extract("last_persisted_step")
            event_times = self._extract("event_time_s")
            ratios = []
            for step_time, compute_time in zip(step_times, compute_times):
                if step_time:
                    ratios.append(compute_time / step_time if compute_time is not None else 0.0)
            completion_rate = None
            if event_times and last_persisted:
                elapsed = max(event_times[-1] - event_times[0], 0.0)
                persisted_delta = last_persisted[-1] - last_persisted[0]
                if elapsed > 0:
                    completion_rate = persisted_delta / elapsed
            stats = {
                "step_time_mean": sum(step_times) / len(step_times) if step_times else None,
                "step_time_var": self._variance(step_times),
                "step_time_p50": self._percentile(step_times, 50),
                "step_time_p95": self._percentile(step_times, 95),
                "step_time_p99": self._percentile(step_times, 99),
                "compute_time_mean": sum(compute_times) / len(compute_times) if compute_times else None,
                "compute_time_p95": self._percentile(compute_times, 95),
                "checkpoint_write_p95": self._percentile(ckpt_write, 95),
                "checkpoint_write_p99": self._percentile(ckpt_write, 99),
                "compute_step_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
                "step_time_trend": self._trend(step_times),
                "compute_ratio_trend": self._trend(ratios),
                "queue_depth_mean": sum(queue_depths) / len(queue_depths) if queue_depths else None,
                "queue_depth_max": max(queue_depths) if queue_depths else None,
                "staleness_mean": sum(staleness) / len(staleness) if staleness else None,
                "staleness_max": max(staleness) if staleness else None,
                "ckpt_completion_rate_per_s": completion_rate,
            }
            if stats["step_time_mean"]:
                stats["progress_rate_steps_per_s"] = 1.0 / stats["step_time_mean"]
            else:
                stats["progress_rate_steps_per_s"] = None
            return stats

    def get_latest_trace(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return dict(self._latest)

    def close(self) -> None:
        self._queue.put(self._stop_sentinel)
        self._thread.join()


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
            if evt is None:
                return None
            cuda_total = getattr(evt, "cuda_time_total", None)
            if cuda_total is None:
                cuda_total = getattr(evt, "self_cuda_time_total", None)
            return cuda_total

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
