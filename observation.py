from __future__ import annotations

"""Lightweight asynchronous observation subsystem.

Design rationale:
- Observation is decoupled from training critical path: the training thread emits
  tiny events to a bounded queue and never blocks.
- Most computation (aggregation, telemetry polling, smoothing) runs in a daemon
  background thread.
- PyTorch profiler is optional calibration/debug backend and disabled by default.
- Returned statistics are windowed and may be stale by design, which is suitable
  for phase-level checkpoint policy decisions.
"""

import enum
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from torch import profiler as torch_profiler  # type: ignore
except Exception:  # pragma: no cover - optional in some envs
    torch_profiler = None

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import dcgm_agent  # type: ignore
    import dcgm_fields  # type: ignore
    import dcgm_structs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dcgm_agent = None
    dcgm_fields = None
    dcgm_structs = None

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility dataclasses kept for existing imports
# ---------------------------------------------------------------------------


@dataclass
class ObservationEvent:
    """Compatibility event shape used by the current training loop."""

    step_id: int
    event_time_s: float
    step_time_s: float
    ckpt_write_time_s: Optional[float] = None
    queue_depth: Optional[int] = None
    last_persisted_step: Optional[int] = None
    staleness_steps: Optional[int] = None


@dataclass
class ProfilerObservationConfig:
    """Config used by current train.py and the new observation manager."""

    enabled: bool = False  # profiler enabled
    aggregate_stats: bool = False
    window_size: int = 50
    resource_poll_interval_s: float = 1.0
    snapshot_export_interval_s: float = 0.5
    ewma_alpha: float = 0.2
    debug_ring_buffer_size: int = 256
    retain_debug_events: bool = False
    worker_sleep_interval_s: float = 0.05
    # optional telemetry backends
    telemetry_enabled: bool = True
    dcgm_enabled: bool = False
    dcgm_poll_interval_s: float = 2.0
    dcgm_window_size: int = 60
    # profiler schedule
    schedule_wait: int = 1
    schedule_warmup: int = 1
    schedule_active: int = 2
    schedule_repeat: int = 0


# ---------------------------------------------------------------------------
# New structured runtime observation interfaces
# ---------------------------------------------------------------------------


class RuntimeEventType(str, enum.Enum):
    STEP_BEGIN = "STEP_BEGIN"
    STEP_END = "STEP_END"
    OPTIMIZER_BEGIN = "OPTIMIZER_BEGIN"
    OPTIMIZER_END = "OPTIMIZER_END"
    CKPT_SUBMIT = "CKPT_SUBMIT"
    CKPT_COMPLETE = "CKPT_COMPLETE"
    RESTORE_BEGIN = "RESTORE_BEGIN"
    RESTORE_END = "RESTORE_END"
    METRIC_PUSH = "METRIC_PUSH"


@dataclass
class RuntimeEvent:
    event_type: RuntimeEventType
    timestamp: float
    step: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class TelemetryProvider(Protocol):
    def poll(self) -> Dict[str, Optional[float]]:
        ...


class NullTelemetryProvider:
    def poll(self) -> Dict[str, Optional[float]]:
        return {}


class PsutilTelemetryProvider:
    """Lightweight system telemetry.

    Prefer lightweight host-level metrics for long-running jobs; avoid expensive
    tracing. Missing metrics are returned as None.
    """

    def poll(self) -> Dict[str, Optional[float]]:
        cpu_util: Optional[float] = None
        cpu_iowait: Optional[float] = None
        mem_usage: Optional[float] = None
        if psutil is not None:
            try:
                cpu_util = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                mem_usage = vm.percent
                times = psutil.cpu_times_percent(interval=None)
                cpu_iowait = getattr(times, "iowait", None)
            except Exception:
                LOGGER.debug("psutil polling failed", exc_info=True)

        gpu_util: Optional[float] = None
        gpu_mem: Optional[float] = None
        if torch is not None and torch.cuda.is_available():
            try:
                gpu_mem = float(torch.cuda.memory_allocated())
            except Exception:
                gpu_mem = None

        return {
            "cpu_utilization": cpu_util,
            "cpu_iowait": cpu_iowait,
            "host_memory_usage": mem_usage,
            "gpu_utilization": gpu_util,
            "gpu_memory_usage": gpu_mem,
            "io_bw": None,
            "disk_write_bw": None,
            "net_bw": None,
        }


class DCGMTelemetryProvider:
    """Optional DCGM telemetry provider.

    DCGM is environment-level telemetry (device health/utilization) and is not
    model semantic information. It is advisory and may be unavailable.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._handle = None
        self._group_id = None
        self._device_id: Optional[int] = None
        self._last_poll_ts = 0.0
        self._last_counter: Dict[int, float] = {}
        if dcgm_agent is None or dcgm_fields is None or dcgm_structs is None:
            return
        try:
            dcgm_agent.dcgmInit()
            self._handle = dcgm_agent.dcgmStartEmbedded(dcgm_structs.DCGM_OPERATION_MODE_AUTO)
            self._group_id = dcgm_agent.dcgmGroupCreate(self._handle, dcgm_structs.DCGM_GROUP_EMPTY, "obs")
            devices = dcgm_agent.dcgmGetAllSupportedDevices(self._handle)
            if devices:
                self._device_id = int(devices[0])
                dcgm_agent.dcgmGroupAddDevice(self._handle, self._group_id, self._device_id)
                self._enabled = True
        except Exception:
            LOGGER.warning("DCGM init failed, disable DCGM telemetry", exc_info=True)
            self._enabled = False

    def close(self) -> None:
        if not self._enabled:
            return
        try:
            if self._group_id is not None:
                dcgm_agent.dcgmGroupDestroy(self._handle, self._group_id)
            dcgm_agent.dcgmShutdown()
        except Exception:
            LOGGER.debug("DCGM close failed", exc_info=True)

    def poll(self) -> Dict[str, Optional[float]]:
        if not self._enabled:
            return {}
        assert dcgm_fields is not None
        fields = [
            dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
            dcgm_fields.DCGM_FI_DEV_SM_ACTIVE,
            dcgm_fields.DCGM_FI_DEV_TENSOR_ACTIVE,
            dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
            dcgm_fields.DCGM_FI_DEV_FB_USED,
            dcgm_fields.DCGM_FI_DEV_FB_FREE,
            dcgm_fields.DCGM_FI_DEV_PCIE_TX_BYTES,
            dcgm_fields.DCGM_FI_DEV_PCIE_RX_BYTES,
            dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
            dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
            dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
            dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
        ]
        result: Dict[str, Optional[float]] = {}
        try:
            values = dcgm_agent.dcgmGetLatestValuesForFields(self._handle, self._group_id, fields)
            now = time.time()
            dt = max(now - self._last_poll_ts, 1e-6)
            self._last_poll_ts = now
            for v in values:
                if getattr(v, "status", 1) != 0:
                    continue
                fid = int(v.fieldId)
                val = float(v.value)
                result[f"dcgm_{fid}"] = val
                if fid in (dcgm_fields.DCGM_FI_DEV_PCIE_TX_BYTES, dcgm_fields.DCGM_FI_DEV_PCIE_RX_BYTES):
                    prev = self._last_counter.get(fid)
                    if prev is not None:
                        result[f"dcgm_{fid}_bw"] = (val - prev) / dt
                    self._last_counter[fid] = val
        except Exception:
            LOGGER.debug("DCGM poll failed", exc_info=True)
        return result


class RollingStat:
    """Bounded online statistics with optional percentile support."""

    def __init__(self, window_size: int, ewma_alpha: float):
        self.window_size = max(1, int(window_size))
        self.alpha = ewma_alpha
        self.values: Deque[float] = deque(maxlen=self.window_size)
        self._ewma: Optional[float] = None

    def update(self, value: Optional[float]) -> None:
        if value is None:
            return
        v = float(value)
        self.values.append(v)
        if self._ewma is None:
            self._ewma = v
        else:
            self._ewma = self.alpha * v + (1.0 - self.alpha) * self._ewma

    def count(self) -> int:
        return len(self.values)

    def mean(self) -> Optional[float]:
        if not self.values:
            return None
        return sum(self.values) / len(self.values)

    def variance(self) -> Optional[float]:
        if len(self.values) < 2:
            return None
        m = self.mean()
        assert m is not None
        return sum((x - m) ** 2 for x in self.values) / (len(self.values) - 1)

    def ewma(self) -> Optional[float]:
        return self._ewma

    def min(self) -> Optional[float]:
        return min(self.values) if self.values else None

    def max(self) -> Optional[float]:
        return max(self.values) if self.values else None

    def percentile(self, p: float) -> Optional[float]:
        if not self.values:
            return None
        ordered = sorted(self.values)
        idx = int(round((p / 100.0) * (len(ordered) - 1)))
        idx = min(max(idx, 0), len(ordered) - 1)
        return ordered[idx]

    def trend(self, tolerance: float = 0.0) -> str:
        vals = list(self.values)
        if len(vals) < 2:
            return "stable"
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if all(d >= -tolerance for d in diffs) and any(d > tolerance for d in diffs):
            return "increasing"
        if all(d <= tolerance for d in diffs) and any(d < -tolerance for d in diffs):
            return "decreasing"
        return "stable"


@dataclass
class ObservationSnapshot:
    global_step: int = 0
    step_time_ewma: Optional[float] = None
    step_time_p95: Optional[float] = None
    step_time_p99: Optional[float] = None
    optimizer_time_ewma: Optional[float] = None
    ckpt_write_time_ewma: Optional[float] = None
    ckpt_restore_time_ewma: Optional[float] = None
    queue_depth: Optional[float] = None
    staleness: Optional[float] = None
    gpu_util: Optional[float] = None
    cpu_util: Optional[float] = None
    cpu_iowait: Optional[float] = None
    host_memory_usage: Optional[float] = None
    io_bw: Optional[float] = None
    net_bw: Optional[float] = None
    loss_ewma: Optional[float] = None
    delta_ratio_ewma: Optional[float] = None
    exported_at: float = 0.0
    debug_counters: Dict[str, int] = field(default_factory=dict)


class ProfilerBackend:
    """Optional intermittent profiler backend.

    Kept optional and disabled by default. Errors are swallowed and converted
    into health counters so training correctness is unaffected.
    """

    def __init__(self, cfg: ProfilerObservationConfig, sink_queue: "queue.Queue[RuntimeEvent]"):
        self.cfg = cfg
        self._sink_queue = sink_queue
        self._profiler = None
        self._enabled = bool(cfg.enabled and torch_profiler is not None)

    def start(self) -> None:
        if not self._enabled:
            return
        try:
            activities = [torch_profiler.ProfilerActivity.CPU]
            if torch is not None and torch.cuda.is_available():
                activities.append(torch_profiler.ProfilerActivity.CUDA)
            schedule = torch_profiler.schedule(
                wait=self.cfg.schedule_wait,
                warmup=self.cfg.schedule_warmup,
                active=self.cfg.schedule_active,
                repeat=self.cfg.schedule_repeat,
            )
            self._profiler = torch_profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=self._on_trace_ready,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self._profiler.__enter__()
        except Exception:
            LOGGER.warning("Profiler start failed; disabling profiler backend", exc_info=True)
            self._enabled = False
            self._profiler = None

    def step(self) -> None:
        if self._profiler is None:
            return
        try:
            self._profiler.step()
        except Exception:
            LOGGER.debug("Profiler step failed", exc_info=True)

    def stop(self) -> None:
        if self._profiler is None:
            return
        try:
            self._profiler.__exit__(None, None, None)
        except Exception:
            LOGGER.debug("Profiler stop failed", exc_info=True)

    def _on_trace_ready(self, prof: Any) -> None:
        try:
            events = {evt.key: evt for evt in prof.key_averages()}

            def _cpu_us(key: str) -> Optional[float]:
                evt = events.get(key)
                return float(evt.cpu_time_total) if evt is not None else None

            def _cuda_us(key: str) -> Optional[float]:
                evt = events.get(key)
                if evt is None:
                    return None
                cuda_total = getattr(evt, "cuda_time_total", None)
                if cuda_total is None:
                    cuda_total = getattr(evt, "self_cuda_time_total", None)
                return float(cuda_total) if cuda_total is not None else None

            step_time = _cpu_us("train_step")
            compute_time = _cuda_us("train_step")
            ckpt_write_time = _cpu_us("checkpoint_write")
            overlap = None
            if step_time is not None and compute_time is not None and ckpt_write_time is not None:
                overlapped = max(compute_time + ckpt_write_time - step_time, 0.0)
                overlap = overlapped / ckpt_write_time if ckpt_write_time > 0 else 0.0

            payload = {
                "step_time": step_time / 1e6 if step_time is not None else None,
                "compute_time": compute_time / 1e6 if compute_time is not None else None,
                "checkpoint_write_time": ckpt_write_time / 1e6 if ckpt_write_time is not None else None,
                "checkpoint_overlap_ratio": overlap,
            }
            evt = RuntimeEvent(RuntimeEventType.METRIC_PUSH, timestamp=time.perf_counter(), payload=payload)
            try:
                self._sink_queue.put_nowait(evt)
            except queue.Full:
                pass
        except Exception:
            LOGGER.debug("Profiler trace parsing failed", exc_info=True)


class ObservationManager:
    """Core observation manager.

    Training threads only emit cheap events; all aggregation and telemetry runs
    in a daemon worker thread. Returned snapshots are best-effort and may lag.
    """

    def __init__(self, config: ProfilerObservationConfig):
        self.config = config
        self._queue: "queue.Queue[RuntimeEvent]" = queue.Queue(maxsize=max(16, config.window_size * 8))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._step_begin_ts: Dict[int, float] = {}
        self._opt_begin_ts: Dict[int, float] = {}
        self._ckpt_submit_ts: Dict[int, float] = {}
        self._restore_begin_ts: Dict[int, float] = {}

        self._stats = {
            "step_time": RollingStat(config.window_size, config.ewma_alpha),
            "optimizer_time": RollingStat(config.window_size, config.ewma_alpha),
            "ckpt_write_time": RollingStat(config.window_size, config.ewma_alpha),
            "ckpt_restore_time": RollingStat(config.window_size, config.ewma_alpha),
            "loss": RollingStat(config.window_size, config.ewma_alpha),
            "delta_ratio": RollingStat(config.window_size, config.ewma_alpha),
            "queue_depth": RollingStat(config.window_size, config.ewma_alpha),
            "staleness": RollingStat(config.window_size, config.ewma_alpha),
            "compute_time": RollingStat(config.window_size, config.ewma_alpha),
            "checkpoint_overlap_ratio": RollingStat(config.window_size, config.ewma_alpha),
        }

        self._latest_step = 0
        self._last_ckpt_submit_ts: Optional[float] = None
        self._last_ckpt_complete_ts: Optional[float] = None
        self._latest_durable_step: Optional[int] = None
        self._last_snapshot = ObservationSnapshot(exported_at=time.time())
        self._last_export_ts = 0.0

        self._debug_events: Deque[RuntimeEvent] = deque(maxlen=config.debug_ring_buffer_size)
        self._health = {
            "dropped_events": 0,
            "telemetry_failures": 0,
            "queue_overflow": 0,
        }

        self._telemetry_providers: List[TelemetryProvider] = []
        if config.telemetry_enabled:
            self._telemetry_providers.append(PsutilTelemetryProvider())
            if config.dcgm_enabled:
                self._telemetry_providers.append(DCGMTelemetryProvider())

        self._profiler = ProfilerBackend(config, self._queue)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._profiler.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._profiler.stop()
        for p in self._telemetry_providers:
            close = getattr(p, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    LOGGER.debug("telemetry close failed", exc_info=True)

    def close(self) -> None:
        self.stop()

    # -- training-facing event methods ------------------------------------

    def on_step_begin(self, step: int) -> None:
        self._emit(RuntimeEvent(RuntimeEventType.STEP_BEGIN, timestamp=time.perf_counter(), step=step))

    def on_step_end(self, step: int, loss: Optional[float] = None) -> None:
        payload: Dict[str, Any] = {}
        if loss is not None:
            payload["loss"] = loss
        self._emit(RuntimeEvent(RuntimeEventType.STEP_END, timestamp=time.perf_counter(), step=step, payload=payload))

    def on_optimizer_begin(self, step: int) -> None:
        self._emit(RuntimeEvent(RuntimeEventType.OPTIMIZER_BEGIN, timestamp=time.perf_counter(), step=step))

    def on_optimizer_end(self, step: int) -> None:
        self._emit(RuntimeEvent(RuntimeEventType.OPTIMIZER_END, timestamp=time.perf_counter(), step=step))

    def on_checkpoint_submit(self, step: int, tag: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {"tag": tag}
        if metadata:
            payload.update(metadata)
        self._emit(RuntimeEvent(RuntimeEventType.CKPT_SUBMIT, timestamp=time.perf_counter(), step=step, payload=payload))

    def on_checkpoint_complete(
        self,
        step: int,
        duration: Optional[float] = None,
        durable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"duration": duration, "durable": durable}
        if metadata:
            payload.update(metadata)
        self._emit(RuntimeEvent(RuntimeEventType.CKPT_COMPLETE, timestamp=time.perf_counter(), step=step, payload=payload))

    def on_restore_begin(self, step: int, path: Optional[str] = None) -> None:
        payload = {"path": path}
        self._emit(RuntimeEvent(RuntimeEventType.RESTORE_BEGIN, timestamp=time.perf_counter(), step=step, payload=payload))

    def on_restore_end(self, step: int, duration: Optional[float] = None, success: bool = True) -> None:
        payload = {"duration": duration, "success": success}
        self._emit(RuntimeEvent(RuntimeEventType.RESTORE_END, timestamp=time.perf_counter(), step=step, payload=payload))

    def push_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        self._emit(
            RuntimeEvent(
                RuntimeEventType.METRIC_PUSH,
                timestamp=time.perf_counter(),
                step=step,
                payload={name: value},
            )
        )

    def step_profiler(self) -> None:
        """Profiler step hook (cheap; no aggregation in training thread)."""
        self._profiler.step()

    # -- readout -----------------------------------------------------------

    def get_snapshot(self) -> ObservationSnapshot:
        with self._lock:
            return ObservationSnapshot(**asdict(self._last_snapshot))

    def get_window_stats(self) -> Dict[str, Optional[float]]:
        snap = self.get_snapshot()
        return {
            "step_time_mean": self._stats["step_time"].mean(),
            "step_time_var": self._stats["step_time"].variance(),
            "step_time_p50": self._stats["step_time"].percentile(50),
            "step_time_p95": self._stats["step_time"].percentile(95),
            "step_time_p99": self._stats["step_time"].percentile(99),
            "step_time_trend": self._stats["step_time"].trend(),
            "compute_time_mean": self._stats["compute_time"].mean(),
            "compute_time_p95": self._stats["compute_time"].percentile(95),
            "compute_time_p99": self._stats["compute_time"].percentile(99),
            "compute_ratio_trend": self._stats["compute_time"].trend(),
            "checkpoint_write_p95": self._stats["ckpt_write_time"].percentile(95),
            "checkpoint_write_p99": self._stats["ckpt_write_time"].percentile(99),
            "checkpoint_total_mean": self._stats["ckpt_write_time"].mean(),
            "compute_step_ratio_mean": (
                None
                if snap.step_time_ewma in (None, 0) or snap.optimizer_time_ewma is None
                else snap.optimizer_time_ewma / snap.step_time_ewma
            ),
            "progress_rate_steps_per_s": (
                None if snap.step_time_ewma in (None, 0) else 1.0 / snap.step_time_ewma
            ),
            "queue_depth_mean": self._stats["queue_depth"].mean(),
            "queue_depth_max": self._stats["queue_depth"].max(),
            "staleness_mean": self._stats["staleness"].mean(),
            "staleness_max": self._stats["staleness"].max(),
            "ckpt_completion_rate_per_s": None,
        }

    def get_latest_trace(self) -> Dict[str, Optional[float]]:
        s = self.get_snapshot()
        return {
            "event_time_s": s.exported_at,
            "step_time": s.step_time_ewma,
            "compute_time": self._stats["compute_time"].ewma(),
            "checkpoint_write_time": s.ckpt_write_time_ewma,
            "checkpoint_overlap_ratio": self._stats["checkpoint_overlap_ratio"].ewma(),
            "queue_depth": s.queue_depth,
            "last_persisted_step": float(self._latest_durable_step) if self._latest_durable_step is not None else None,
            "staleness_steps": s.staleness,
            "cpu_mem_percent": s.host_memory_usage,
            "gpu_mem_ratio": s.gpu_util,
            "gpu_util_percent": s.gpu_util,
        }

    def get_debug_events(self, limit: int = 100) -> List[RuntimeEvent]:
        with self._lock:
            return list(self._debug_events)[-limit:]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "latest_step": self._latest_step,
            "latest_durable_step": self._latest_durable_step,
            "health": dict(self._health),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._latest_step = int(state.get("latest_step", 0))
        durable = state.get("latest_durable_step", None)
        self._latest_durable_step = int(durable) if durable is not None else None
        health = state.get("health")
        if isinstance(health, dict):
            self._health.update({k: int(v) for k, v in health.items()})

    # -- internal ----------------------------------------------------------

    def _emit(self, event: RuntimeEvent) -> None:
        if self.config.retain_debug_events:
            with self._lock:
                self._debug_events.append(event)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._health["queue_overflow"] += 1

    def _run(self) -> None:
        last_telemetry_poll = 0.0
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now - last_telemetry_poll >= self.config.resource_poll_interval_s:
                self._poll_telemetry()
                last_telemetry_poll = now

            try:
                event = self._queue.get(timeout=self.config.worker_sleep_interval_s)
            except queue.Empty:
                self._maybe_export_snapshot()
                continue

            try:
                self._consume_event(event)
            except Exception:
                LOGGER.debug("observation event consume failed", exc_info=True)
            finally:
                self._maybe_export_snapshot()

    def _poll_telemetry(self) -> None:
        for provider in self._telemetry_providers:
            try:
                data = provider.poll()
                if data:
                    self._consume_telemetry(data)
            except Exception:
                self._health["telemetry_failures"] += 1
                LOGGER.debug("telemetry provider failed", exc_info=True)

    def _consume_telemetry(self, data: Dict[str, Optional[float]]) -> None:
        # telemetry fields are advisory and may be missing
        with self._lock:
            snap = self._last_snapshot
            snap.cpu_util = data.get("cpu_utilization", snap.cpu_util)
            snap.cpu_iowait = data.get("cpu_iowait", snap.cpu_iowait)
            snap.host_memory_usage = data.get("host_memory_usage", snap.host_memory_usage)
            snap.gpu_util = data.get("gpu_utilization", snap.gpu_util)
            snap.io_bw = data.get("io_bw", snap.io_bw)
            snap.net_bw = data.get("net_bw", snap.net_bw)

    def _consume_event(self, event: RuntimeEvent) -> None:
        if event.step is not None:
            self._latest_step = max(self._latest_step, event.step)

        et = event.event_type
        if et == RuntimeEventType.STEP_BEGIN:
            if event.step is not None:
                self._step_begin_ts[event.step] = event.timestamp
            return

        if et == RuntimeEventType.STEP_END:
            if event.step is not None and event.step in self._step_begin_ts:
                duration = event.timestamp - self._step_begin_ts.pop(event.step)
                self._stats["step_time"].update(duration)
            self._stats["loss"].update(_to_float(event.payload.get("loss")))
            return

        if et == RuntimeEventType.OPTIMIZER_BEGIN:
            if event.step is not None:
                self._opt_begin_ts[event.step] = event.timestamp
            return

        if et == RuntimeEventType.OPTIMIZER_END:
            if event.step is not None and event.step in self._opt_begin_ts:
                duration = event.timestamp - self._opt_begin_ts.pop(event.step)
                self._stats["optimizer_time"].update(duration)
            return

        if et == RuntimeEventType.CKPT_SUBMIT:
            if event.step is not None:
                self._ckpt_submit_ts[event.step] = event.timestamp
            self._last_ckpt_submit_ts = event.timestamp
            self._stats["queue_depth"].update(_to_float(event.payload.get("queue_depth")))
            return

        if et == RuntimeEventType.CKPT_COMPLETE:
            duration = _to_float(event.payload.get("duration"))
            if duration is None and event.step is not None and event.step in self._ckpt_submit_ts:
                duration = event.timestamp - self._ckpt_submit_ts.pop(event.step)
            self._stats["ckpt_write_time"].update(duration)
            self._last_ckpt_complete_ts = event.timestamp
            durable = bool(event.payload.get("durable", True))
            if durable and event.step is not None:
                self._latest_durable_step = max(self._latest_durable_step or 0, event.step)
            return

        if et == RuntimeEventType.RESTORE_BEGIN:
            if event.step is not None:
                self._restore_begin_ts[event.step] = event.timestamp
            return

        if et == RuntimeEventType.RESTORE_END:
            duration = _to_float(event.payload.get("duration"))
            if duration is None and event.step is not None and event.step in self._restore_begin_ts:
                duration = event.timestamp - self._restore_begin_ts.pop(event.step)
            self._stats["ckpt_restore_time"].update(duration)
            return

        if et == RuntimeEventType.METRIC_PUSH:
            for k, v in event.payload.items():
                fv = _to_float(v)
                if k in ("delta_ratio", "update_magnitude"):
                    self._stats["delta_ratio"].update(fv)
                elif k == "loss":
                    self._stats["loss"].update(fv)
                elif k == "compute_time":
                    self._stats["compute_time"].update(fv)
                elif k == "checkpoint_overlap_ratio":
                    self._stats["checkpoint_overlap_ratio"].update(fv)
                elif k == "queue_depth":
                    self._stats["queue_depth"].update(fv)
                elif k == "staleness_steps":
                    self._stats["staleness"].update(fv)
                elif k == "ckpt_write_time":
                    self._stats["ckpt_write_time"].update(fv)
                elif k == "step_time":
                    self._stats["step_time"].update(fv)
            return

    def _maybe_export_snapshot(self) -> None:
        now = time.perf_counter()
        if now - self._last_export_ts < self.config.snapshot_export_interval_s:
            return
        self._last_export_ts = now
        staleness = None
        if self._latest_durable_step is not None:
            staleness = float(max(self._latest_step - self._latest_durable_step, 0))
            self._stats["staleness"].update(staleness)

        with self._lock:
            self._last_snapshot = ObservationSnapshot(
                global_step=self._latest_step,
                step_time_ewma=self._stats["step_time"].ewma(),
                step_time_p95=self._stats["step_time"].percentile(95),
                step_time_p99=self._stats["step_time"].percentile(99),
                optimizer_time_ewma=self._stats["optimizer_time"].ewma(),
                ckpt_write_time_ewma=self._stats["ckpt_write_time"].ewma(),
                ckpt_restore_time_ewma=self._stats["ckpt_restore_time"].ewma(),
                queue_depth=self._stats["queue_depth"].mean(),
                staleness=staleness,
                gpu_util=self._last_snapshot.gpu_util,
                cpu_util=self._last_snapshot.cpu_util,
                cpu_iowait=self._last_snapshot.cpu_iowait,
                host_memory_usage=self._last_snapshot.host_memory_usage,
                io_bw=self._last_snapshot.io_bw,
                net_bw=self._last_snapshot.net_bw,
                loss_ewma=self._stats["loss"].ewma(),
                delta_ratio_ewma=self._stats["delta_ratio"].ewma(),
                exported_at=time.time(),
                debug_counters=dict(self._health),
            )


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def create_observer(config: ProfilerObservationConfig) -> ObservationManager:
    observer = ObservationManager(config)
    observer.start()
    return observer


# ---------------------------------------------------------------------------
# Compatibility wrapper for existing train.py integration
# ---------------------------------------------------------------------------


class ObservationWorker:
    """Compatibility shim preserving existing public methods."""

    def __init__(self, cfg: ProfilerObservationConfig):
        self._manager = ObservationManager(cfg)
        self._manager.start()
        self._step_start_ts: Dict[int, float] = {}

    def emit(self, event: ObservationEvent) -> None:
        # Keep event-based compatibility used by train.py hot path.
        self._manager.push_metric("step_time", event.step_time_s, step=event.step_id)
        self._manager.push_metric("ckpt_write_time", event.ckpt_write_time_s, step=event.step_id)
        self._manager.push_metric("queue_depth", event.queue_depth, step=event.step_id)
        self._manager.push_metric("staleness_steps", event.staleness_steps, step=event.step_id)
        self._manager.on_step_end(event.step_id)

    def step_begin(self) -> None:
        # Kept for external compatibility.
        return

    def step_end(self) -> None:
        # profiler step is lightweight and optional.
        self._manager.step_profiler()

    def get_window_stats(self) -> Dict[str, Optional[float]]:
        if not self._manager.config.aggregate_stats:
            return {}
        return self._manager.get_window_stats()

    def get_latest_trace(self) -> Dict[str, Optional[float]]:
        return self._manager.get_latest_trace()

    def close(self) -> None:
        self._manager.close()


# Legacy classes retained as minimal wrappers for compatibility in case
# downstream code imports them directly.


@dataclass
class ObsSample:
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
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.records: Deque[ObsSample] = deque(maxlen=window_size)

    def update(self, sample: ObsSample) -> None:
        self.records.append(sample)

    def stats(self) -> Dict[str, Optional[float]]:
        if not self.records:
            return {}
        step_times = [r.step_time_s for r in self.records]
        return {
            "step_time_mean": sum(step_times) / len(step_times),
            "step_time_p95": sorted(step_times)[int(0.95 * (len(step_times) - 1))],
        }


class AsyncObservationWorker:
    def __init__(self, obs: ObservationBuffer, maxsize: int = 256):
        self.obs = obs
        self.queue: "queue.Queue[Optional[ObsSample]]" = queue.Queue(maxsize=maxsize)
        self._latest: Dict[str, Optional[float]] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, sample: ObsSample) -> None:
        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            return

    def _worker(self) -> None:
        while not self._stop:
            try:
                sample = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if sample is None:
                break
            self.obs.update(sample)
            self._latest = self.obs.stats()

    def latest_stats(self) -> Dict[str, Optional[float]]:
        return dict(self._latest)

    def close(self) -> None:
        self._stop = True
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)


class ProfilerObservation:
    """Deprecated compatibility wrapper."""

    def __init__(self, cfg: ProfilerObservationConfig):
        self._worker = ObservationWorker(cfg)

    def step_begin(self) -> None:
        self._worker.step_begin()

    def step_end(self) -> None:
        self._worker.step_end()

    def snapshot(self) -> Dict[str, Optional[float]]:
        return {**self._worker.get_latest_trace(), **self._worker.get_window_stats()}

    def close(self) -> None:
        self._worker.close()
