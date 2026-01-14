from __future__ import annotations

# Phase-aware Checkpoint Runtime（简化版 4.2）

import csv
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from utils import capture_rng_state


@dataclass
class PhaseRuntimeConfig:
    """运行时配置参数。"""
    strategy: str = "phase"
    ckpt_interval_steps: int = 10
    phase_a_ratio: float = 0.3
    phase_a_interval_mul: int = 2
    async_queue_size: int = 4
    async_timeout_s: float = 1.0
    total_steps: int = 100
    log_every: int = 1




@dataclass
class PhaseState:
    """Phase 推断输出状态。"""
    async_applicable: bool
    delta_applicable: bool
    compression_applicable: bool
    reason: Dict[str, Any]
    phase_id: str


@dataclass
class PhaseInferenceConfig:
    """Phase inference 的阈值与稳定性配置。"""

    window_size: int = 50
    min_phase_steps: int = 10
    vote_threshold_async: int = 3
    vote_threshold_delta: int = 3
    vote_threshold_compression: int = 2
    async_enter_ratio: float = 1.0
    async_exit_ratio: float = 0.9
    async_queue_trend_tolerance: float = 0.0
    compute_cv_enter: float = 0.2
    compute_cv_exit: float = 0.3
    delta_ratio_enter: float = 0.2
    delta_ratio_exit: float = 0.3
    delta_spike_ratio_enter: float = 3.0
    delta_spike_ratio_exit: float = 4.0
    compression_error_enter: float = 0.02
    compression_error_exit: float = 0.03
    loss_trend_tolerance: float = 0.0
    param_dist_cv_enter: float = 0.2
    param_dist_cv_exit: float = 0.3
    log_path: Optional[str] = None


class PhaseInference:
    """基于窗口统计的适用性 Phase 推断（系统运行时视角）。"""

    def __init__(self, cfg: PhaseInferenceConfig):
        self.cfg = cfg
        self._history: List[Dict[str, Optional[float]]] = []
        self._current = PhaseState(False, False, False, {}, "A0_D0_C0")
        self._phase_steps = 0
        self._log_file = None
        self._log_writer = None
        if cfg.log_path:
            os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
            self._log_file = open(cfg.log_path, "w", newline="")
            self._log_writer = csv.DictWriter(
                self._log_file,
                fieldnames=[
                    "step",
                    "async_applicable",
                    "delta_applicable",
                    "compression_applicable",
                    "phase_id",
                    "transition",
                    "compute_time_avg",
                    "checkpoint_write_p95",
                    "queue_depth_trend",
                    "compute_time_cv",
                    "delta_ratio_avg",
                    "delta_trend",
                    "param_change_spike_ratio",
                    "compression_error_p95",
                    "loss_trend",
                    "param_dist_cv",
                    "async_votes",
                    "delta_votes",
                    "compression_votes",
                    "async_conditions",
                    "delta_conditions",
                    "compression_conditions",
                ],
            )
            self._log_writer.writeheader()

    def _window(self) -> List[Dict[str, Optional[float]]]:
        return self._history[-self.cfg.window_size :]

    def _extract_series(self, key: str) -> List[float]:
        return [item[key] for item in self._window() if item.get(key) is not None]

    def _moving_avg(self, values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def _percentile(self, values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round((pct / 100.0) * (len(ordered) - 1)))
        return ordered[min(max(idx, 0), len(ordered) - 1)]

    def _coefficient_of_variation(self, values: List[float]) -> Optional[float]:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        if mean == 0:
            return None
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return (variance**0.5) / abs(mean)

    def _trend(self, values: List[float], tolerance: float) -> str:
        if len(values) < 2:
            return "stable"
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        increasing = all(diff >= -tolerance for diff in diffs) and any(diff > tolerance for diff in diffs)
        decreasing = all(diff <= tolerance for diff in diffs) and any(diff < -tolerance for diff in diffs)
        if increasing:
            return "increasing"
        if decreasing:
            return "decreasing"
        return "stable"

    def _ratio(self, numer: List[float], denom: List[float]) -> List[float]:
        ratios = []
        for n, d in zip(numer, denom):
            if d:
                ratios.append(n / d)
        return ratios

    def _apply_hysteresis(self, current: bool, enter: bool, exit: bool) -> bool:
        if current:
            return not exit
        return enter

    def _vote(self, signals: Dict[str, bool], threshold: int) -> bool:
        return sum(1 for val in signals.values() if val) >= threshold

    def update(self, observation_snapshot: Dict[str, Optional[float]]) -> None:
        """更新观测窗口并做 phase 推断。"""
        self._history.append(dict(observation_snapshot))
        if len(self._history) > self.cfg.window_size * 2:
            self._history = self._history[-self.cfg.window_size :]
        window = self._window()
        if len(window) < 2:
            return

        compute_times = self._extract_series("compute_time")
        ckpt_write_times = self._extract_series("checkpoint_write_time")
        queue_depth = self._extract_series("checkpoint_queue_depth")
        delta_sizes = self._extract_series("delta_size")
        full_sizes = self._extract_series("full_ckpt_size")
        param_change = self._extract_series("parameter_change_norm")
        compression_error = self._extract_series("compression_error")
        loss_series = self._extract_series("loss")
        param_dist_metric = self._extract_series("param_dist_metric")

        compute_avg = self._moving_avg(compute_times)
        ckpt_p95 = self._percentile(ckpt_write_times, 95)
        queue_trend = self._trend(queue_depth, self.cfg.async_queue_trend_tolerance) if queue_depth else "stable"
        compute_cv = self._coefficient_of_variation(compute_times)
        delta_ratio = self._ratio(delta_sizes, full_sizes)
        delta_avg = self._moving_avg(delta_ratio)
        delta_trend = self._trend(delta_sizes, 0.0) if delta_sizes else "stable"
        param_change_p50 = self._percentile(param_change, 50)
        param_change_p95 = self._percentile(param_change, 95)
        param_change_spike_ratio = (
            (param_change_p95 / param_change_p50) if param_change_p50 and param_change_p95 else None
        )
        compression_p95 = self._percentile(compression_error, 95)
        loss_trend = self._trend(loss_series, self.cfg.loss_trend_tolerance) if loss_series else "stable"
        param_dist_cv = self._coefficient_of_variation(param_dist_metric)

        async_conditions = {
            "compute_vs_ckpt": (
                compute_avg is not None
                and ckpt_p95 is not None
                and compute_avg >= ckpt_p95 * self.cfg.async_enter_ratio
            ),
            "queue_not_increasing": queue_trend != "increasing",
            "compute_stable": (compute_cv is not None and compute_cv <= self.cfg.compute_cv_enter),
        }
        async_exit = (
            compute_avg is not None
            and ckpt_p95 is not None
            and compute_avg < ckpt_p95 * self.cfg.async_exit_ratio
        ) or (compute_cv is not None and compute_cv >= self.cfg.compute_cv_exit)

        delta_conditions = {
            "delta_ratio": delta_avg is not None and delta_avg <= self.cfg.delta_ratio_enter,
            "param_locality": (
                param_change_spike_ratio is not None and param_change_spike_ratio <= self.cfg.delta_spike_ratio_enter
            ),
            "delta_trend": delta_trend in ("stable", "decreasing"),
        }
        delta_exit = (
            delta_avg is not None and delta_avg >= self.cfg.delta_ratio_exit
        ) or (
            param_change_spike_ratio is not None
            and param_change_spike_ratio >= self.cfg.delta_spike_ratio_exit
        )

        compression_conditions = {
            "param_dist_stable": (
                param_dist_cv is not None and param_dist_cv <= self.cfg.param_dist_cv_enter
            ),
            "compression_error": compression_p95 is not None and compression_p95 <= self.cfg.compression_error_enter,
            "loss_not_degrading": loss_trend != "increasing",
        }
        compression_exit = (
            compression_p95 is not None and compression_p95 >= self.cfg.compression_error_exit
        ) or (
            param_dist_cv is not None and param_dist_cv >= self.cfg.param_dist_cv_exit
        ) or (
            loss_trend == "increasing"
        )

        async_vote = self._vote(async_conditions, self.cfg.vote_threshold_async)
        delta_vote = self._vote(delta_conditions, self.cfg.vote_threshold_delta)
        compression_vote = self._vote(compression_conditions, self.cfg.vote_threshold_compression)

        async_applicable = self._apply_hysteresis(self._current.async_applicable, async_vote, async_exit)
        delta_applicable = self._apply_hysteresis(self._current.delta_applicable, delta_vote, delta_exit)
        compression_applicable = self._apply_hysteresis(
            self._current.compression_applicable, compression_vote, compression_exit
        )

        candidate_id = f"A{int(async_applicable)}_D{int(delta_applicable)}_C{int(compression_applicable)}"
        transition = ""
        if candidate_id != self._current.phase_id and self._phase_steps >= self.cfg.min_phase_steps:
            transition = f"{self._current.phase_id}->{candidate_id}"
            self._current = PhaseState(
                async_applicable,
                delta_applicable,
                compression_applicable,
                {
                    "async_conditions": async_conditions,
                    "delta_conditions": delta_conditions,
                    "compression_conditions": compression_conditions,
                },
                candidate_id,
            )
            self._phase_steps = 0
        else:
            self._phase_steps += 1

        if self._log_writer:
            step = observation_snapshot.get("step")
            self._log_writer.writerow(
                {
                    "step": step,
                    "async_applicable": int(self._current.async_applicable),
                    "delta_applicable": int(self._current.delta_applicable),
                    "compression_applicable": int(self._current.compression_applicable),
                    "phase_id": self._current.phase_id,
                    "transition": transition,
                    "compute_time_avg": compute_avg,
                    "checkpoint_write_p95": ckpt_p95,
                    "queue_depth_trend": queue_trend,
                    "compute_time_cv": compute_cv,
                    "delta_ratio_avg": delta_avg,
                    "delta_trend": delta_trend,
                    "param_change_spike_ratio": param_change_spike_ratio,
                    "compression_error_p95": compression_p95,
                    "loss_trend": loss_trend,
                    "param_dist_cv": param_dist_cv,
                    "async_votes": int(async_vote),
                    "delta_votes": int(delta_vote),
                    "compression_votes": int(compression_vote),
                    "async_conditions": json.dumps(async_conditions),
                    "delta_conditions": json.dumps(delta_conditions),
                    "compression_conditions": json.dumps(compression_conditions),
                }
            )
            self._log_file.flush()

    def current_phase_state(self) -> PhaseState:
        """返回当前 phase 适用性状态。"""
        return self._current

    def should_enable_async_ckpt(self) -> bool:
        return self._current.async_applicable

    def should_enable_delta_ckpt(self) -> bool:
        return self._current.delta_applicable

    def should_enable_compression_ckpt(self) -> bool:
        return self._current.compression_applicable

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()




@dataclass
class AsyncTask:
    step: int
    payload: Dict[str, Any]
    path: str


class AsyncWriter:
    """后台异步写盘线程与队列。"""
    def __init__(self, maxsize: int, timeout_s: float, on_complete):
        self.queue: queue.Queue[Optional[AsyncTask]] = queue.Queue(maxsize=maxsize)
        self.timeout_s = timeout_s
        self.on_complete = on_complete
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, task: AsyncTask) -> bool:
        """提交异步写盘任务（可能阻塞/超时）。"""
        try:
            self.queue.put(task, timeout=self.timeout_s)
            return True
        except queue.Full:
            return False

    def _worker(self) -> None:
        """后台线程：串行写盘并回调完成信息。"""
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            start = time.time()
            with torch.profiler.record_function("checkpoint_write"):
                torch.save(task.payload, task.path)
            end = time.time()
            ckpt_bytes = os.path.getsize(task.path)
            self.on_complete(task.step, end - start, ckpt_bytes)
            self.queue.task_done()

    def flush(self) -> None:
        """等待队列任务全部完成。"""
        self.queue.join()

    def close(self) -> None:
        """安全关闭线程（先 flush 再退出）。"""
        self.flush()
        self.queue.put(None)
        self._thread.join()

    def depth(self) -> int:
        """队列当前深度。"""
        return self.queue.qsize()


class PhaseAwareCheckpointRuntime:
    """统一的 Phase-aware Checkpoint Runtime 接口。"""
    def __init__(self, cfg: PhaseRuntimeConfig, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"run_{timestamp}.csv")
        self._start_time = time.time()
        self._last_persisted_step = 0
        self._last_ckpt_latency = 0.0
        self._last_ckpt_bytes = 0
        self.last_completed_latency_s = 0.0
        self.num_issued = 0
        self.num_completed = 0
        self.max_ckpt_latency_s = 0.0
        self._lock = threading.Lock()
        self._async_writer = AsyncWriter(cfg.async_queue_size, cfg.async_timeout_s, self._on_async_complete)
        self._log_file = open(self.log_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._log_file,
            fieldnames=[
                "wall_time",
                "step",
                "step_time",
                "ckpt_triggered",
                "ckpt_mode",
                "queue_depth",
                "staleness_steps",
                "ckpt_bytes",
                "ckpt_latency",
                "C_bar",
                "W_bar",
                "Q_bar",
                "S_bar",
                "D_bar",
                "async_applicable",
                "incr_applicable",
                "comp_applicable",
                "phase_blocked",
                "phase_id",
            ],
        )
        self._writer.writeheader()

    def _on_async_complete(self, step: int, latency: float, ckpt_bytes: int) -> None:
        """异步写盘完成回调，用于更新 staleness 等状态。"""
        with self._lock:
            self._last_persisted_step = max(self._last_persisted_step, step)
            self._last_ckpt_latency = latency
            self._last_ckpt_bytes = ckpt_bytes
            self.last_completed_latency_s = latency
            self.num_completed += 1
            self.max_ckpt_latency_s = max(self.max_ckpt_latency_s, latency)

    def _checkpoint_path(self, step: int) -> str:
        return os.path.join(self.ckpt_dir, f"step_{step:06d}.pt")

    def _phase_a_steps(self) -> int:
        return max(1, int(self.cfg.total_steps * self.cfg.phase_a_ratio))

    def maybe_checkpoint(self, step: int, model, optimizer, metrics_dict: Dict[str, Any]) -> None:
        """按策略触发 checkpoint，并记录观测指标。"""
        ckpt_triggered = False
        ckpt_mode = "none"
        ckpt_bytes = 0
        ckpt_latency = 0.0

        interval = self.cfg.ckpt_interval_steps
        if self.cfg.strategy == "phase":
            if step <= self._phase_a_steps():
                interval = max(1, interval * self.cfg.phase_a_interval_mul)
                ckpt_mode = "sync"
            else:
                ckpt_mode = "async"
        elif self.cfg.strategy == "sync":
            ckpt_mode = "sync"
        elif self.cfg.strategy == "async":
            ckpt_mode = "async"
        else:
            raise ValueError(f"Unsupported strategy: {self.cfg.strategy}")

        if step % interval == 0:
            ckpt_triggered = True
            with torch.profiler.record_function("checkpoint_serialize"):
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "rng_state": capture_rng_state(),
                }
            path = self._checkpoint_path(step)
            if ckpt_mode == "sync":
                start = time.time()
                with torch.profiler.record_function("checkpoint_write"):
                    torch.save(payload, path)
                ckpt_latency = time.time() - start
                ckpt_bytes = os.path.getsize(path)
                with self._lock:
                    self._last_persisted_step = max(self._last_persisted_step, step)
                    self._last_ckpt_latency = ckpt_latency
                    self._last_ckpt_bytes = ckpt_bytes
                    self.last_completed_latency_s = ckpt_latency
                    self.num_completed += 1
                    self.num_issued += 1
                    self.max_ckpt_latency_s = max(self.max_ckpt_latency_s, ckpt_latency)
            else:
                task = AsyncTask(step=step, payload=payload, path=path)
                self.num_issued += 1
                submitted = self._async_writer.submit(task)
                if not submitted:
                    ckpt_mode = "async_fallback_sync"
                    start = time.time()
                    with torch.profiler.record_function("checkpoint_write"):
                        torch.save(payload, path)
                    ckpt_latency = time.time() - start
                    ckpt_bytes = os.path.getsize(path)
                    with self._lock:
                        self._last_persisted_step = max(self._last_persisted_step, step)
                        self._last_ckpt_latency = ckpt_latency
                        self._last_ckpt_bytes = ckpt_bytes
                        self.last_completed_latency_s = ckpt_latency
                        self.num_completed += 1
                        self.max_ckpt_latency_s = max(self.max_ckpt_latency_s, ckpt_latency)

        with self._lock:
            last_persisted = self._last_persisted_step
            ckpt_latency = ckpt_latency or self._last_ckpt_latency
            ckpt_bytes = ckpt_bytes or self._last_ckpt_bytes

        staleness_steps = step - last_persisted
        log_row = {
            "wall_time": time.time() - self._start_time,
            "step": step,
            "step_time": metrics_dict.get("step_time", 0.0),
            "ckpt_triggered": int(ckpt_triggered),
            "ckpt_mode": ckpt_mode,
            "queue_depth": self._async_writer.depth(),
            "staleness_steps": staleness_steps,
            "ckpt_bytes": ckpt_bytes,
            "ckpt_latency": ckpt_latency,
            "C_bar": metrics_dict.get("C_bar"),
            "W_bar": metrics_dict.get("W_bar"),
            "Q_bar": metrics_dict.get("Q_bar"),
            "S_bar": metrics_dict.get("S_bar"),
            "D_bar": metrics_dict.get("D_bar"),
            "async_applicable": metrics_dict.get("async_applicable"),
            "incr_applicable": metrics_dict.get("incr_applicable"),
            "comp_applicable": metrics_dict.get("comp_applicable"),
            "phase_blocked": metrics_dict.get("phase_blocked"),
            "phase_id": metrics_dict.get("phase_id"),
        }
        self._writer.writerow(log_row)
        self._log_file.flush()

    def close(self) -> None:
        """训练结束时安全关闭并 flush 异步队列。"""
        try:
            self._async_writer.close()
        finally:
            self._log_file.close()

    def get_queue_depth(self) -> int:
        """已提交未完成的队列深度。"""
        return self.num_issued - self.num_completed

    def get_last_persisted_step(self) -> int:
        """最近一次落盘的 step。"""
        with self._lock:
            return self._last_persisted_step

    def get_last_ckpt_latency(self) -> float:
        """最近一次 checkpoint 的写盘耗时。"""
        with self._lock:
            return self._last_ckpt_latency

    def get_last_completed_latency(self) -> float:
        """最近一次完成的 checkpoint 写盘耗时。"""
        with self._lock:
            return self.last_completed_latency_s
