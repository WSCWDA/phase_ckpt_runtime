from __future__ import annotations

# Phase-aware Checkpoint Runtime（简化版 4.2）

import csv
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    incr_applicable: bool
    comp_applicable: bool
    reason: Dict[str, Any]
    phase_id: str


class PhaseInference:
    """基于窗口统计与阈值的 Phase Inference（带抗抖动）。"""
    def __init__(
        self,
        window_size: int,
        min_duration: int,
        async_q_max: int,
        async_staleness_max: int,
        async_overlap_margin: float,
        incr_delta_thres: float,
        comp_stability_thres: float,
        incr_default: bool,
        switch_k: int = 1,
    ):
        self.window_size = window_size
        self.min_duration = min_duration
        self.async_q_max = async_q_max
        self.async_staleness_max = async_staleness_max
        self.async_overlap_margin = async_overlap_margin
        self.incr_delta_thres = incr_delta_thres
        self.comp_stability_thres = comp_stability_thres
        self.incr_default = incr_default
        self.switch_k = switch_k
        self.current_phase = PhaseState(False, incr_default, False, {}, "A0_I0_C0")
        self.phase_duration_steps = 0
        self._candidate_count = 0

    def infer(self, current_step: int, stats: Dict[str, Optional[float]]) -> PhaseState:
        """根据统计量输出稳定的 phase 状态。"""
        c_bar = stats.get("C_bar")
        w_bar = stats.get("W_bar")
        q_bar = stats.get("Q_bar")
        s_bar = stats.get("S_bar")
        d_bar = stats.get("D_bar")

        async_ok = False
        if q_bar is not None and s_bar is not None:
            if w_bar is not None and c_bar is not None:
                async_ok = c_bar >= w_bar + self.async_overlap_margin and q_bar <= self.async_q_max
            else:
                async_ok = q_bar <= self.async_q_max
            async_ok = async_ok and s_bar <= self.async_staleness_max

        if d_bar is None:
            incr_ok = self.incr_default
        else:
            incr_ok = d_bar <= self.incr_delta_thres

        if d_bar is None:
            comp_ok = False
        else:
            comp_ok = d_bar <= self.comp_stability_thres

        next_phase_id = f"A{int(async_ok)}_I{int(incr_ok)}_C{int(comp_ok)}"
        reason = {
            "C_bar": c_bar,
            "W_bar": w_bar,
            "Q_bar": q_bar,
            "S_bar": s_bar,
            "D_bar": d_bar,
            "candidate_phase": next_phase_id,
            "current_phase": self.current_phase.phase_id,
        }

        if next_phase_id == self.current_phase.phase_id:
            self.phase_duration_steps += 1
            self._candidate_count = 0
            return self.current_phase

        if self.phase_duration_steps < self.min_duration:
            self.phase_duration_steps += 1
            reason["blocked_by_min_duration"] = True
            return PhaseState(
                self.current_phase.async_applicable,
                self.current_phase.incr_applicable,
                self.current_phase.comp_applicable,
                reason,
                self.current_phase.phase_id,
            )

        self._candidate_count += 1
        if self._candidate_count < self.switch_k:
            reason["blocked_by_switch_k"] = True
            return PhaseState(
                self.current_phase.async_applicable,
                self.current_phase.incr_applicable,
                self.current_phase.comp_applicable,
                reason,
                self.current_phase.phase_id,
            )

        self.current_phase = PhaseState(async_ok, incr_ok, comp_ok, reason, next_phase_id)
        self.phase_duration_steps = 0
        self._candidate_count = 0
        return self.current_phase




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
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "rng_state": capture_rng_state(),
            }
            path = self._checkpoint_path(step)
            if ckpt_mode == "sync":
                start = time.time()
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
