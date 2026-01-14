from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from phase_runtime import PhaseState


@dataclass
class PolicyDecision:
    """Policy 决策结果，控制触发与机制开关。"""

    do_checkpoint: bool
    use_async: bool
    use_delta: bool
    use_compression: bool
    compression_level: Optional[int]
    reason: Dict[str, Any]


@dataclass
class CheckpointPolicyConfig:
    """Policy 控制参数。"""

    base_interval: int = 10
    max_staleness_steps: int = 50
    min_interval_steps: int = 2
    high_latency_s: float = 0.5
    force_sync_on_staleness: bool = True
    compression_level_low: int = 1
    compression_level_high: int = 3


class CheckpointPolicyController:
    """基于适用性与运行时统计的策略控制器。"""

    def __init__(self, cfg: CheckpointPolicyConfig):
        self.cfg = cfg
        self._last_ckpt_step = 0

    def _should_force_checkpoint(self, staleness: Optional[int]) -> bool:
        if staleness is None:
            return False
        return staleness >= self.cfg.max_staleness_steps

    def decide(
        self,
        step_id: int,
        phase_state: Optional[PhaseState],
        observation_stats: Dict[str, Any],
    ) -> PolicyDecision:
        """决策是否触发 checkpoint，以及启用哪些机制。"""

        async_applicable = phase_state.async_applicable if phase_state else False
        delta_applicable = phase_state.delta_applicable if phase_state else False
        compression_applicable = phase_state.compression_applicable if phase_state else False
        staleness = observation_stats.get("staleness_steps")
        last_latency = observation_stats.get("ckpt_latency")
        step_gap = step_id - self._last_ckpt_step

        reason: Dict[str, Any] = {
            "async_applicable": async_applicable,
            "delta_applicable": delta_applicable,
            "compression_applicable": compression_applicable,
            "staleness_steps": staleness,
            "last_ckpt_latency": last_latency,
            "step_gap": step_gap,
        }

        force_checkpoint = self._should_force_checkpoint(staleness)
        if force_checkpoint:
            reason["trigger"] = "staleness_exceeded"

        if step_gap < self.cfg.min_interval_steps and not force_checkpoint:
            return PolicyDecision(
                do_checkpoint=False,
                use_async=False,
                use_delta=False,
                use_compression=False,
                compression_level=None,
                reason={**reason, "trigger": "min_interval_hold"},
            )

        if last_latency is not None and last_latency >= self.cfg.high_latency_s and not force_checkpoint:
            return PolicyDecision(
                do_checkpoint=False,
                use_async=False,
                use_delta=False,
                use_compression=False,
                compression_level=None,
                reason={**reason, "trigger": "high_latency_skip"},
            )

        soft_interval_reached = step_gap >= self.cfg.base_interval
        if not soft_interval_reached and not force_checkpoint:
            return PolicyDecision(
                do_checkpoint=False,
                use_async=False,
                use_delta=False,
                use_compression=False,
                compression_level=None,
                reason={**reason, "trigger": "interval_not_reached"},
            )

        use_async = async_applicable and not (force_checkpoint and self.cfg.force_sync_on_staleness)
        use_delta = delta_applicable
        use_compression = compression_applicable
        compression_level = (
            self.cfg.compression_level_high if compression_applicable else self.cfg.compression_level_low
        )
        reason["trigger"] = "force" if force_checkpoint else "interval_or_policy"
        decision = PolicyDecision(
            do_checkpoint=True,
            use_async=use_async,
            use_delta=use_delta,
            use_compression=use_compression,
            compression_level=compression_level if use_compression else None,
            reason=reason,
        )
        self._last_ckpt_step = step_id
        return decision
