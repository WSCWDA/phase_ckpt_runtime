#!/usr/bin/env python3
from __future__ import print_function
from __future__ import annotations
"""Training entrypoint with decoupled runtime initialization."""



import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

from data import generate_cv_batch, generate_dlrm_batch, generate_lm_batch
from models import DLRM, DLRMConfig, MoETransformerLM, ResNet50, TransformerConfig, TransformerLM
from observation import ObservationEvent, ObservationWorker, ProfilerObservationConfig
from phase_runtime import PhaseAwareCheckpointRuntime, PhaseInference, PhaseInferenceConfig, PhaseRuntimeConfig
from policy_controller import CheckpointPolicyConfig, CheckpointPolicyController
from utils import estimate_state_bytes


@dataclass
class TrainConfig:
    """训练配置，集中管理运行参数。"""

    model: str = "gpt2"
    strategy: str = "phase"
    steps: int = 200
    batch_size: int = 32
    seq_len: int = 64
    vocab_size: int = 32000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 3e-4
    ckpt_interval: int = 10
    ckpt_max_staleness: int = 50
    ckpt_min_interval: int = 2
    ckpt_high_latency_s: float = 0.5
    ckpt_force_sync_on_stale: bool = False
    ckpt_compression_low: int = 1
    ckpt_compression_high: int = 3
    phase_a_ratio: float = 0.3
    phase_a_interval_mul: int = 2
    output_dir: str = "runs"
    async_queue_size: int = 4
    async_timeout_s: float = 1.0
    obs_window: int = 50
    obs_report_every: int = 10
    obs_aggregate_stats: bool = False
    dcgm_enabled: bool = False
    dcgm_poll_interval_s: float = 2.0
    dcgm_window_size: int = 60
    profiler_enabled: bool = False
    profiler_wait: int = 1
    profiler_warmup: int = 1
    profiler_active: int = 2
    profiler_repeat: int = 0
    num_classes: int = 100
    image_size: int = 64
    dlrm_num_dense: int = 8
    dlrm_num_sparse: int = 8
    dlrm_vocab: int = 1000
    dlrm_embed_dim: int = 32
    moe_experts: int = 4
    moe_top_k: int = 2
    moe_aux_weight: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-aware checkpoint runtime prototype")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    parser.add_argument("--model", choices=["dlrm", "resnet50", "gpt2", "llama3", "deepseek_moe", "pythia-410m"])
    parser.add_argument("--strategy", choices=["sync", "async", "phase"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--ckpt-interval", type=int)
    parser.add_argument("--ckpt-max-staleness", type=int)
    parser.add_argument("--ckpt-min-interval", type=int)
    parser.add_argument("--ckpt-high-latency-s", type=float)
    parser.add_argument("--ckpt-force-sync-on-stale", action="store_true")
    parser.add_argument("--ckpt-compression-low", type=int)
    parser.add_argument("--ckpt-compression-high", type=int)
    parser.add_argument("--phase-a-ratio", type=float)
    parser.add_argument("--phase-a-interval-mul", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--async-queue-size", type=int)
    parser.add_argument("--async-timeout-s", type=float)
    parser.add_argument("--obs-window", type=int)
    parser.add_argument("--obs-report-every", type=int)
    parser.add_argument("--obs-aggregate-stats", action="store_true")
    parser.add_argument("--dcgm-enabled", action="store_true")
    parser.add_argument("--dcgm-poll-interval-s", type=float)
    parser.add_argument("--dcgm-window-size", type=int)
    parser.add_argument("--profiler-enabled", action="store_true")
    parser.add_argument("--profiler-wait", type=int)
    parser.add_argument("--profiler-warmup", type=int)
    parser.add_argument("--profiler-active", type=int)
    parser.add_argument("--profiler-repeat", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--dlrm-num-dense", type=int)
    parser.add_argument("--dlrm-num-sparse", type=int)
    parser.add_argument("--dlrm-vocab", type=int)
    parser.add_argument("--dlrm-embed-dim", type=int)
    parser.add_argument("--moe-experts", type=int)
    parser.add_argument("--moe-top-k", type=int)
    parser.add_argument("--moe-aux-weight", type=float)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None and hasattr(cfg, key.replace("-", "_")):
            setattr(cfg, key.replace("-", "_"), value)
    return cfg


def build_model(cfg: TrainConfig, device: torch.device):
    if cfg.model == "dlrm":
        config = DLRMConfig(
            num_dense=cfg.dlrm_num_dense,
            num_sparse=cfg.dlrm_num_sparse,
            vocab_size=cfg.dlrm_vocab,
            embed_dim=cfg.dlrm_embed_dim,
        )
        model = DLRM(config)
        loss_fn = nn.BCEWithLogitsLoss()
        batcher = lambda: generate_dlrm_batch(
            cfg.batch_size,
            config.num_dense,
            config.num_sparse,
            config.vocab_size,
            device,
        )
        return model.to(device), loss_fn, batcher

    if cfg.model == "resnet50":
        model = ResNet50(num_classes=cfg.num_classes, width=16).to(device)
        loss_fn = nn.CrossEntropyLoss()
        batcher = lambda: generate_cv_batch(cfg.batch_size, cfg.num_classes, cfg.image_size, device)
        return model, loss_fn, batcher

    if cfg.model == "pythia-410m":
        # Pythia-410M-like preset: larger width/depth than default toy GPT config.
        # Keep CLI compatibility by preserving seq_len/vocab_size/dropout overrides.
        config = TransformerConfig(
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.seq_len,
            d_model=1024,
            n_heads=16,
            n_layers=24,
            dropout=cfg.dropout,
        )
        model = TransformerLM(config, use_rmsnorm=False, llama_style=False).to(device)
    else:
        config = TransformerConfig(
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
        if cfg.model == "gpt2":
            model = TransformerLM(config, use_rmsnorm=False, llama_style=False).to(device)
        elif cfg.model == "llama3":
            model = TransformerLM(config, use_rmsnorm=True, llama_style=True).to(device)
        elif cfg.model == "deepseek_moe":
            model = MoETransformerLM(config, num_experts=cfg.moe_experts, top_k=cfg.moe_top_k).to(device)
        else:
            raise ValueError(f"Unsupported model: {cfg.model}")

    loss_fn = nn.CrossEntropyLoss()
    batcher = lambda: generate_lm_batch(cfg.batch_size, cfg.seq_len, cfg.vocab_size, device)
    return model, loss_fn, batcher


def init_runtime(cfg: TrainConfig) -> Dict[str, Any]:
    policy_cfg = CheckpointPolicyConfig(
        base_interval=cfg.ckpt_interval,
        max_staleness_steps=cfg.ckpt_max_staleness,
        min_interval_steps=cfg.ckpt_min_interval,
        high_latency_s=cfg.ckpt_high_latency_s,
        force_sync_on_staleness=cfg.ckpt_force_sync_on_stale,
        compression_level_low=cfg.ckpt_compression_low,
        compression_level_high=cfg.ckpt_compression_high,
    )
    policy_controller = CheckpointPolicyController(policy_cfg)
    runtime_cfg = PhaseRuntimeConfig(
        strategy=cfg.strategy,
        ckpt_interval_steps=cfg.ckpt_interval,
        phase_a_ratio=cfg.phase_a_ratio,
        phase_a_interval_mul=cfg.phase_a_interval_mul,
        async_queue_size=cfg.async_queue_size,
        async_timeout_s=cfg.async_timeout_s,
        total_steps=cfg.steps,
        log_every=1,
    )
    runtime = PhaseAwareCheckpointRuntime(runtime_cfg, cfg.output_dir, policy_controller=policy_controller)
    phase_cfg = PhaseInferenceConfig(log_path=os.path.join(cfg.output_dir, "logs", "phase_inference.csv"), window_size=cfg.obs_window)
    phase_inference = PhaseInference(phase_cfg)
    profiler_cfg = ProfilerObservationConfig(
        enabled=cfg.profiler_enabled,
        aggregate_stats=cfg.obs_aggregate_stats,
        window_size=cfg.obs_window,
        dcgm_enabled=cfg.dcgm_enabled,
        dcgm_poll_interval_s=cfg.dcgm_poll_interval_s,
        dcgm_window_size=cfg.dcgm_window_size,
        schedule_wait=cfg.profiler_wait,
        schedule_warmup=cfg.profiler_warmup,
        schedule_active=cfg.profiler_active,
        schedule_repeat=cfg.profiler_repeat,
    )
    observation = ObservationWorker(profiler_cfg)
    return {
        "runtime": runtime,
        "phase_inference": phase_inference,
        "observation": observation,
    }


def init_observation_logger(cfg: TrainConfig) -> tuple[csv.DictWriter, Any]:
    obs_log_path = os.path.join(cfg.output_dir, "logs", "obs_metrics.csv")
    os.makedirs(os.path.dirname(obs_log_path), exist_ok=True)
    obs_log_file = open(obs_log_path, "w", newline="")
    obs_writer = csv.DictWriter(
        obs_log_file,
        fieldnames=[
            "train_step",
            "elapsed_s",
            "event_time_s",
            "step_time",
            "compute_time",
            "cpu_overhead_time",
            "optimizer_time",
            "checkpoint_serialize_time",
            "checkpoint_write_time",
            "checkpoint_total_time",
            "checkpoint_overlap_ratio",
            "queue_depth",
            "last_persisted_step",
            "staleness_steps",
            "cpu_mem_percent",
            "gpu_mem_ratio",
            "gpu_util_percent",
            "step_time_mean",
            "step_time_var",
            "step_time_p50",
            "step_time_p95",
            "step_time_p99",
            "compute_time_mean",
            "compute_time_p95",
            "compute_time_p99",
            "checkpoint_write_p95",
            "checkpoint_write_p99",
            "checkpoint_total_mean",
            "compute_step_ratio_mean",
            "step_time_trend",
            "compute_ratio_trend",
            "progress_rate_steps_per_s",
            "queue_depth_mean",
            "queue_depth_max",
            "staleness_mean",
            "staleness_max",
            "ckpt_completion_rate_per_s",
        ],
    )
    obs_writer.writeheader()
    return obs_writer, obs_log_file


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    cfg = load_config(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model, loss_fn, batcher = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    runtime_bundle = init_runtime(cfg)
    runtime: PhaseAwareCheckpointRuntime = runtime_bundle["runtime"]
    phase_inference: PhaseInference = runtime_bundle["phase_inference"]
    observation: ObservationWorker = runtime_bundle["observation"]
    obs_writer, obs_log_file = init_observation_logger(cfg)

    start_time = time.perf_counter()
    prev_completed = runtime.num_completed
    full_ckpt_size = None
    prev_params = None

    try:
        for step in range(1, cfg.steps + 1):
            step_start = time.perf_counter()
            model.train()

            if cfg.model == "dlrm":
                dense, sparse, targets = batcher()
                logits = model(dense, sparse)
                loss = loss_fn(logits, targets)
            elif cfg.model == "resnet50":
                images, targets = batcher()
                logits = model(images)
                loss = loss_fn(logits, targets)
            else:
                input_ids, targets = batcher()
                if cfg.model == "deepseek_moe":
                    logits, aux_loss = model(input_ids)
                    loss = loss_fn(logits.view(-1, cfg.vocab_size), targets.view(-1))
                    loss = loss + cfg.moe_aux_weight * aux_loss
                else:
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, cfg.vocab_size), targets.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step_time = time.perf_counter() - step_start
            queue_depth = runtime.get_queue_depth()
            last_persisted = runtime.get_last_persisted_step()
            staleness = step - last_persisted
            elapsed_s = time.perf_counter() - start_time
            ckpt_latency = None
            if runtime.num_completed > prev_completed:
                ckpt_latency = runtime.get_last_completed_latency()
                prev_completed = runtime.num_completed

            observation.emit(
                ObservationEvent(
                    step_id=step,
                    event_time_s=time.perf_counter(),
                    step_time_s=step_time,
                    ckpt_write_time_s=ckpt_latency,
                    queue_depth=queue_depth,
                    last_persisted_step=last_persisted,
                    staleness_steps=staleness,
                )
            )
            observation.step_end()

            loss_value = float(loss.detach().cpu().item())
            if full_ckpt_size is None:
                full_ckpt_size = estimate_state_bytes(model.state_dict())
            params_list = [p for p in model.parameters() if p.requires_grad]
            with torch.no_grad():
                if prev_params is None:
                    parameter_change_norm = None
                else:
                    sq_sum = torch.zeros((), device=device)
                    for prev, param in zip(prev_params, params_list):
                        diff = param.detach() - prev
                        sq_sum += diff.pow(2).sum()
                    parameter_change_norm = float(torch.sqrt(sq_sum).item())
                prev_params = [param.detach().clone() for param in params_list]

            obs_stats = observation.get_window_stats()
            obs_snapshot = {**observation.get_latest_trace(), **obs_stats}
            obs_snapshot.update({"step_time": obs_snapshot.get("step_time") or step_time})

            phase_inference.update(
                {
                    "step": step,
                    "compute_time": obs_stats.get("compute_time_mean"),
                    "checkpoint_write_time": obs_stats.get("checkpoint_write_p95") or ckpt_latency,
                    "checkpoint_queue_depth": queue_depth,
                    "delta_size": parameter_change_norm,
                    "full_ckpt_size": full_ckpt_size,
                    "parameter_change_norm": parameter_change_norm,
                    "compression_error": None,
                    "loss": loss_value,
                    "param_dist_metric": None,
                }
            )
            phase_state = phase_inference.current_phase_state()

            if step % cfg.obs_report_every == 0:
                obs_writer.writerow(
                    {
                        "train_step": step,
                        "elapsed_s": elapsed_s,
                        **obs_snapshot,
                    }
                )
                obs_log_file.flush()

            runtime.maybe_checkpoint(
                step,
                model,
                optimizer,
                {
                    "step_time": step_time,
                    "async_applicable": int(phase_state.async_applicable),
                    "incr_applicable": int(phase_state.delta_applicable),
                    "comp_applicable": int(phase_state.compression_applicable),
                    "phase_id": phase_state.phase_id,
                },
                phase_state=phase_state,
                observation_stats={
                    "staleness_steps": staleness,
                    "ckpt_latency": ckpt_latency,
                    "queue_depth": queue_depth,
                },
            )
    finally:
        obs_log_file.close()
        observation.close()
        phase_inference.close()
        runtime.close()
    logging.info("Run complete. Logs saved to %s", runtime.log_path)


if __name__ == "__main__":
    main()
