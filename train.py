import argparse
import csv
import os
import time

import torch
from torch import nn

from observation import ObservationEvent, ObservationWorker, ProfilerObservationConfig
from phase_runtime import PhaseAwareCheckpointRuntime, PhaseInference, PhaseInferenceConfig, PhaseRuntimeConfig
from policy_controller import CheckpointPolicyConfig, CheckpointPolicyController
from data import generate_cv_batch, generate_dlrm_batch, generate_lm_batch
from models import DLRM, DLRMConfig, MoETransformerLM, ResNet50, TransformerConfig, TransformerLM
from utils import estimate_state_bytes


def parse_args() -> argparse.Namespace:
    """解析训练与 phase inference 相关参数。"""
    parser = argparse.ArgumentParser(description="Phase-aware checkpoint runtime prototype")
    parser.add_argument(
        "--model",
        choices=["dlrm", "resnet50", "gpt2", "llama3", "deepseek_moe"],
        default="gpt2",
    )
    parser.add_argument("--strategy", choices=["sync", "async", "phase"], default="phase")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt-interval", type=int, default=10)
    parser.add_argument("--ckpt-max-staleness", type=int, default=50)
    parser.add_argument("--ckpt-min-interval", type=int, default=2)
    parser.add_argument("--ckpt-high-latency-s", type=float, default=0.5)
    parser.add_argument("--ckpt-force-sync-on-stale", action="store_true", default=False)
    parser.add_argument("--ckpt-compression-low", type=int, default=1)
    parser.add_argument("--ckpt-compression-high", type=int, default=3)
    parser.add_argument("--phase-a-ratio", type=float, default=0.3)
    parser.add_argument("--phase-a-interval-mul", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--async-queue-size", type=int, default=4)
    parser.add_argument("--async-timeout-s", type=float, default=1.0)
    parser.add_argument("--obs-window", type=int, default=50)
    parser.add_argument("--obs-report-every", type=int, default=10)
    parser.add_argument("--obs-aggregate-stats", action="store_true", default=False)
    parser.add_argument("--profiler-enabled", action="store_true", default=False)
    parser.add_argument("--profiler-wait", type=int, default=1)
    parser.add_argument("--profiler-warmup", type=int, default=1)
    parser.add_argument("--profiler-active", type=int, default=2)
    parser.add_argument("--profiler-repeat", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dlrm-num-dense", type=int, default=8)
    parser.add_argument("--dlrm-num-sparse", type=int, default=8)
    parser.add_argument("--dlrm-vocab", type=int, default=1000)
    parser.add_argument("--dlrm-embed-dim", type=int, default=32)
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-aux-weight", type=float, default=0.1)
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device):
    """构建模型、损失函数与批数据生成器。"""
    if args.model == "dlrm":
        config = DLRMConfig(
            num_dense=args.dlrm_num_dense,
            num_sparse=args.dlrm_num_sparse,
            vocab_size=args.dlrm_vocab,
            embed_dim=args.dlrm_embed_dim,
        )
        model = DLRM(config)
        loss_fn = nn.BCEWithLogitsLoss()
        batcher = lambda: generate_dlrm_batch(
            args.batch_size,
            config.num_dense,
            config.num_sparse,
            config.vocab_size,
            device,
        )
        return model.to(device), loss_fn, batcher

    if args.model == "resnet50":
        model = ResNet50(num_classes=args.num_classes, width=16).to(device)
        loss_fn = nn.CrossEntropyLoss()
        batcher = lambda: generate_cv_batch(args.batch_size, args.num_classes, args.image_size, device)
        return model, loss_fn, batcher

    config = TransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    if args.model == "gpt2":
        model = TransformerLM(config, use_rmsnorm=False, llama_style=False).to(device)
    elif args.model == "llama3":
        model = TransformerLM(config, use_rmsnorm=True, llama_style=True).to(device)
    elif args.model == "deepseek_moe":
        model = MoETransformerLM(config, num_experts=args.moe_experts, top_k=args.moe_top_k).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    loss_fn = nn.CrossEntropyLoss()
    batcher = lambda: generate_lm_batch(args.batch_size, args.seq_len, args.vocab_size, device)
    return model, loss_fn, batcher


def main() -> None:
    """主训练入口，包含 phase inference 与 checkpoint runtime。"""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, loss_fn, batcher = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    runtime_cfg = PhaseRuntimeConfig(
        strategy=args.strategy,
        ckpt_interval_steps=args.ckpt_interval,
        phase_a_ratio=args.phase_a_ratio,
        phase_a_interval_mul=args.phase_a_interval_mul,
        async_queue_size=args.async_queue_size,
        async_timeout_s=args.async_timeout_s,
        total_steps=args.steps,
        log_every=1,
    )
    policy_cfg = CheckpointPolicyConfig(
        base_interval=args.ckpt_interval,
        max_staleness_steps=args.ckpt_max_staleness,
        min_interval_steps=args.ckpt_min_interval,
        high_latency_s=args.ckpt_high_latency_s,
        force_sync_on_staleness=args.ckpt_force_sync_on_stale,
        compression_level_low=args.ckpt_compression_low,
        compression_level_high=args.ckpt_compression_high,
    )
    policy_controller = CheckpointPolicyController(policy_cfg)
    runtime = PhaseAwareCheckpointRuntime(runtime_cfg, args.output_dir, policy_controller=policy_controller)
    phase_log_path = os.path.join(args.output_dir, "logs", "phase_inference.csv")
    phase_cfg = PhaseInferenceConfig(log_path=phase_log_path, window_size=args.obs_window)
    phase_inference = PhaseInference(phase_cfg)
    profiler_cfg = ProfilerObservationConfig(
        enabled=args.profiler_enabled,
        aggregate_stats=args.obs_aggregate_stats,
        window_size=args.obs_window,
        schedule_wait=args.profiler_wait,
        schedule_warmup=args.profiler_warmup,
        schedule_active=args.profiler_active,
        schedule_repeat=args.profiler_repeat,
    )
    observation = ObservationWorker(profiler_cfg)
    obs_log_path = os.path.join(args.output_dir, "logs", "obs_metrics.csv")
    os.makedirs(os.path.dirname(obs_log_path), exist_ok=True)
    obs_log_file = open(obs_log_path, "w", newline="")
    obs_writer = csv.DictWriter(
        obs_log_file,
        fieldnames=[
            "train_step",
            "elapsed_s",
            "step_time",
            "compute_time",
            "cpu_overhead_time",
            "optimizer_time",
            "checkpoint_serialize_time",
            "checkpoint_write_time",
            "checkpoint_total_time",
            "checkpoint_overlap_ratio",
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
        ],
    )
    obs_writer.writeheader()
    start_time = time.perf_counter()
    prev_completed = runtime.num_completed
    full_ckpt_size = None
    prev_params = None

    try:
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            observation.step_begin()
            model.train()

            with torch.profiler.record_function("train_step"):
                if args.model == "dlrm":
                    dense, sparse, targets = batcher()
                    logits = model(dense, sparse)
                    loss = loss_fn(logits, targets)
                elif args.model == "resnet50":
                    images, targets = batcher()
                    logits = model(images)
                    loss = loss_fn(logits, targets)
                else:
                    input_ids, targets = batcher()
                    if args.model == "deepseek_moe":
                        logits, aux_loss = model(input_ids)
                        loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))
                        loss = loss + args.moe_aux_weight * aux_loss
                    else:
                        logits = model(input_ids)
                        loss = loss_fn(logits.view(-1, args.vocab_size), targets.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()
            observation.emit(ObservationEvent(step_id=step, step_time_s=time.perf_counter() - step_start))
            observation.step_end()

            step_time = time.perf_counter() - step_start
            queue_depth = runtime.get_queue_depth()
            last_persisted = runtime.get_last_persisted_step()
            staleness = step - last_persisted
            elapsed_s = time.perf_counter() - start_time
            ckpt_latency = None
            if runtime.num_completed > prev_completed:
                ckpt_latency = runtime.get_last_completed_latency()
                prev_completed = runtime.num_completed
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

            if step % args.obs_report_every == 0:
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
    print(f"Run complete. Logs saved to {runtime.log_path}")


if __name__ == "__main__":
    main()
