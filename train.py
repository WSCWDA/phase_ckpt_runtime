import argparse
import csv
import os
import time

import torch
from torch import nn

from observation import AsyncObservationWorker, ObsSample, ObservationBuffer
from phase_runtime import PhaseAwareCheckpointRuntime, PhaseRuntimeConfig
from data import generate_cv_batch, generate_dlrm_batch, generate_lm_batch
from models import DLRM, DLRMConfig, MoETransformerLM, ResNet50, TransformerConfig, TransformerLM


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
    parser.add_argument("--phase-a-ratio", type=float, default=0.3)
    parser.add_argument("--phase-a-interval-mul", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--async-queue-size", type=int, default=4)
    parser.add_argument("--async-timeout-s", type=float, default=1.0)
    parser.add_argument("--obs-window", type=int, default=50)
    parser.add_argument("--obs-report-every", type=int, default=10)
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
    runtime = PhaseAwareCheckpointRuntime(runtime_cfg, args.output_dir)
    # obs缓存与 phase 推断器
    obs = ObservationBuffer(args.obs_window)
    obs_worker = AsyncObservationWorker(obs)
    obs_log_path = os.path.join(args.output_dir, "logs", "obs_metrics.csv")
    os.makedirs(os.path.dirname(obs_log_path), exist_ok=True)
    obs_log_file = open(obs_log_path, "w", newline="")
    obs_writer = csv.DictWriter(
        obs_log_file,
        fieldnames=[
            "train_step",
            "elapsed_s",
            "step_time_s",
            "step_time_mean",
            "step_time_p95",
            "step_time_p99",
            "grad_norm_mean",
            "grad_nz_ratio_mean",
            "ckpt_latency_mean",
            "ckpt_latency_p95",
            "ckpt_latency_p99",
            "queue_depth_mean",
            "queue_depth_max",
            "ckpt_completion_rate_per_s",
            "staleness_mean",
            "staleness_max",
        ],
    )
    obs_writer.writeheader()
    start_time = time.perf_counter()
    prev_completed = runtime.num_completed

    try:
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            model.train()

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

            sample = ObsSample(
                train_step=step,
                elapsed_s=elapsed_s,
                step_time_s=step_time,
                grad_norm=None,
                grad_nz_ratio=None,
                ckpt_completed_latency_s=ckpt_latency,
                queue_depth=queue_depth,
                num_ckpt_issued=runtime.num_issued,
                num_ckpt_completed=runtime.num_completed,
                last_persisted_step=last_persisted,
                staleness_steps=staleness,
            )
            obs_worker.submit(sample)

            if step % args.obs_report_every == 0:
                stats = obs_worker.latest_stats()
                obs_writer.writerow(
                    {
                        "train_step": step,
                        "elapsed_s": elapsed_s,
                        "step_time_s": step_time,
                        **stats,
                    }
                )
                obs_log_file.flush()

            runtime.maybe_checkpoint(step, model, optimizer, {"step_time": step_time})
    finally:
        obs_worker.close()
        obs_log_file.close()
        runtime.close()
    print(f"Run complete. Logs saved to {runtime.log_path}")


if __name__ == "__main__":
    main()
