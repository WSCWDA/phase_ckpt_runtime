import argparse
import csv
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """解析绘图参数。"""
    parser = argparse.ArgumentParser(description="Plot phase-aware checkpoint metrics")
    parser.add_argument("csv_path", type=str, help="Path to run CSV log")
    parser.add_argument("--window", type=int, default=20, help="Rolling window for percentile")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    return parser.parse_args()


def rolling_percentiles(values: List[float], window: int, percentile: float) -> List[float]:
    """计算滑动窗口分位数。"""
    results = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_vals = values[start : idx + 1]
        results.append(float(np.percentile(window_vals, percentile)))
    return results


def main() -> None:
    """从 CSV 生成图表。"""
    args = parse_args()
    steps = []
    queue_depth = []
    staleness = []
    step_times = []

    with open(args.csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            steps.append(int(row["step"]))
            queue_depth.append(int(float(row["queue_depth"])))
            staleness.append(int(float(row["staleness_steps"])))
            step_times.append(float(row["step_time"]))

    out_dir = args.out_dir or os.path.dirname(args.csv_path)
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_time = plt.subplots(figsize=(10, 4))
    ax_time.plot(steps, step_times, color="#1f77b4", linewidth=1.6, label="step_time")
    ax_time.set_xlabel("step")
    ax_time.set_ylabel("step_time (s)")
    ax_time.set_title("Step time vs staleness over steps")
    ax_time.grid(True, which="both", linestyle="--", alpha=0.4)

    ax_stale = ax_time.twinx()
    ax_stale.plot(steps, staleness, color="#ff7f0e", linewidth=1.6, label="staleness_steps")
    ax_stale.set_ylabel("staleness_steps")

    lines = ax_time.get_lines() + ax_stale.get_lines()
    labels = [line.get_label() for line in lines]
    ax_time.legend(lines, labels, loc="upper left", frameon=True)

    fig.tight_layout()
    queue_path = os.path.join(out_dir, "step_time_staleness.png")
    fig.savefig(queue_path)
    plt.close(fig)

    p95 = rolling_percentiles(step_times, args.window, 95)
    p99 = rolling_percentiles(step_times, args.window, 99)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, p95, label="rolling_p95", color="#2ca02c", linewidth=1.8)
    ax.plot(steps, p99, label="rolling_p99", color="#d62728", linewidth=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("step_time (s)")
    ax.set_title(f"Rolling step time percentiles (window={args.window})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    step_path = os.path.join(out_dir, "step_time_p95_p99.png")
    fig.savefig(step_path)
    plt.close(fig)

    print(f"Saved plots to {queue_path} and {step_path}")


if __name__ == "__main__":
    main()
