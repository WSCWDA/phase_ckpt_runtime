#!/usr/bin/env python3
"""Evaluate observation overhead by running multiple training configurations."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observation overhead evaluation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--obs-report-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--train-script", type=str, default="train.py")
    return parser.parse_args()


def mean_step_time(csv_path: Path) -> Optional[float]:
    if not csv_path.exists():
        return None
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        values: List[float] = []
        for row in reader:
            value = row.get("step_time")
            if value:
                values.append(float(value))
        if not values:
            return None
        return sum(values) / len(values)


def run_case(name: str, args: List[str]) -> Dict[str, Optional[float]]:
    start = time.perf_counter()
    subprocess.run(args, check=True)
    elapsed = time.perf_counter() - start
    return {"name": name, "wall_time_s": elapsed}


def main() -> None:
    args = parse_args()
    base_cmd = [
        sys.executable,
        args.train_script,
        "--steps",
        str(args.steps),
        "--obs-report-every",
        str(args.obs_report_every),
        "--output-dir",
        args.output_dir,
    ]

    cases = [
        ("baseline", base_cmd),
        ("aggregate", base_cmd + ["--obs-aggregate-stats"]),
        ("profiler", base_cmd + ["--profiler-enabled", "--profiler-wait", "1", "--profiler-warmup", "1", "--profiler-active", "2"]),
        ("dcgm", base_cmd + ["--dcgm-enabled"]),
    ]

    results = []
    for name, cmd in cases:
        print(f"[run] {name}: {' '.join(cmd)}")
        outcome = run_case(name, cmd)
        obs_csv = Path(args.output_dir) / "logs" / "obs_metrics.csv"
        outcome["mean_step_time_s"] = mean_step_time(obs_csv)
        results.append(outcome)

    print("\n=== Observation Overhead Summary ===")
    for row in results:
        print(f"{row['name']}: wall_time={row['wall_time_s']:.2f}s, mean_step_time={row['mean_step_time_s']}")


if __name__ == "__main__":
    main()
