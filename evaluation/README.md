# Observation Overhead Evaluation

This folder contains scripts to quantify observation overhead by comparing
runtime and step-time statistics with different observation configurations.

## Quick Start

```bash
python evaluation/run_observation_overhead.py --steps 200 --obs-report-every 10
```

The script runs multiple training configurations (baseline, aggregation,
profiler, DCGM) and reports wall-clock and step-time metrics.
