# 测试说明（基于当前代码实现）

本文档给出当前项目可执行的测试与验证步骤，覆盖：
- 基础训练可运行性
- Observation 开销评估
- Policy/Checkpoint 单元测试
- 常见故障排查

## 1. 环境准备

建议 Python 3.8+。

如果要运行训练与单元测试，需安装 PyTorch；
如果要启用资源监控，建议安装 `psutil`；
如果要启用 DCGM，需要系统已安装并可访问 DCGM 相关库/服务。

## 2. 基础可运行性测试

### 2.1 最小训练运行

```bash
python train.py --steps 50 --obs-report-every 10
```

预期：
- 控制台打印 `Using device: ...`
- 正常结束并输出 `Run complete. Logs saved to ...`
- 生成以下日志文件：
  - `runs/logs/run_*.csv`
  - `runs/logs/obs_metrics.csv`
  - `runs/logs/phase_inference.csv`

### 2.2 启用 observation 聚合统计

```bash
python train.py --steps 50 --obs-report-every 10 --obs-aggregate-stats
```

预期：
- `obs_metrics.csv` 中出现窗口统计字段（如 `step_time_mean`、`step_time_p95`、`queue_depth_mean` 等）。

## 3. Observation 开销评估

项目已提供评估脚本：`evaluation/run_observation_overhead.py`。

### 3.1 一键评估

```bash
python evaluation/run_observation_overhead.py --steps 200 --obs-report-every 10
```

脚本会按以下 case 依次运行并输出摘要：
- `baseline`
- `aggregate`
- `profiler`
- `dcgm`

输出指标：
- `wall_time_s`（总耗时）
- `mean_step_time_s`（平均步耗时）

### 3.2 结果解释建议

- `aggregate` 对比 `baseline`：评估窗口统计开销
- `profiler` 对比 `baseline`：评估 profiler 校准开销
- `dcgm` 对比 `baseline`：评估硬件遥测开销

> 说明：`dcgm` case 在 DCGM 不可用时应退化为近似 baseline 行为（不会阻塞训练）。

## 4. 单元测试

### 4.1 Checkpoint payload 测试

```bash
python -m unittest tests/test_policy_checkpointing.py
```

该测试验证：
- delta checkpoint payload
- compressed checkpoint payload

若环境缺少 PyTorch，测试会自动跳过。

## 5. 可选功能测试

### 5.1 Profiler 间歇采样

```bash
python train.py \
  --steps 100 \
  --obs-report-every 10 \
  --profiler-enabled \
  --profiler-wait 1 \
  --profiler-warmup 1 \
  --profiler-active 2
```

### 5.2 DCGM 监控

```bash
python train.py \
  --steps 100 \
  --obs-report-every 10 \
  --dcgm-enabled \
  --dcgm-poll-interval-s 2.0 \
  --dcgm-window-size 60
```

## 6. 常见问题排查

### 6.1 `obs_metrics.csv` 写入字段报错

若出现 `dict contains fields not in fieldnames`，通常是日志 schema 与观测输出字段不一致。
请确保使用最新代码（`train.py` 的 `fieldnames` 已覆盖当前 observation 输出）。

### 6.2 Profiler 线程错误

当前实现将 profiler 生命周期与 step 推进集中在 `ObservationWorker` 的受控路径中，训练主循环仅发事件。
若仍出现环境相关错误，建议先关闭 profiler 验证主流程：

```bash
python train.py --steps 50
```

### 6.3 DCGM 不可用

如果系统未安装/未启动 DCGM，`--dcgm-enabled` 不应导致训练失败。
可先不启用 DCGM，或确认 DCGM 服务状态。

## 7. 推荐最小回归集合

每次改动后建议至少执行：

```bash
python train.py --steps 50 --obs-report-every 10
python train.py --steps 50 --obs-report-every 10 --obs-aggregate-stats
python -m unittest tests/test_policy_checkpointing.py
```

若关注性能回归，再执行：

```bash
python evaluation/run_observation_overhead.py --steps 200 --obs-report-every 10
```
