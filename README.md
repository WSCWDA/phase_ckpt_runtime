# phase_ckpt_runtime

## 模块职能与实现细节（按模块拆解）

### 1) 观测层：`observation.py`
**职能**  
- 使用 PyTorch Profiler 周期性采样运行时信号，形成可用于决策的观测快照。  
- 对观测结果做滑动窗口统计，输出均值/方差/分位数/趋势等稳定指标。  

**实现要点**  
- `ProfilerObservation` 通过 `torch.profiler.profile` 配合 `schedule(wait/warmup/active)` 进行间歇式采样，避免持续 profiling 带来的高开销。【F:observation.py†L171-L199】  
- `step_begin()`/`step_end()` 标记训练步边界并推进 profiler 状态机，使采样与训练步对齐。【F:observation.py†L229-L244】  
- `on_profiler_trace()` 从 `key_averages()` 抽取 `train_step` / `optimizer_step` / `checkpoint_*` 统计，并计算 step_time、compute_time、checkpoint_total_time、overlap_ratio 等指标。【F:observation.py†L245-L292】  
- `window_stats()` 输出 P50/P95/P99、方差、趋势、progress rate 等窗口统计，用于上层适用性判定与策略控制。【F:observation.py†L294-L326】  

### 2) 适用性推断：`PhaseInference`（`phase_runtime.py`）
**职能**  
- 判断 async / delta / compression 三类 checkpoint 机制在当前运行条件下是否“适用”。  
- 不做语义训练阶段判定，仅做机制适用性推断。  

**实现要点**  
- `update()` 接收观测快照并维护滑动窗口，计算均值、P95、CV、趋势等统计量。【F:phase_runtime.py†L166-L199】  
- 三类适用性规则：  
  - **Async**：compute 覆盖 ckpt 写盘成本、队列非递增、计算稳定。【F:phase_runtime.py†L201-L215】  
  - **Delta**：delta/full 比例小、参数变化无峰值、delta 趋势稳定/下降。【F:phase_runtime.py†L216-L228】  
  - **Compression**：参数分布稳定、压缩误差低、loss 无恶化趋势。【F:phase_runtime.py†L230-L243】  
- 多信号投票 + 滞回 + 最小持续步数降低抖动，输出 `PhaseState`（async/delta/compression 适用性 + phase_id）。【F:phase_runtime.py†L245-L319】  

### 3) 策略控制层：`policy_controller.py`
**职能**  
- 基于 `PhaseInference` 的适用性输出与运行时指标，决定是否触发 checkpoint，并选择启用的机制（async/delta/compression）与参数（compression_level）。  

**实现要点**  
- `CheckpointPolicyController.decide()` 结合 staleness、上次 ckpt 延迟、软间隔等信息决定是否触发。【F:policy_controller.py†L46-L104】  
- 机制开关由适用性直接驱动，并生成结构化 `reason` 供日志与可解释性分析使用。【F:policy_controller.py†L105-L121】  

### 4) Checkpoint 运行时：`phase_runtime.py`
**职能**  
- 执行同步/异步 checkpoint 写盘。  
- 支持 full / delta / compressed payload 的构建，并记录策略决策与运行时指标到 CSV。  

**实现要点**  
- `_build_payload()` 生成三类 payload：  
  - **full**：保存完整模型与优化器状态。  
  - **delta**：与最近一次 full state 做差分并写入 `model_delta`。  
  - **compressed**：将 payload 序列化后使用 zlib 压缩存储。  
  【F:phase_runtime.py†L460-L501】  
- `maybe_checkpoint()` 调用策略控制器获取决策，决定是否写盘、走 sync 还是 async，并执行写盘或提交异步队列。【F:phase_runtime.py†L456-L579】  
- 在每步日志中追加 `policy_*` 字段，记录触发原因与机制选择，便于后续分析。【F:phase_runtime.py†L545-L573】  

### 5) 训练主循环：`train.py`
**职能**  
- 初始化 Observation、PhaseInference、PolicyController 与 Checkpoint Runtime，并在训练循环中按步调用。  

**实现要点**  
- 使用 `TrainConfig` 与可选的 JSON 配置文件集中管理参数，运行时初始化逻辑封装在 `init_runtime()` 中，避免与训练循环耦合。【F:train.py†L26-L241】  
- 训练步结束后发送 `ObservationEvent`（非阻塞）并写入 `obs_metrics.csv`，观测数据来自异步窗口统计与最新快照。【F:train.py†L225-L341】  
- 观测快照喂给 `PhaseInference.update()` 产生适用性状态，并将 `phase_state`/`observation_stats` 传入 `maybe_checkpoint()`。【F:train.py†L278-L341】  

### 6) 旧版 checkpoint 管理器：`checkpointing.py`
**职能**  
- 保留旧版的同步/异步 checkpoint 管理器，支持通过 `policy_controller` 或 `phase_inference` 决定 sync/async 模式。  

**实现要点**  
- `trigger()` 可在 policy 决策跳过时直接返回，并记录原因。【F:checkpointing.py†L64-L120】  

### 7) 绘图分析：`plot.py`
**职能**  
- 读取 runtime 日志 CSV，生成 step_time/staleness 曲线与滚动分位数图，用于离线分析。  
【F:plot.py†L1-L100】  
