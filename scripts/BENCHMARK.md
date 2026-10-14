# 单帧推理性能测试

性能报告统一保存到 `benchmarks/<日期>/`，与 `results/` 中的动画 PC2 输出分开。
已有报告归档于 `benchmarks/2026-10-13/`：`inference/` 为推理基准，
`viewer/` 为渲染报告和截图，`diagnostics/` 为诊断结果；目录内 README 标注最终结果。
以下示例使用 `benchmarks/latest/`；需保留历史结果时改为对应日期。

在项目根目录使用同一 checkpoint 比较原始三帧前处理、缓存历史和 XLA 融合路径：

```bash
.venv/bin/python scripts/benchmark_inference.py \
  --config ncs/configs/mixamo_resume_2778.json \
  --gpu-id 0 --mode compare --warmup 50 --iterations 500 \
  --json benchmarks/latest/inference/gpu_optimized_benchmark.json
```

模式说明：

| 模式 | 计算范围 |
| --- | --- |
| `network` | 预计算特征 → GRU 和顶点解码 |
| `pipeline` | 每步重新计算三帧特征 → 网络 → 衣服蒙皮 |
| `cached` | 只计算当前帧，使用设备上的历史缓存 → 网络 → 衣服蒙皮 |
| `cached-xla` | 缓存路径使用 XLA 编译，减少中间算子和调度开销 |
| `sequence` | 原始 `model.predict()`，包括人体和衣服蒙皮 |
| `sequence-graph` | 原始 `model.predict()` 使用 `tf.function` 包装 |

`--mode both` 测试前三项；`compare` 测试前四项；`all` 测试全部。
XLA 需要当前 TensorFlow/GPU 环境支持。它的编译耗时不计入预热后的延迟。
TF32 默认关闭，各路径以原整段输出进行数值校验，保留严格容差。

计时包含 Python/TF 调用、计算、同步和衣服及 GRU 状态的 CPU 读回。
输入已经在设备上，特征历史缓存始终留在设备上。
每个动画循环开始时重新初始化缓存和 GRU；初始化在计时之外。
不包含碰撞修正、法线更新、渲染或磁盘写入。
单帧路径不计算人体网格。整段的 FPS 是吞吐量，不能当作单帧响应延迟。
`--frames` 是原动画长度内的上限，不会复制短动画来扩充序列。

## 在程序中使用缓存推理

`ncs/model/streaming.py` 中的 `StreamingNCS` 接收已经构建、加载权重的 NCS。
每个角色使用独立实例，按时间顺序调用：

```python
with tf.device('/GPU:0'):
    stream = StreamingNCS(model, motion=1.0)
    # 输入形状为 [1, 1, 全骨骼数, 4] 和 [1, 1, 3]。
    # 开始时重复首帧作为两帧历史，与原测试数据的 clip padding 一致。
    stream.reset(first_pose, first_trans, first_pose, first_trans)
    cloth, hidden = stream.step(current_pose, current_trans)
    # 下一帧只传下一帧姿态和平移，内部自动更新缓存和 hidden。
```

缓存保存上一帧静态特征、两帧关节位置和 GRU 状态，不修改模型权重。
相邻输入姿态的动画时间间隔须与配置的 `time_step` 一致；调用可快于墙钟。
切换独立动画或瞬移时调用 `reset()`。
`step_local()` 返回未蒙皮的局部衣物顶点、完整骨骼矩阵和更新后的 hidden，
供显示端插值并蒙皮；它与 `step()` 使用相同的固定步长特征缓存和状态更新。
同一实例不能并发处理不同角色。XLA 路径必须先 reset；不要依赖编译器保留运行时断言。

## ModernGL + GLFW 实时渲染

在项目根目录安装可选依赖，然后启动查看器：

```bash
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python scripts/view_realtime.py \
  --config ncs/configs/mixamo_resume_2778.json --gpu-id 0
```

默认 `--speed 1`：动画按真实时间原速播放，模型按训练配置
`config.time_step` 运行，绘制不限帧，默认关闭垂直同步。模型更新率与显示 FPS 分开。
直接加载测试动作和 checkpoint，无需 PC2 或预先生成整段衣物网格。
`--checkpoint`、`--sequence-index` 可覆盖默认权重和动作。

固定步长只计算当前姿态，复用设备上的历史特征和 GRU 状态；每个模型步只运行一次网络。
保留相邻两次未蒙皮衣物输出，OpenGL compute shader 在局部空间线性插值，再蒙皮。
人体使用同一个显示时刻：源姿态最短弧 quaternion SLERP、根平移线性插值，
按模型骨架做显示用 FK，避免直接线性混合骨骼矩阵导致旋转收缩。
显示端不调用 TensorFlow，不更新 GRU，也不再计算三帧预览输入。
骨骼变换在 CPU 计算，衣物形变插值、人体/衣物蒙皮和法线在 GPU 执行。

显示时间落后动画时钟一个模型时间步，例如 30 Hz 时为 33.3 ms 动画时间。
人体和衣物共享此显示时间；倍速时对应的墙钟延迟为 `time_step / speed`。
启动先显示首帧；循环接缝恢复初始模型状态，不混合独立循环的首尾形变。
低 FPS 时按顺序补算当前循环内的到期模型步，仍只保留两个结果。
若一次卡顿跨越多个循环，直接重置至当前循环，跳过已经不再显示的独立循环。
不增加碰撞修正，插值也不能保证模型原有穿模消失。

需要 OpenGL 4.3。模型推理默认使用 CUDA/XLA，完整固定步长路径可融合；
`--no-xla` 关闭融合，`--device cpu` 使用 CPU 推理，显示仍由 OpenGL 执行。
查看器只保留这一条显示路径；旧三帧预览、重复帧显示、世界空间顶点上传和
不限速显示/预取分支已移除，不再提供 `--mode`、`--pipeline`、`--iterations`、
`--normals`、`--body-skinning` 等切换旧路径的参数。

鼠标左键拖动旋转相机、滚轮缩放；空格暂停/继续，右方向键暂停并前进一个模型步，
R 重播，W 切换线框，Esc 退出。暂停时冻结动画时间，恢复不追赶暂停期间时间。
窗口标题固定，性能数据统一显示在 Dear ImGui 面板：

- 显示 FPS、帧间隔曲线、实际模型更新率。
- 动画时间、固定模型频率、显示延迟。
- 暂停/继续、单步、重播、0.1–4 倍速、线框、VSync 开关。
- 模型单步平均/P95 延迟、模型预算占比、串行单步速率估算。
- 显示姿态处理、上传、GPU 蒙皮/法线、绘制和交换各阶段耗时。

模型单步统计只包含真正发生的推理，不用无推理绘制帧的零值稀释。
预算占比为 `单步平均耗时 / 训练时间步`，串行速率估算为 `1000 / 单步平均毫秒数`；
二者包含同步读回，排除绘制、调度和循环重置，不能视为游戏引擎的保证帧率或角色容量。
面板保留最近 240 个显示帧，GPU 查询延后读取；未返回时显示 `--`。
CPU/GPU 阶段有重叠，不可直接相加。FPS 表示交换缓冲速率，不是显示器实际呈现速率。
`--no-ui` 关闭面板，可量化其开销；性能比较应保持该选项一致。

### 性能测试

```bash
# 原速播放 60 秒，测量模型延迟与完整显示链路
.venv/bin/python scripts/view_realtime.py --duration 60 \
  --json benchmarks/latest/viewer/realtime.json --csv benchmarks/latest/viewer/realtime.csv

# 原速播放，排除 ImGui 开销
.venv/bin/python scripts/view_realtime.py --no-ui --duration 60 \
  --json benchmarks/latest/viewer/realtime_no_ui.json

# 不限速纯推理性能测试使用独立脚本，不改变查看器播放行为
.venv/bin/python scripts/benchmark_inference.py --mode cached-xla --iterations 1000 \
  --json benchmarks/latest/inference/cached_xla.json

# 无窗口验证：局部输出蒙皮后与原模型对齐
.venv/bin/python scripts/view_realtime.py --validate-only
```

每次启动对最多 32 帧进行人体、衣物、GRU 和重置校验。
窗口建立后验证 OpenGL 人体/衣物蒙皮以及 NumPy 显示 FK 与 TensorFlow 一致。
容差 `rtol=1e-4, atol=1e-5`，
TF32 关闭。模型权重不变；校验或 XLA 编译失败直接报错。

查看器按到期模型步串行更新，不再预取或不限速推进模型状态。
`--duration` 限制测量会话墙钟时间，排除启动/编译/预热。
默认窗口 1280×720，可用 `--width` / `--height` 修改。

报告最多保留最近 10000 个显示帧；`total_frames` / `rendered_frames` 为总绘制数，
`simulation_steps` 为总推理步数，二者不再等同。
`render_fps` 为显示吞吐量，`simulation_updates_per_second` 为实际模型更新率，
原速播放时后者应接近 `1 / config.time_step`，而非算力上限。
逐帧 `steps_advanced` 表示本次显示前执行的模型步数，可以为 0 或大于 1。
JSON/CSV 记录 `animation_time_s`、`display_time_s`、循环内 `source_time_s` 和插值系数。

| 计时字段 | 范围 |
| --- | --- |
| `inference_readback_ms` | 本次显示前各真实模型步延迟的平均值；没有推理时为空 |
| `inference_samples_ms` | 各模型步的完整延迟列表，用于平均和分位数统计 |
| `inference_total_ms` | 本次显示前所有模型步延迟之和；无模型步时为零 |
| `inference_wait_ms` | 主线程取得模型结果的耗时，包含到期补算/循环重置 |
| `clip_reset_ms` | 恢复初始模型缓存的耗时，包含于等待区间 |
| `display_pose_ms` | 显示姿态 SLERP、根平移与 NumPy FK |
| `vertex_upload_cpu_ms` | 模型步变化时上传局部衣物端点，加每显示帧骨骼矩阵上传 |
| `mesh_update_submit_cpu_ms` | GPU 衣物插值、人体/衣物蒙皮与法线命令提交 |
| `gpu_mesh_update_ms` | 上述操作的 GPU 时间查询 |
| `draw_submit_cpu_ms` | 场景和 ImGui 绘制命令提交 |
| `gpu_draw_ms` | 清屏、场景和 ImGui 的 GPU 时间，不含上传或呈现 |
| `event_poll_ms` | 事件处理与 ImGui 面板构建，在 processing 区间之外 |
| `query_collect_ms` | 延迟收集 GPU 查询，在 processing 区间之外 |
| `swap_ms` | 缓冲交换调用，可能包含驱动/合成/同步等待 |
| `processing_ms` | 模型结果获取、显示姿态、上传、绘制至交换返回 |
| `update_interval_ms` | 相邻显示帧交换返回的间隔，包含事件处理和暂停等待 |

汇总提供 mean/P50/P95/P99；模型延迟分位数使用所有真实模型步样本。
每 30 秒采样 RSS 和 TensorFlow GPU 分配器；GPU 分配器不含 OpenGL/驱动显存。
窗口最小化时没有绘制查询，`gpu_draw_ms` 为空。

WSL 下需要可用的图形会话，启动时显示 OpenGL renderer。
默认拒绝 `llvmpipe` / `softpipe` 等软件渲染；`--render-device software` 用于诊断，
`--render-device auto` 接受系统选择，报告标记 `software_renderer`。
查看器自动为 WSL 加载 `/usr/lib/wsl/lib`，选择 Mesa D3D12 及推理显卡对应适配器；
`--render-adapter NVIDIA` 可覆盖适配器名称，不修改系统配置。
没有图形会话时用 `--validate-only`，不会自动切换离屏或 CPU。

### 2026-10-14 原速显示改造测量

RTX 3060 Ti、WSL Mesa D3D12、1280×720、同一 19 帧动作循环、1 倍速、XLA。
旧预览与新显示分别测量 10 秒，使用相同当前 ImGui 面板；旧路径禁用已移除的动态标题调用。

| 指标 | 旧三帧预览，ImGui | 新局部形变插值，ImGui | 新局部形变插值，无 UI |
| --- | ---: | ---: | ---: |
| 渲染 FPS | 56.60 | 86.33 | 185.64 |
| 推理调用平均延迟（含读回） | 5.36 ms（预览） | 2.09 ms（正式更新） | 1.76 ms（正式更新） |
| 10 秒正式模型更新数 | 284，另有 566 次预览 | 300 | 300 |
| 动画时钟 / 会话墙钟 | 原速 | 9.997 / 10.008 s | 9.998 / 10.003 s |

新路径未预计算整段衣物，只有到期时才顺序推理，并保留相邻两次局部输出。
人体、衣物和显示 FK 的关键帧对照最大误差均低于 `2e-6`。
ImGui 开启时 GPU 蒙皮/法线约 0.07 ms，面板和驱动 CPU 提交仍限制整体显示 FPS。
单步倒数只是模型速率估算；清理前的不限速显示路径（现已移除）在无 UI、300 步基准中为
178.83 个完整显示/推理步每秒，平均模型调用 1.70 ms。
报告位于 `benchmarks/2026-10-14/viewer/`，包含 `previous_preview.json`、
`fixed_local_ui.json`、`fixed_local_no_ui.json` 和 `local_benchmark_no_ui.json`。
短时单角色测量不代表游戏引擎集成后的多角色性能。

### 2026-10-13 历史优化测量（相关显示分支已移除）

RTX 3060 Ti、WSL Mesa D3D12、1280×720、同一 19 帧动作循环、XLA 开启。
缓冲使用 `STATIC_DRAW` 提示（仍允许逐帧写入及 compute shader 更新），
在此环境中比 `DYNAMIC_DRAW` 显著减少上传和 GPU 访问开销。
仅保留所需的 SSBO、顶点读取与缓冲读回屏障，避免全类别内存屏障。

| 指标 | 优化前 | 缓冲与流水线优化后 |
| --- | ---: | ---: |
| 会话实际更新率 | 98.97 FPS | 178.74 FPS |
| 每帧上传 CPU 耗时 | 2.013 ms | 0.054 ms |
| 渲染线程等待推理 | 约 1.85 ms | 0.014 ms |
| GPU 人体蒙皮与法线 | 0.661 ms | 0.057 ms |
| 交换阶段 | 4.274 ms | 3.967 ms |

优化后 1000 帧，工作线程与串行输出最大误差为 0；优化前为 20 秒、1980 帧样本。
同样使用新缓冲和缓存重置、关闭流水线的 15 秒对照为 143.03 FPS。
最终流水线 15 秒复核为 180.50 FPS；推理开始至交换完成平均 9.39 ms，
因此吞吐量收益不应表述为同等比例的响应延迟收益。
GPU 绘制未跳过、窗口分辨率未降低。
交换前插入 `glFinish` 的独立诊断显示：约 1.65 ms 是 GPU/驱动完成等待，
之后交换仍约 2.64 ms。原生 Wayland 对照未显著提升性能。
这些测试表明交换时间同时包含队列完成及呈现路径开销，
尚不能把后半部分完全归因于 WSL 桌面合成器。
`glFinish` 仅用于诊断，不保留在正常帧循环中。
