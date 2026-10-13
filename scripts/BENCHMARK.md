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
固定时间步接口的调用频率须与配置的 `data.fps` 一致；切换独立动画或瞬移时调用 `reset()`。
`preview_for_render()` / `preview_with_body()` 接收间隔为配置时间步的三帧输入，
只读当前 GRU 状态进行预览，不写回历史缓存或 hidden；固定网格帧仍使用 `step()` 提交。
同一实例不能并发处理不同角色。XLA 路径必须先 reset；不要依赖编译器保留运行时断言。

## ModernGL + GLFW 实时渲染

在项目根目录安装可选依赖，然后启动查看器：

```bash
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python scripts/view_realtime.py \
  --config ncs/configs/mixamo_resume_2778.json --gpu-id 0
```

查看器直接对每帧姿态进行推理，显示同帧的人体和衣服，无需生成 PC2。
默认读取配置测试列表中的第一条动作，使用 `checkpoints/<config name>` 权重。
`--checkpoint`、`--sequence-index` 和 `--motion` 可以覆盖这些设置。
默认使用 GPU 硬件渲染、GPU 法线、OpenGL 人体蒙皮和 XLA，运行 `interpolated` 模式。
按真实动画时间在源关键帧间插值输入姿态：四元数使用最短弧 SLERP，根平移线性插值；
每个未暂停的绘制帧都执行一次推理，不限制 FPS 或总帧数，动画保持正常播放速度。
这不是对衣服输出顶点插值，也不是重复播放预计算网格。
每次预览在动画时刻 `t-2Δt、t-Δt、t` 采样输入，`Δt` 始终为训练配置时间步。
只有完整模型帧推进持久 GRU 状态；逐绘制帧的预览 hidden 被丢弃，
避免渲染 FPS 改变模型记忆衰减速度，或微小/抖动间隔放大加速度噪声。
循环边界恢复原始首帧缓存，并按固定网格推进到当前时刻，不跨动画接缝求差分。
启动时校验预览关键帧与固定步长输出一致、重复预览可复现且不修改持久状态。
预览的三帧前处理不使用整段 XLA，以保持 FK 与差分的严格 FP32 数值行为；
`--xla` 仍用于网络及固定步长提交路径。
此修复针对时间步和状态更新导致的不稳定，不包含额外碰撞修正，仍可能有模型原有穿模。
默认关闭垂直同步，`--vsync` 可单独限制绘制刷新率。
`--mode benchmark` 使用设备常驻、固定时间步输入并不限速推进，每次推理都绘制，
动画会随吞吐量加速；适合比较推理/渲染最高性能。
需要固定速率播放时显式使用 `--mode realtime`：动画按配置 `data.fps` 推进，
两个模型时间步之间重复绘制当前结果；绘制慢时补算到期推理步，仅显示最新结果。
该模式暂停时冻结动画时间，恢复时不追赶暂停期间的时间，重播重新开始计时。
`--vsync` 可单独限制绘制刷新率。
benchmark 模式默认开启单帧预取：一个工作线程顺序执行下一帧推理，
主线程执行当前帧的 OpenGL 上传、绘制和交换。最多预取一帧，不跳过中间结果。
下一帧在当前帧绘制命令提交后启动，与交换等待重叠，减少 Python GIL 争用。
动画循环重置直接恢复首帧的设备缓存，不重复计算初始骨骼特征。
不限速原始帧性能测试需显式使用 `--mode benchmark`。

鼠标左键拖动旋转相机、滚轮缩放；空格暂停/继续，右方向键暂停并前进一帧，
R 重播，W 切换线框，Esc 退出。默认启用 Dear ImGui 面板，支持暂停/继续、单步、
重播、0.1–4 倍播放速度、线框和垂直同步开关；操作面板时不会拖动或缩放相机。
`--speed 2` 可设置初始倍速。倍速只改变动画时钟，插值推理的差分时间步和持久 GRU
提交网格保持训练配置值；`benchmark` 模式忽略倍速控制。
面板显示滚动窗口 FPS（包含暂停时重绘）、帧间隔曲线、推理更新率、动画时间，
以及最近 240 次更新的各阶段平均/P95 耗时，包括推理、输入插值、上传、GPU 蒙皮/法线、
绘制和缓冲交换。GPU 查询延后读取，未返回时显示 `--`。
CPU 提交与 GPU 执行存在重叠，各阶段不可直接相加；Frame processing 不包含 Events + UI build。
窗口标题单独显示动画更新率和整帧处理时间。FPS 表示缓冲交换速率，不是显示器呈现速率。
面板构建和绘制也有开销，GPU draw + UI 包含面板绘制。比较纯推理/场景渲染性能时
可用 `--no-ui` 关闭面板，并在不同运行间保持此选项一致。
各模式默认均关闭垂直同步；`benchmark` 和 `interpolated` 模式均不等待模型帧率。

### 完整链路性能测试

```bash
.venv/bin/python scripts/view_realtime.py \
  --mode benchmark --no-xla --warmup 50 --iterations 1000 \
  --json benchmarks/latest/viewer/viewer_benchmark.json --csv benchmarks/latest/viewer/viewer_benchmark.csv

# 使用融合的逐帧路径，仍包含人体蒙皮和绘制
.venv/bin/python scripts/view_realtime.py \
  --mode benchmark --xla --warmup 50 --iterations 1000 \
  --json benchmarks/latest/viewer/viewer_xla_benchmark.json

# 同一渲染路径关闭推理预取，作串行对照（保留缓冲/重置优化）
.venv/bin/python scripts/view_realtime.py --mode benchmark --no-pipeline --iterations 1000 \
  --json benchmarks/latest/viewer/viewer_serial_benchmark.json

# 定时实时播放，结束时保存报告；启动、编译和预热不计入 300 秒
.venv/bin/python scripts/view_realtime.py --mode realtime --duration 300 \
  --json benchmarks/latest/viewer/viewer_realtime.json

# 无窗口数值校验；可加 --device cpu 显式使用 CPU
.venv/bin/python scripts/view_realtime.py --validate-only
```

每次启动会校验最多 32 帧：人体/衣服与原始 `model.predict()` 对齐，
新旧单帧接口的衣服及 hidden 对齐，并验证重置可复现。
启用流水线时额外比较工作线程与串行输出，覆盖动画循环边界。
容差为 `rtol=1e-4, atol=1e-5`，默认关闭 TF32。校验失败不进入渲染。
`--xla` 编译失败直接报错，不自动改变测试路径。
XLA 默认开启，`--no-xla` 可以关闭。`--iterations 0` 表示不限总帧数（默认）；
`--mode benchmark --duration 60` 可以做一分钟不限速测试，
`--mode benchmark --iterations 1000` 用于固定样本数比较。
无限运行只保留最近 10000 个逐帧样本，避免内存无限增长；JSON 的
`measured_frames` 记录总帧数，`retained_samples` 为保留窗口，耗时分位数基于保留窗口。

不限速模式关闭垂直同步，每个推理结果都提交绘制；驱动可能覆盖同步请求。
默认窗口为 1280×720，可用 `--width` / `--height` 修改。CSV 保留每帧实际绘制尺寸。
测量阶段不包括预热和启动；报告单独记录模型加载、首次推理（含编译）
和窗口初始化耗时。

| 字段 | 含义 |
| --- | --- |
| `input_upload_ms` | 选择设备常驻输入的 CPU 开销；输入上传在测量前完成 |
| `sample_upload_ms` | interpolated 模式的输入 SLERP、平移插值及张量创建 CPU 耗时，已包含于 inference_wait |
| `inference_readback_ms` | 前处理、网络、衣服蒙皮，以及衣服顶点/骨骼矩阵 CPU 读回；TensorFlow 人体路径还包含人体蒙皮及读回 |
| `inference_wait_ms` | 渲染线程取得当前帧的等待耗时；流水线中通常远小于实际推理耗时 |
| `clip_reset_ms` | 此帧推理前恢复初始缓存的耗时，包含于等待区间或与当前帧绘制重叠 |
| `prefetch_age_ms` | 渲染线程取出结果时，结果完成至今的时间 |
| `normals_ms` | 有限值检查；选用 CPU 法线时还包含法线计算 |
| `vertex_upload_cpu_ms` | 顶点打包和 OpenGL 上传调用的 CPU 耗时 |
| `mesh_update_submit_cpu_ms` | GPU 人体蒙皮/法线提交的 CPU 耗时 |
| `gpu_mesh_update_ms` | GPU 人体蒙皮（启用时）与顶点法线的 GPU 耗时 |
| `draw_submit_cpu_ms` | 清屏、绘制提交及时间查询的 CPU 耗时 |
| `gpu_draw_ms` | OpenGL 时间查询，覆盖清屏与绘制，不含顶点上传和显示 |
| `prefetch_submit_ms` | 提交下一帧推理任务的 CPU 耗时 |
| `event_poll_ms` | 当前更新前一次窗口事件处理耗时，在 processing 区间之外 |
| `query_collect_ms` | 当前帧交换后收集较早 GPU 查询的 CPU 耗时，在 processing 区间之外 |
| `swap_ms` | 交换缓冲调用耗时，可能包含垂直同步等待 |
| `processing_ms` | 渲染线程开始取得当前帧至交换缓冲返回，包含推理/重置等待，不含主动播放等待和延迟读取查询 |
| `inference_to_swap_ms` | 当前帧实际推理开始至交换缓冲返回，反映预取带来的结果等待时间 |
| `update_interval_ms` | 相邻动画帧交换缓冲返回之间的间隔，包含等待、暂停和重置 |

GPU 查询延后至少四帧读取，结束时收集尾部结果。CPU 与 GPU 阶段可以重叠，
各字段不可直接相加。整帧统计包含交换缓冲，但交换完成不代表屏幕实际呈现。
流水线中的 `inference_readback_ms` 与当前/前一帧渲染重叠，不能加到
`processing_ms` 上；渲染线程分段计时使用 `inference_wait_ms`。
吞吐量提升不等于响应延迟等比例下降，应同时查看 `inference_to_swap_ms`。
`--mode realtime` 默认使用串行路径，避免提前推进姿态；暂停保留至多一个预取结果，
重播会等待在途推理完成后丢弃旧结果，再恢复首帧状态。
窗口最小化时没有绘制查询，该帧 `gpu_draw_ms` 为空。
JSON 包含逐帧样本、各耗时的 mean/P50/P95/P99、处理吞吐量、
超过模型帧预算的比例、显卡及 OpenGL renderer 信息。
`observed_updates_per_second` 包含循环重置、事件处理和播放等待，表示会话实际动画更新率。
实时播放额外报告 `simulation_steps` / `simulation_updates_per_second`（含补算的模型步）
及 `rendered_frames` / `render_fps`（含重复绘制的视口刷新），每行 `steps_advanced`
说明本次显示前实际推进的模型步数。算力不足以完成配置更新率时仍可能落后真实时间。
输入插值模式中 `animation_time_s` 为去除暂停时间后的动画时钟，
`simulation_steps` 表示逐绘制帧预览推理次数，更新率可以高于源动作帧率；
`committed_model_steps` 单独记录固定时间步的状态推进次数。
运行期间每 30 秒记录进程 RSS 和 TensorFlow GPU 分配器的 current/peak；
GPU 分配器数值不包含 OpenGL/驱动显存，进程 RSS 包含保留的逐帧报告样本。
实时模式的吞吐量含同步等待，不能用于推断不限速处理能力。

`StreamingNCS.step_with_body()` 返回 `(body, cloth, hidden)`，网格形状为
`[B, 1, V, 3]`，并复用同一次完整骨骼变换；原 `step()` 的接口不变。
`step_for_render()` 返回 `(cloth, full_matrices, hidden)`，骨骼矩阵形状为
`[B, 1, J, 3, 4]`，供渲染器蒙皮使用。
默认只把衣服位置和骨骼矩阵从 TensorFlow 读回并上传 OpenGL，人体静态顶点/权重
初始化时上传一次。OpenGL compute shader 执行人体蒙皮和面积加权顶点法线。
需要 OpenGL 4.3；启动时逐帧校验 OpenGL 人体与 TensorFlow 人体输出。
`--body-skinning tensorflow` 可用于对照完整人体顶点传输；`--normals cpu`
切换 CPU 法线并自动使用 TensorFlow 人体，仅需要 OpenGL 3.3。
碰撞修正、视频编码和插值不在测试中。

WSL 下需要可用的 WSLg/X11/Wayland 会话。查看启动输出中的 `OpenGL renderer`：
`llvmpipe` / `softpipe` 表示软件渲染，即使 TensorFlow 能访问 CUDA，
也不能把此时的绘制结果当作显卡渲染性能。默认拒绝软件渲染，
可显式用 `--render-device software` 诊断，或 `--render-device auto` 接受系统选择。
报告会标注 `software_renderer`。
查看器自动为 WSL 加载 `/usr/lib/wsl/lib`，选择 Mesa D3D12 驱动和推理显卡对应的适配器；
使用 `--render-adapter NVIDIA` 可指定适配器名称子串。此配置仅作用于当前进程，
不修改系统配置。参见 [Mesa D3D12 文档](https://docs.mesa3d.org/drivers/d3d12.html)
及 [WSLg GPU 选择说明](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg)。
没有显示环境时，可用 `--validate-only` 验证模型；不自动切换离屏或 CPU 路径。

最高性能以实际 FPS 和完整链路延迟评估。单角色时序推理、CPU/GPU 传输、
驱动调度和桌面合成都可能成为瓶颈，取消限帧不能保证 GPU 占用 100%。
增加无关绘制工作虽能提高占用率，却不会提高每帧推理吞吐量。

### 2026-10-13 优化测量

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
