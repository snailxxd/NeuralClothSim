# 单帧推理性能测试

在项目根目录使用同一 checkpoint 比较原始三帧前处理、缓存历史和 XLA 融合路径：

```bash
.venv/bin/python scripts/benchmark_inference.py \
  --config ncs/configs/mixamo_resume_2778.json \
  --gpu-id 0 --mode compare --warmup 50 --iterations 500 \
  --json results/gpu_optimized_benchmark.json
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
调用频率须与配置的 `data.fps` 一致；切换独立动画或瞬移时调用 `reset()`。
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
模型始终按配置 `data.fps` 的时间步推进，首帧和循环边界重置历史及 GRU。
默认使用 GPU 硬件渲染、GPU 法线、OpenGL 人体蒙皮和 XLA，不限速、不限总帧数，
关闭垂直同步，循环推理并绘制每帧，直到 Esc/关闭窗口。
动画会加速播放，但模型差分时间步保持不变。
按实际节奏播放使用 `--mode realtime`：处理落后时降低播放速度，不跳过姿态。

鼠标左键拖动旋转相机、滚轮缩放；空格暂停/继续，右方向键暂停并前进一帧，
R 重播，W 切换线框，Esc 退出。视口左上角显示滚动窗口 FPS（包含暂停时重绘），
窗口标题单独显示动画更新率和整帧处理时间。FPS 表示缓冲交换速率，不是显示器呈现速率。
`realtime` 模式请求垂直同步，`benchmark` 模式不等待模型帧率。

### 完整链路性能测试

```bash
.venv/bin/python scripts/view_realtime.py \
  --mode benchmark --no-xla --warmup 50 --iterations 1000 \
  --json results/viewer_benchmark.json --csv results/viewer_benchmark.csv

# 使用融合的逐帧路径，仍包含人体蒙皮和绘制
.venv/bin/python scripts/view_realtime.py \
  --mode benchmark --xla --warmup 50 --iterations 1000 \
  --json results/viewer_xla_benchmark.json

# 定时实时播放，结束时保存报告；启动、编译和预热不计入 300 秒
.venv/bin/python scripts/view_realtime.py --mode realtime --duration 300 \
  --json results/viewer_realtime.json

# 无窗口数值校验；可加 --device cpu 显式使用 CPU
.venv/bin/python scripts/view_realtime.py --validate-only
```

每次启动会校验最多 32 帧：人体/衣服与原始 `model.predict()` 对齐，
新旧单帧接口的衣服及 hidden 对齐，并验证重置可复现。
容差为 `rtol=1e-4, atol=1e-5`，默认关闭 TF32。校验失败不进入渲染。
`--xla` 编译失败直接报错，不自动改变测试路径。
XLA 默认开启，`--no-xla` 可以关闭。`--iterations 0` 表示不限总帧数（默认）；
`--duration 60` 可以做一分钟不限速测试，`--iterations 1000` 用于固定样本数比较。
无限运行只保留最近 10000 个逐帧样本，避免内存无限增长；JSON 的
`measured_frames` 记录总帧数，`retained_samples` 为保留窗口，耗时分位数基于保留窗口。

不限速模式关闭垂直同步，每个推理结果都提交绘制；驱动可能覆盖同步请求。
默认窗口为 1280×720，可用 `--width` / `--height` 修改。CSV 保留每帧实际绘制尺寸。
测量阶段不包括预热、启动和循环重置；报告单独记录模型加载、首次推理（含编译）
和窗口初始化耗时。

| 字段 | 含义 |
| --- | --- |
| `input_upload_ms` | 选择设备常驻输入的 CPU 开销；输入上传在测量前完成 |
| `inference_readback_ms` | 前处理、网络、衣服蒙皮，以及衣服顶点/骨骼矩阵 CPU 读回；TensorFlow 人体路径还包含人体蒙皮及读回 |
| `normals_ms` | 有限值检查；选用 CPU 法线时还包含法线计算 |
| `vertex_upload_cpu_ms` | 顶点打包和 OpenGL 上传调用的 CPU 耗时 |
| `mesh_update_submit_cpu_ms` | GPU 人体蒙皮/法线提交的 CPU 耗时 |
| `gpu_mesh_update_ms` | GPU 人体蒙皮（启用时）与顶点法线的 GPU 耗时 |
| `draw_submit_cpu_ms` | 清屏、绘制提交及时间查询的 CPU 耗时 |
| `gpu_draw_ms` | OpenGL 时间查询，覆盖清屏与绘制，不含顶点上传和显示 |
| `swap_ms` | 交换缓冲调用耗时，可能包含垂直同步等待 |
| `processing_ms` | 输入创建至交换缓冲返回，不含主动播放等待、重置和延迟读取查询 |
| `update_interval_ms` | 相邻动画帧交换缓冲返回之间的间隔，包含等待、暂停和重置 |

GPU 查询延后至少四帧读取，结束时收集尾部结果。CPU 与 GPU 阶段可以重叠，
各字段不可直接相加。整帧统计包含交换缓冲，但交换完成不代表屏幕实际呈现。
窗口最小化时没有绘制查询，该帧 `gpu_draw_ms` 为空。
JSON 包含逐帧样本、各耗时的 mean/P50/P95/P99、处理吞吐量、
超过模型帧预算的比例、显卡及 OpenGL renderer 信息。
`observed_updates_per_second` 包含循环重置、事件处理和播放等待，表示会话实际动画更新率。
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
