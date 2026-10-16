# 运行环境（Linux / WSL2）

项目使用 Python 3.11、TensorFlow 2.15.1 / Keras 2、NumPy 1.26.4、SciPy 1.11.4 和 Ray 2.9.3。依赖版本见 `requirements.txt`。

本机已验证：RTX 3060 Ti 上的 Mixamo / SMPL 示例前向计算、单步训练与预测，CPU 上的单步 `fit` / 验证 / checkpoint 保存与权重恢复，以及 Blender 3.0.1 后台启动。`pip check` 无依赖冲突。未执行完整的 5000 epoch 训练。

TensorFlow 2.15 保留项目使用的 Keras 2 接口；其 GPU 依赖为 CUDA 12.2 / cuDNN 8.9，可通过 pip 安装，无需另外安装系统 CUDA Toolkit。NVIDIA 驱动由宿主系统提供。参考 [TensorFlow 版本兼容表](https://www.tensorflow.org/install/source#gpu) 和 [Keras 版本说明](https://keras.io/getting_started/)。

## 使用当前环境

在项目根目录激活已经安装的环境：

```bash
conda activate "$PWD/.venv"
python -m pip check
python scripts/check_environment.py --require-gpu
```

环境检查使用仓库自带的 Mixamo 和 SMPL 数据，分别执行模型构建、前向计算、一次训练更新及预测，不保存训练结果。

训练须从 `ncs` 目录启动：

```bash
cd ncs
python train.py --config configs/mixamo.json --gpu_id 0
# 或使用另一套示例
python train.py --config configs/smpl.json --gpu_id 0
```

默认每个 epoch 输出一次指标，省略模型结构表和已知重复警告；如需详细 TensorFlow 日志、模型结构和 batch 进度，在训练命令末尾加 `--verbose`。TensorFlow 原生错误日志仍会显示。优化器关闭 XLA 编译，以避免当前 eager 训练流程中反复追踪不同形状参数的更新。

动态编码器使用可复用的 `tf.function` 计算图：本机 TensorFlow 2.15.1 的无偏置 GRU 在 eager 模式下每次前向计算会持续保留约 512 KiB GPU 内存，训练和验证都会累积。碰撞检测仍在 eager 模式下调用 SciPy。全连接层通过 `tensordot` 收缩特征维，避免广播矩阵乘法在反向传播时生成逐样本的权重梯度临时张量。模型参数结构和 checkpoint 路径不变，已有训练进程需要重启才能应用修复。

数值与显存回归检查（在项目根目录运行；GPU 检查使用临时目录，不覆盖已有实验）：

```bash
.venv/bin/python scripts/test_training_memory.py
NCS_TEST_GPU=1 .venv/bin/python scripts/test_training_memory.py
.venv/bin/python scripts/test_training_restart.py
```

GPU 检查运行 dress 模型的 24 轮短训练，包含验证、TensorBoard、恢复 checkpoint、最佳权重和完整模型保存，并检查预热后的实际分配显存是否稳定。

中断后继续训练时，可复制配置为新的文件名，在 `experiment` 中设置 `checkpoint` 为旧实验名、`initial_epoch` 为 checkpoint 已完成的轮数。训练会恢复模型与 Adam 优化器状态；例如 `initial_epoch: 207`、`epochs: 5000` 会从第 208 轮继续到第 5000 轮，新结果保存在新配置文件名对应的目录中。日志可能先于 checkpoint 写入，应以实际保存状态为准。

`train.py` 默认在独立子进程中训练。遇到 GPU OOM 时，子进程退出以释放显存，外层进程自动启动新的训练进程，从最后成功保存的轮数继续。每轮验证结束后，先在 `checkpoints/[实验名]/recovery/` 保存模型、Adam 状态和 `state.json`，再导出原有 SavedModel；恢复记录只在 checkpoint 写入成功后提交，因此不会把尚未保存的轮数当成完成。保留最近两份恢复 checkpoint，增加约两份 checkpoint 的磁盘占用。普通错误和 Ctrl-C 不会触发自动重启。

原有启动命令无需修改。对于明确配置为同名续训（`checkpoint` 等于配置文件名）的实验，如果恢复记录已经存在，以记录中的轮数为准，覆盖配置中的 `initial_epoch`，无需每次手动更新。旧 SavedModel 没有恢复记录时，首次仍使用配置里的 `initial_epoch`，并在开始训练前建立恢复快照。OOM 发生在一轮中途时，该轮从头重跑。最多自动重启 100 次；连续两次 OOM 都无法完成下一轮时停止，提示降低 `batch_size`。可用 `--max-oom-restarts 0` 禁用自动重试，或指定其他上限。已经运行的旧训练进程需要下次重新启动后才会使用此机制。

训练还会将验证指标 `val_m/Loss` 最低的模型权重保存为 `checkpoints/[实验名]/best.weights.h5`。只有有限且严格降低的验证 Loss 才会更新该文件；先写临时文件，写入成功后替换。文件内同时记录最佳轮数（从 1 开始）和验证 Loss，同名续训或 OOM 重启会读取该记录，继续与历史最佳值比较。原有 SavedModel 仍保存最新轮，`recovery/` 仍用于恢复最新模型和 Adam 状态；最佳文件用于加载模型权重，自动续训仍使用 `recovery/`。新增机制从启用后的第一轮验证开始记录，无法恢复此前已覆盖的最优权重。

实时查看最佳模型（在项目根目录运行，默认即加载该文件）：

```bash
.venv/bin/python -m view_realtime --config ncs/configs/mannequin_dress_multi.json
```

`predict.py`、实时查看器和性能基准默认都加载 `checkpoints/[实验名]/best.weights.h5`；添加 `--weights latest` 可改用最后一轮模型。实时查看器和性能基准仍可用 `--checkpoint` 显式指定其他权重。最佳轮数和 Loss 可通过 `h5py.File` 读取文件属性 `ncs_best_epoch` 和 `ncs_best_val_loss`。已有训练进程需要重新启动才能启用最佳模型保存。

训练完成后，使用对应配置和生成的 checkpoint 预测：

```bash
python predict.py --config configs/mixamo.json --gpu_id 0
```

仓库未附带预训练 checkpoint，预测需要先训练。原始训练和预测入口均要求 GPU。

## 重建环境

在项目根目录执行：

```bash
conda env create --prefix ./.venv --file environment.yml
conda activate "$PWD/.venv"
```

也可分步安装：

```bash
conda create --prefix ./.venv python=3.11 pip -y
conda activate "$PWD/.venv"
python -m pip install -r requirements-gpu.txt
```

国内网络可在 pip 命令末尾加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。
仅做 CPU 检查时，安装 `requirements.txt`，然后运行 `python scripts/check_environment.py`。

## 实时可视化

固定步长推理可直接用 ModernGL + GLFW 查看器显示，无需 Blender 或 PC2 文件。
默认原速播放，显示帧在局部空间插值衣物形变并在 GPU 蒙皮，模型更新与显示 FPS 分开。
可选渲染依赖包含 Dear ImGui 面板，用于 FPS、播放速度控制和性能统计：

```bash
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python -m view_realtime --xla
```

模块结构和测试命令见 [view_realtime/README.md](view_realtime/README.md)，实时播放、键鼠操作及性能测试方法见 [scripts/BENCHMARK.md](scripts/BENCHMARK.md#moderngl--glfw-实时渲染)。
需要可用的图形会话；CUDA 可用不代表 OpenGL 使用硬件加速，启动时会输出 renderer。

## Blender 可视化

项目本地安装的 Blender 3.0.1 位于 `.tools/blender-3.0.1-linux-x64/`，使用其自带 Python，无需在 Conda 中安装 `bpy`。

在项目根目录执行：

```bash
export PATH="$PWD/.tools/blender-3.0.1-linux-x64:$PATH"
blender --python view.py -- ncs/configs/mixamo.json
```

便携包来自 [Blender 3.0 官方发布目录](https://download.blender.org/release/Blender3.0/)，并用官方 SHA256 校验。重建环境时需另行下载 `blender-3.0.1-linux-x64.tar.xz` 并解压至 `.tools/`；Conda 环境文件只安装 Python 与训练依赖。

在 Codex 受限沙箱内，GPU 可能显示为不可访问；应在普通 WSL 终端中运行 GPU 检查和训练。
