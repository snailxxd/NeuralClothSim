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

训练默认启用显存按需增长，不再固定限制为 6144 MB，适用于 96 GB H20 等大显存 GPU。按需增长不预先占满整卡，也不为其他进程预留固定空间。共享 GPU 时可添加 `--gpu-memory-limit-mb 81920`，将每张 GPU 的分配上限设为 80 GiB；固定上限与按需增长二选一。显存配置在首次 GPU 运算前应用。

默认每个 epoch 输出一次指标，省略模型结构表和已知重复警告；如需详细 TensorFlow 日志、模型结构和 batch 进度，在训练命令末尾加 `--verbose`。进度条沿用原有 `--verbose` 开关，无额外进度参数。每轮指标同时写入 `logs/[实验名]/metrics.csv` 和 TensorBoard；CSV 的 `epoch` 从 0 开始。batch 进度条仅显示总 Loss；每轮结束显示验证总 Loss，并分行展示验证集的静态、惯性、布料、弯曲、碰撞、重力及固定约束分项。完整指标也写入 CSV 和 TensorBoard，避免进度行过长。TensorFlow 原生错误日志仍会显示。优化器关闭 XLA 编译，以避免当前 eager 训练流程中反复追踪不同形状参数的更新。

动态编码器使用可复用的 `tf.function` 计算图：本机 TensorFlow 2.15.1 的无偏置 GRU 在 eager 模式下每次前向计算会持续保留约 512 KiB GPU 内存，训练和验证都会累积。碰撞检测仍在 eager 模式下调用 SciPy。全连接层通过 `tensordot` 收缩特征维，避免广播矩阵乘法在反向传播时生成逐样本的权重梯度临时张量。模型参数结构和 checkpoint 路径不变，已有训练进程需要重启才能应用修复。

数值与显存回归检查（在项目根目录运行；GPU 检查使用临时目录，不覆盖已有实验）：

```bash
.venv/bin/python scripts/test_training_memory.py
NCS_TEST_GPU=1 .venv/bin/python scripts/test_training_memory.py
.venv/bin/python scripts/test_training_checkpoint.py
.venv/bin/python scripts/test_training_resilience.py
```

GPU 检查运行 dress 模型的 24 轮短训练，包含验证、TensorBoard、恢复 checkpoint、最佳权重和完整模型保存，并检查预热后的实际分配显存是否稳定。

中断后继续训练时，可复制配置为新的文件名，在 `experiment` 中设置 `checkpoint` 为旧实验名、`initial_epoch` 为 checkpoint 已完成的轮数。训练会恢复模型与 Adam 优化器状态；例如 `initial_epoch: 207`、`epochs: 5000` 会从第 208 轮继续到第 5000 轮，新结果保存在新配置文件名对应的目录中。日志可能先于 checkpoint 写入，应以实际保存状态为准。

`train.py` 直接在当前进程训练。GPU OOM、普通错误及 Ctrl-C 均直接终止，不启动子进程或自动重试。每轮验证结束后，先在 `checkpoints/[实验名]/recovery/` 保存模型、Adam 状态和 `state.json`，再导出 SavedModel；恢复记录只在 checkpoint 写入成功后提交。保留最近两份恢复 checkpoint。

手动重新运行前，将 `experiment.checkpoint` 设为配置文件名对应的实验名。对于这种明确的同名续训，如果恢复记录存在，就以其中的轮数为准，覆盖配置中的 `initial_epoch`。旧 SavedModel 没有恢复记录时，使用配置中的 `initial_epoch`，并在开始训练前建立恢复快照。中断时尚未完成的 epoch 会在下次手动续训时从头执行。遇到 OOM，应先检查 GPU 占用、显存上限及 `batch_size`，再手动启动。

`m/Loss` 现在记录包含惯性项的总目标；`m/StaticLoss`、`m/InertiaLoss` 分别记录静态与惯性损失，`m/ClothLoss`、`m/BendingLoss`、`m/CollisionLoss`、`m/PinningLoss` 是按配置加权后的静态分项，`m/Gravity` 是重力势能，原有 Stretch/Shear/Bending/Collision/Inertia 误差指标仍保留。指标按实际样本数加权，避免末尾小 batch 占过高权重。

训练还会将验证指标 `val_m/Loss` 最低的模型权重保存为 `checkpoints/[实验名]/best.weights.h5`。只有有限且严格降低的验证 Loss 才会更新该文件；先写临时文件，写入成功后替换。文件内同时记录最佳轮数（从 1 开始）和验证 Loss，手动续训会读取该记录，继续与历史最佳值比较。旧最佳权重使用静态 Loss，缺少 `ncs_best_objective=total_loss_v1` 标记；升级后保留原文件，但重新开始总 Loss 的最佳值比较，并输出提示，避免直接比较两种指标。原有 SavedModel 仍保存最新轮，`recovery/` 仍用于恢复最新模型和 Adam 状态；最佳文件用于加载模型权重，手动续训使用 `recovery/`。新增机制从启用后的第一轮验证开始记录，无法恢复此前已覆盖的最优权重。


训练在应用梯度前检查总损失及梯度是否为有限值；保存前再次检查模型权重、Adam 状态和 epoch 指标。出现 NaN/Inf 时直接报错退出，保留之前已提交的恢复记录。恢复状态使用相对路径，兼容旧绝对路径，并检查 index 和数据分片是否齐全，方便迁移项目目录。

输入数据检查空清单、重复项、关节数、帧率、形状及非有限数值。训练入口保留模型构建、数据加载、训练和续训，checkpoint 回调位于 `ncs/utils/training_callbacks.py`。

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
