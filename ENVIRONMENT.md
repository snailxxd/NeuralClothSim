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

中断后继续训练时，可复制配置为新的文件名，在 `experiment` 中设置 `checkpoint` 为旧实验名、`initial_epoch` 为 checkpoint 已完成的轮数。训练会恢复模型与 Adam 优化器状态；例如 `initial_epoch: 207`、`epochs: 5000` 会从第 208 轮继续到第 5000 轮，新结果保存在新配置文件名对应的目录中。日志可能先于 checkpoint 写入，应以实际保存状态为准。

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

逐帧推理可直接用 ModernGL + GLFW 查看器显示，无需 Blender 或 PC2 文件：

```bash
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python scripts/view_realtime.py --xla
```

实时播放、键鼠操作及性能测试方法见 [scripts/BENCHMARK.md](scripts/BENCHMARK.md#moderngl--glfw-实时渲染)。
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
