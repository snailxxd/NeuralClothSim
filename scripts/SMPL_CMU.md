# 使用 AMASS CMU 训练 SMPL 衣物模型

这里训练的是固定 SMPL 人体上的衣物网络。默认使用仓库自带的
`body_models/smpl_female_neutral/` 和 `tshirt.obj`，不需要另装 SMPL+H
人体模型，也不需要衣物仿真真值。动作的 `betas`、`gender` 和 `dmpls`
不改变训练人体。

## 数据准备

Windows 数据位置：`D:\Work\amass\CMU`；当前 WSL 中对应
`/mnt/d/Work/amass/CMU`。输入必须已解压，目录形如
`CMU/01/01_01_poses.npz`。

在项目根目录运行（只需 CPU、NumPy 和 SciPy）：

```bash
.venv/bin/python scripts/prepare_amass.py --source /mnt/d/Work/amass/CMU
```

也可在服务器上转换，将 `--source` 换成上传后的 CMU 目录。Windows
Python 可使用 `python scripts/prepare_amass.py --source "D:\Work\amass\CMU"`。

脚本将数据写入 `data/amass_cmu/`，保留受试者子目录；清单写入
`ncs/dataset/txt/amass_cmu/`。使用固定随机种子 42，按受试者约
80%/10%/10% 划分训练、验证和测试，同一个受试者不会跨集合。
目标目录必须为空；失败后应移走或删除本次不完整输出，再重跑。

转换保留根关节和 21 个身体关节，SMPL 末端两个 hand 关节设为零；
SMPL+H 的手指参数不能直接作为 SMPL 的末端手关节。输入必须为
`poses: (T,156)`、`trans: (T,3)` 和 `mocap_framerate`。
输出为 `poses: (T,24,3)`、`trans: (T,3)`、`fps`、`source`。

默认按原始 AMASS Z-up 转换到本项目 Y-up：根旋转左乘绕 X 轴 -90°
旋转，平移从 `(x,y,z)` 变为 `(x,z,-y)`，身体局部旋转不变。
平移减去首帧位置，保留相对运动。对于自行处理过、已是 Y-up 的输入，
使用 `--source-up-axis y`。
高帧率动作使用旋转 SLERP 和平移线性插值降至 30 FPS；低于 30 FPS
的动作保留原帧率，训练读取器会按配置的 30 FPS 采样。
不会仅修改 FPS 标签来改变动作速度。

## 搬到训练服务器

推荐从当前开发分支导出训练包，服务器无需另开 Git 分支。上传包只包含
训练依赖模块、选定配置、人体/衣物资源、数据清单和依赖文件；排除查看器、
Blender、预测入口、开发脚本、虚拟环境、日志和已有 checkpoint。
代码包使用当前工作区文件，因此包含尚未提交的训练代码修改。

全量转换结束后，在当前 WSL 的项目根目录运行：

```bash
python scripts/package_training.py --config ncs/configs/smpl_tshirt.json \
  --include-data --output /tmp/ncs-training-smpl_tshirt.tar.gz
scp /tmp/ncs-training-smpl_tshirt.tar.gz USER@SERVER:/SERVER/PROJECT/
```

替换 SSH 地址和服务器路径。在服务器空项目目录解包：

```bash
tar -xzf ncs-training-smpl_tshirt.tar.gz
```

以后仅更新代码时省略 `--include-data`，使用新的包文件名；数据无需重复
上传。续训 checkpoint 需另外传输。转换后的 `data/amass_cmu/` 已加入
Git 忽略规则。服务器按包内 README 新建 Linux 环境；完整开发环境及
GPU 检查说明见本地 [ENVIRONMENT.md](../ENVIRONMENT.md)。训练包可用：

```bash
conda create --prefix ./.venv python=3.11 pip -y
conda activate "$PWD/.venv"
python -m pip install -r requirements-gpu.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 启动训练

```bash
cd ncs
python train.py --config configs/smpl_tshirt.json --gpu_id 0
```
在命令末尾添加原有 `--verbose` 即可显示 batch 进度，batch 进度条仅显示总 Loss，
每轮结束后分行显示验证集分项损失。每轮完整指标写入
`logs/smpl_tshirt/metrics.csv` 和 TensorBoard，`m/Loss` 为包含惯性项的总损失。
NaN/Inf 损失或梯度会终止训练；保存前还会检查模型和 Adam 状态。


默认为 batch size 16、30 FPS、0.5 秒窗口、5000 epochs，从头训练。
完整 CMU 每轮比单动作示例长得多，可先复制配置为
`smpl_tshirt_smoke.json`，设置 `epochs: 1` 做一次完整训练/验证检查，
之后用正式配置启动独立实验。正式启动前先查看一个转换后的动作，
确认人体竖直、运动速度及衣物初始位置合理。

96 GB H20 可直接使用上述命令：默认按需分配 GPU 显存，无固定 6 GiB 上限，
OOM 直接报错退出，无自动重启机制。共享 GPU 时可添加
`--gpu-memory-limit-mb 81920` 设置每卡 80 GiB 上限。
先保留 batch size 16，实测吞吐和显存后再调整。数据读取器会将训练和验证
动作载入 CPU 内存，应为全量 CMU 留出足够系统内存。

输出为 `checkpoints/smpl_tshirt/` 和 `logs/smpl_tshirt/`，其中
`best.weights.h5` 保存验证损失最低的权重，`recovery/` 保存最新恢复状态。
已有同名输出时，训练会拒绝覆盖；恢复已有实验时将配置的
`experiment.checkpoint` 设为 `"smpl_tshirt"`，保留恢复目录即可自动读取
已完成的轮数。详见 [ENVIRONMENT.md](../ENVIRONMENT.md)。

训练后预测：

```bash
python predict.py --config configs/smpl_tshirt.json --gpu_id 0
```

默认预测整个测试集。首次查看可将配置的 `data.test` 临时指向仅包含
一段测试动作的新清单。`predict.py` 按配置文件名寻找训练权重，
应保留配置文件名 `smpl_tshirt.json`。
