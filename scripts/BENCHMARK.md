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
