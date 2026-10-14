# 实时预览

在项目根目录安装渲染依赖并启动：

```bash
.venv/bin/python -m pip install -r requirements-render.txt
.venv/bin/python -m view_realtime --config ncs/configs/test.json --gpu-id 0
```

使用 `python -m view_realtime --help` 查看全部参数；无窗口模型校验使用
`python -m view_realtime --validate-only`。

ImGui 面板的 `Motion` 滑块可在播放过程中调节动态布料强度（0–2），
`Reset motion (1.0)` 恢复默认值。初始值由 `--motion` 指定。
修改从下一个模型推理步生效，保留 GRU 历史；暂停时可点击 `Step` 查看效果。
显示端的局部形变插值会带来最多约一个模型步的过渡延迟。

模块文件：

- `__main__.py`：模块启动入口。
- `cli.py`：命令行参数、模型加载、固定步长推理和会话报告。
- `playback.py`：播放时钟、局部形变插值调度和显示骨架。
- `realtime_renderer.py`：OpenGL 渲染、GPU 蒙皮与法线计算。
- `viewer_ui.py`：Dear ImGui 控件和性能统计。
- `test_realtime_renderer.py`：播放、几何、统计和可选集成测试。

运行无需显示窗口或 TensorFlow 的测试：

```bash
.venv/bin/python -m unittest view_realtime.test_realtime_renderer
```

设置 `NCS_TEST_TENSORFLOW=1` 可启用 TensorFlow 测试；设置 `NCS_TEST_OPENGL=1`
可启用需要硬件 OpenGL 窗口的测试。

播放控制、渲染选项及性能测量见
[性能测试文档](../scripts/BENCHMARK.md#moderngl--glfw-实时渲染)。
