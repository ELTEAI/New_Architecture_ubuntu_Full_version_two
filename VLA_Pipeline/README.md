# VLA_Pipeline

最小可运行闭环管道，集成以下链路：

- 文本输入 -> LLM 规划（复用 `VLA_Agent_Core/core/agent_planner.py`）
- 视觉感知（Go2 摄像头优先，失败回退本机摄像头）
- MediaPipe Tasks（`pose_landmarker_lite.task` + `hand_landmarker.task`）
- 1D-CNN+GRU 手势分类 -> ReflexBridge -> 动作队列
- `TaskQueue` + `FSMGuardian` 执行
- vLLM OpenAI 兼容服务自动检测/拉起

## 重要说明：双环境

由于依赖冲突，建议使用两个 Python 环境：

- **管道环境**：安装 `requirements.txt`（含 `mediapipe`，要求 `protobuf<4`）
- **vLLM环境**：安装 `requirements-vllm.txt`（`vllm` 要求 `protobuf>=5`）

不要把两份依赖装在同一个环境里。

## 快速开始

### 1) 安装管道环境依赖

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m pip install -r requirements.txt
```

### 2) 准备 vLLM 独立环境（示例）

```bash
# 示例：你自己的 vllm 环境路径
/path/to/vllm-env/bin/python -m pip install -r /home/ubuntu/New_Architecture/VLA_Pipeline/requirements-vllm.txt
```

### 3) 启动管道

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
export VLLM_PYTHON=/path/to/vllm-env/bin/python
python3 -m src.pipeline.orchestrator
```

## 启动时自动做的事情

`orchestrator` 启动阶段会先做资源闸门检查，缺失则自动下载，全部就绪后才继续：

- `models/best_mp_gesture_model.pth`
- `models/Qwen3.5-4B/`（通过 `scripts/download_qwen35_4b.py`）
- `models/MediaPipe_Models/pose_landmarker_lite.task`
- `models/MediaPipe_Models/hand_landmarker.task`

资源下载与就绪状态会在终端打印详细日志。

## 配置文件

主配置：`config/pipeline.yaml`

常用项：

- `pipeline.mode`：`planner_only` / `reflex_only` / `hybrid`
- `pipeline.use_text_cli`：是否启用终端文本输入
- `vllm.autostart`：是否自动拉起 vLLM
- `perception.enabled`：是否启用视觉感知
- `perception.pose_model_path` / `hand_model_path`：`.task` 路径
- `resources.*`：资源自动拉取设置

## 常见问题

- **看到 `MessageFactory` / `GetPrototype` 报错**  
  说明 `mediapipe` 与 `protobuf` 版本冲突。请确认管道环境使用 `requirements.txt`（`protobuf<4`），并与 vLLM 环境分离。

- **`camera[0]` 打不开**  
  说明 Go2 不可用且本机摄像头也不可用。检查 `perception.local_camera_index`、`/dev/video*`、权限或直接连接 Go2。

## 测试

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m pytest -q
```
