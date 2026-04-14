# VLA_Pipeline (Minimal Closed Loop)

这是一个最小可运行管道，把以下链路串起来：

- 文本输入（可替代 Whisper 输出）
- LLM 规划（调用 `VLA_Agent_Core/core/agent_planner.py`）
- 反射桥（手势分类结果转动作，含防抖/冷却）
- 动作缓冲池（`TaskQueue`）
- 状态机执行（`FSMGuardian`）

## 目录

见 `src/` 下各模块，结构与主工程约定一致。

## 快速开始

1. 安装依赖

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
pip install -r requirements.txt
```

2. 运行最小闭环（默认会自动拉起 vLLM）

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python -m src.pipeline.orchestrator
```

> 自动启动配置见 `config/pipeline.yaml -> vllm.autostart`。若你已手动启动 vLLM，管道会先探测 `/v1/models`，可用则跳过自启。

3. 运行测试

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
pytest -q
```

## 运行模式

- `planner_only`：仅语义规划
- `reflex_only`：仅视觉反射桥
- `hybrid`：二者都启用（默认）

配置文件见 `config/pipeline.yaml`。
