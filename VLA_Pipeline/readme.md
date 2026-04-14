pip install -r /home/ubuntu/New_Architecture/VLA_Pipeline/requirements.txt
pip install -r /home/ubuntu/New_Architecture/VLA_Pipeline/requirements-vllm.txt


VLA_Pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── pipeline.yaml
├── scripts/
│   ├── run_vllm_server.sh
│   └── download_qwen35_4b.py
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── contracts.py
│   │   ├── event_bus.py
│   │   └── health.py
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── mediapipe_stream.py
│   │   ├── gesture_classifier.py
│   │   └── reflex_bridge.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── whisper_input.py
│   ├── cognition/
│   │   ├── __init__.py
│   │   ├── planner_client.py
│   │   └── prompt_router.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── task_queue_adapter.py
│   │   └── fsm_adapter.py
│   └── runtime/
│       ├── __init__.py
│       ├── logger.py
│       └── metrics.py
└── tests/
    ├── test_contracts.py
    ├── test_reflex_bridge.py
    └── test_orchestrator_smoke.py
scripts/：模型下载与 vLLM 启动脚本
config/：管道运行参数（模式、自动拉起 vLLM 等）
src/pipeline/：总编排与通用契约
src/perception/：视觉感知与反射桥
src/audio/：Whisper 音频输入适配
src/cognition/：LLM 规划与文本路由
src/execution/：复用 VLA_Agent_Core 的执行层适配
src/runtime/：日志与指标