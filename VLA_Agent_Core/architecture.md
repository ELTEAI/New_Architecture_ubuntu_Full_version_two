VLA_Agent_Core/
│
├── 📁 schemas/                 # 阶段一：技能接口定义层
│   └── robot_skill_schema.json # 存放发给大模型的 Function Calling JSON 规范 (定义那 13 个动作)
│
├── 📁 core/                    # 阶段二：大脑中枢代理层
│   ├── __init__.py
│   ├── llm_client.py           # 封装 OpenAI/Qwen 的 API 请求逻辑
│   └── agent_planner.py        # 核心：维护 System Prompt，组装对话，解析 Tool Call
│
├── 📁 execution/               # 阶段三 & 四：硬件对接与安全执行层
│   ├── __init__.py
│   ├── task_queue.py           # 线程安全的动作队列 (Task Queue) 缓冲池
│   └── fsm_guardian.py         # FSM 有限状态机引擎 (处理阻塞、模式切换、紧急熔断)
│
├── ⚙️ config.yaml              # 配置文件 (存放 API_KEY, Base_URL, 默认模型名称等)
├── 📝 prompts.py               # 集中管理各种 System Prompt 模板
└── 🚀 main_agent.py            # Agent 模块的启动总入口 (整合 Whisper + Agent + FSM)