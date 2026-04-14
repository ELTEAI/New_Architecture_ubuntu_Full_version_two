import importlib.util
import json
import os
from pathlib import Path

import yaml
from openai import OpenAI

# 本文件位于 VLA_Agent_Core/core/，配置与 schemas 在上一级目录
_CORE_DIR = Path(__file__).resolve().parent
_VLA_AGENT_ROOT = _CORE_DIR.parent
_DEFAULT_CONFIG_PATH = _VLA_AGENT_ROOT / "config.yaml"
_DEFAULT_SCHEMA_PATH = _VLA_AGENT_ROOT / "schemas" / "robot_skill_schema.json"

# 与 schemas/robot_skill_schema.json 中的 function.name 一致
_TACTICAL_TOOL_NAME = "execute_robot_tactical_sequence"

_PROMPTS_MODULE = None


def _load_prompts():
    """从 VLA_Agent_Core/prompts.py 加载提示词模块（不依赖 cwd、支持 python core/agent_planner.py）。"""
    global _PROMPTS_MODULE
    if _PROMPTS_MODULE is not None:
        return _PROMPTS_MODULE
    path = _VLA_AGENT_ROOT / "prompts.py"
    if not path.is_file():
        raise FileNotFoundError(f"❌ 找不到提示词文件：{path}")
    spec = importlib.util.spec_from_file_location("vla_agent_prompts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载提示词模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _PROMPTS_MODULE = mod
    return mod


class VLABrainPlanner:
    """
    Layer 3: 认知中枢大脑 (Agent Planner)
    负责连接本地 vLLM 大模型，将自然语言编译为机器狗底层状态机可执行的动作序列。
    """
    def __init__(self, config_path: str | os.PathLike[str] | None = None):
        # 1. 加载全局配置（默认固定为 VLA_Agent_Core/config.yaml，不依赖 cwd）
        self.config_path = (
            os.fspath(config_path)
            if config_path is not None
            else str(_DEFAULT_CONFIG_PATH)
        )
        self._load_config()
        
        # 2. 初始化 OpenAI 兼容客户端 (指向本地 vLLM)
        self.client = OpenAI(
            api_key=self.llm_config.get("api_key", "EMPTY"),
            base_url=self.llm_config.get("base_url", "http://127.0.0.1:8000/v1")
        )
        self.model_name = self.llm_config.get("model_name", "Qwen3.5-4B")
        self.temperature = self.llm_config.get("temperature", 0.1)
        
        # 3. 动态加载物理技能契约 (Schema)
        self.tools = self._load_skill_schema()

        # 4. 大脑人设：见上级目录 prompts.py
        self.system_prompt = _load_prompts().AGENT_PLANNER_SYSTEM_PROMPT
        print(f"🧠 [Agent Brain] 认知中枢初始化完成! (挂载模型: {self.model_name})")

    def _load_config(self):
        """读取外部 yaml 配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"❌ 找不到配置文件：{self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
            self.llm_config = full_config.get("llm", {})

    def _load_skill_schema(self, schema_path: str | os.PathLike[str] | None = None):
        """读取外部的 JSON 动作说明书，并组装为 OpenAI Tools 格式（默认相对 VLA_Agent_Core 根目录）。"""
        if schema_path is None:
            resolved = _DEFAULT_SCHEMA_PATH
        else:
            p = Path(schema_path)
            resolved = p if p.is_absolute() else (_VLA_AGENT_ROOT / p)

        if not resolved.is_file():
            raise FileNotFoundError(f"❌ 找不到技能说明书：{resolved}")

        with open(resolved, "r", encoding="utf-8") as f:
            skill_dict = json.load(f)
            return [skill_dict]

    def compile_tactical_plan(self, user_text: str):
        """
        核心思考循环：接收文本 -> 请求本地大模型 -> 截获函数调用 -> 返回战术参数
        """
        print(f"\n🤔 [Agent Brain] 正在推演战术: '{user_text}' ...")
        
        try:
            # 发起请求，带上机器狗物理工具
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_text}
                ],
                tools=self.tools,
                # vLLM：只要带 tools / tool_choice，服务端须 --tool-call-parser（见 run_vllm_server.example.sh）。
                # 固定调用战术工具，避免 model 闲聊时不调工具。
                tool_choice={
                    "type": "function",
                    "function": {"name": _TACTICAL_TOOL_NAME},
                },
                temperature=self.temperature,
            )
            
            message = response.choices[0].message
            
            # 判断大模型是否成功调用了物理技能
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == _TACTICAL_TOOL_NAME:
                        # 完美拦截！解析大模型吐出的 JSON 参数
                        args = json.loads(tool_call.function.arguments)
                        sequence_name = args.get("sequence_name", "unnamed_sequence")
                        actions_queue = args.get("actions", [])
                        
                        print(f"✅ [Agent Brain] 编译成功！战术代号: <{sequence_name}>")
                        return sequence_name, actions_queue
            
            # 如果大模型觉得不用动（比如你在和它纯聊天）
            print(f"💬 [Agent Brain] 模型回复纯文本: {message.content}")
            return "chat_only", []

        except Exception as e:
            print(f"❌ [Agent Brain] 大脑思考宕机或连接超时: {e}")
            # 🛡️ 安全降级兜底：遇到任何错误，强行下发静止指令
            return "emergency_stop", [{"action_id": 1, "duration": 0}]

# ==========================================
# 本地快速脑力测试 
# ==========================================
if __name__ == "__main__":
    # 任意 cwd 下可运行：python core/agent_planner.py 或 python -m ...
    brain = VLABrainPlanner()
    
    # 2. 给本地 Qwen3.5-4B 出一个逻辑难题
    test_instruction = "向前走两步后作揖"
    
    # 3. 编译战术
    seq_name, action_plan = brain.compile_tactical_plan(test_instruction)
    
    # 4. 打印结果
    if action_plan:
        print(f"\n📦 解析出的最终物理执行队列 ({seq_name}):")
        for step in action_plan:
            print(f"   -> ID: {step.get('action_id'):02d} | Duration: {step.get('duration')}s")