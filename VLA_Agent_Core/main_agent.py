import os
import threading
import time
from pathlib import Path

from core.agent_planner import VLABrainPlanner
from execution.fsm_guardian import FSMGuardian
from execution.task_queue import TaskQueue


def _resolve_config_path() -> str:
    """固定定位到 VLA_Agent_Core/config.yaml，避免依赖当前工作目录。"""
    return str(Path(__file__).resolve().parent / "config.yaml")


def _is_emergency_text(text: str) -> bool:
    """识别用户的紧急停机意图。"""
    t = text.strip().lower()
    return t in {"停下", "停止", "急停", "stop", "halt", "emergency"}


def _start_speed_monitor(
    fsm_guardian: FSMGuardian,
    queue: TaskQueue,
    pause_event: threading.Event,
    interval_s: float = 0.2,
) -> None:
    """后台实时打印当前动作与速度（vx/vy/omega）。"""
    def _loop() -> None:
        last_line = ""
        while True:
            if pause_event.is_set():
                time.sleep(interval_s)
                continue

            state = fsm_guardian.get_runtime_state()
            pending = queue.pending_count()
            line = (
                "📡 [速度监视] "
                f"动作:{state['action_name']}#{state['action_id']} | "
                f"vx={state['vx']:+.2f}  vy={state['vy']:+.2f}  ω={state['omega']:+.2f} | "
                f"blocking={'Y' if state['is_blocking'] else 'N'} | "
                f"queue={pending}"
            )
            # 仅状态变化时打印，避免刷屏和覆盖用户输入
            if line != last_line:
                print(line, flush=True)
                last_line = line
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, daemon=True, name="speed-monitor")
    t.start()


def main() -> None:
    print("==================================================")
    print("🤖 VLA Main Agent 启动中...")
    print("==================================================")

    # 1) 初始化任务队列与硬件状态机守护线程
    queue = TaskQueue(max_size=20)
    fsm_guardian = FSMGuardian(queue)
    fsm_guardian.start()
    monitor_pause_event = threading.Event()
    _start_speed_monitor(fsm_guardian, queue, monitor_pause_event, interval_s=0.2)

    # 2) 初始化战术规划大脑
    config_path = _resolve_config_path()
    planner = VLABrainPlanner(config_path=config_path)

    print("\n✅ 系统就绪。输入自然语言指令并回车；输入 'exit' 退出。")
    print("   紧急指令: 停下 / 停止 / 急停 / stop")

    # 3) 主循环：接收文本 -> LLM 编译 -> 投递动作队列
    while True:
        try:
            monitor_pause_event.set()   # 输入时暂停监视输出，避免打断 typing
            user_text = input("\n🗣️ 指令> ").strip()
            monitor_pause_event.clear()
        except (KeyboardInterrupt, EOFError):
            monitor_pause_event.clear()
            print("\n👋 收到退出信号，主程序结束。")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("👋 已退出主控循环。")
            break

        # 本地紧急语义优先：不走 LLM，直接清空队列并下发静止动作
        if _is_emergency_text(user_text):
            queue.clear_queue()
            queue.push_sequence([{"action_id": 1, "duration": 0}])
            print("🛑 已触发紧急停止。")
            continue

        seq_name, action_plan = planner.compile_tactical_plan(user_text)

        if not action_plan:
            print(f"ℹ️ 本次无可执行动作（seq={seq_name}）。")
            continue

        # 若模型返回 emergency_stop，先清队列再执行静止
        if seq_name == "emergency_stop":
            queue.clear_queue()

        queue.push_sequence(action_plan)
        print(f"🚚 已投递战术队列: {seq_name} | 动作数: {len(action_plan)}")
        print("⏳ 等待当前战术动作全部执行完成...")
        queue.wait_until_all_done()
        print("✅ 当前战术执行完成，可输入下一条指令。")


if __name__ == "__main__":
    # 避免某些环境里缓冲日志不及时
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
