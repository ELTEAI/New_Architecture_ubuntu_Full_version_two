import time
import threading

class FSMGuardian(threading.Thread):
    """
    Layer 4: 硬件状态机守护者 (以独立后台线程运行)
    负责从 TaskQueue 取出动作，进行模式分类、安全阻塞验证，并映射到底层 SDK。
    """
    
    ACTION_NAMES = {
        0: "前进", 1: "完全静止", 2: "退后", 3: "左横移", 4: "右横移",
        5: "左转", 6: "右转", 7: "坐下", 8: "起立", 9: "伸懒腰",
        10: "打滚", 11: "摆姿势", 12: "拜年/作揖"
    }

    MODE_0_EMERGENCY = [1]
    MODE_1_CONTINUOUS = [0, 2, 3, 4, 5, 6]
    MODE_2_BLOCKING = [7, 8, 9, 10, 11, 12]
    # 与战术工具文档中的速度约定保持一致（单位示例：m/s, rad/s）
    MODE_1_SPEED_MAP = {
        0: {"vx": 0.2, "vy": 0.0, "omega": 0.0},   # 招手(约定为前进示意)
        2: {"vx": -0.3, "vy": 0.0, "omega": 0.0},  # 退后
        3: {"vx": 0.0, "vy": 0.3, "omega": 0.0},   # 左横移
        4: {"vx": 0.0, "vy": -0.3, "omega": 0.0},  # 右横移
        5: {"vx": 0.0, "vy": 0.0, "omega": 0.5},   # 左转
        6: {"vx": 0.0, "vy": 0.0, "omega": -0.5},  # 右转
    }

    def __init__(self, task_queue):
        # 初始化线程，设为 Daemon（主程序退出时它会自动安全阵亡）
        super().__init__(daemon=True)
        self.queue = task_queue
        self.is_blocking = False
        self._state_lock = threading.Lock()
        self._runtime_state = {
            "action_id": 1,
            "action_name": self.ACTION_NAMES[1],
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "is_blocking": False,
            "updated_at": time.time(),
        }
        print("🛡️  [FSM Guardian] 底层运动控制线程已就绪。")

    def run(self):
        """线程主循环：永远在后台等待并执行动作"""
        print("⚙️  [FSM Guardian] 状态机开始监听队列...")
        while True:
            # 1. 这里会一直阻塞，直到队列里有新动作进来
            task = self.queue.pop_action()
            
            action_id = task.get("action_id", 1)
            duration = task.get("duration", 0)
            
            # 2. 执行核心逻辑
            self._execute_single_action(action_id, duration)
            self._log_speed_snapshot_after_action()
            
            # 3. 告诉队列这个动作做完了
            self.queue.mark_done()

    def _execute_single_action(self, action_id, duration):
        """解析并执行单个动作"""
        action_name = self.ACTION_NAMES.get(action_id, "完全静止")
        
        # 安全拦截：未知 ID 强行设为静止
        if action_id not in self.ACTION_NAMES:
            action_id = 1
            duration = 0

        # === 底盘分发逻辑 (对接 Unitree/宇树 SDK) ===
        if action_id in self.MODE_0_EMERGENCY:
            self._set_motion_state(action_id, action_name, vx=0.0, vy=0.0, omega=0.0, is_blocking=False)
            print(f"   🛑 [执行] 紧急制动 ({action_name}) -> 速度全量归零。")
            # SDK: set_velocity(0, 0, 0)
            
        elif action_id in self.MODE_1_CONTINUOUS:
            speed = self.MODE_1_SPEED_MAP.get(action_id, {"vx": 0.0, "vy": 0.0, "omega": 0.0})
            print(f"   🔄 [执行] 连续运动 ({action_name}) -> 改变底盘速度矢量。")
            # SDK: set_velocity(...) 
            self._ramp_speed_during_action(
                action_id=action_id,
                action_name=action_name,
                target_vx=speed["vx"],
                target_vy=speed["vy"],
                target_omega=speed["omega"],
                steps=6,
                step_sleep=0.08,
            )
            
        elif action_id in self.MODE_2_BLOCKING:
            # 强行纠正大模型乱写的超长阻塞时间 (最高允许10秒)
            safe_duration = max(1.0, min(float(duration), 10.0)) 
            self.is_blocking = True
            self._set_motion_state(action_id, action_name, vx=0.0, vy=0.0, omega=0.0, is_blocking=True)
            
            print(f"   🔒 [执行] 高层动作 ({action_name}) -> 劫持底盘，物理阻塞 {safe_duration} 秒...")
            # SDK: call_high_level_api(action_id)
            
            time.sleep(safe_duration) # 模拟硬件真实的执行时间
            self.is_blocking = False
            self._set_motion_state(action_id, action_name, vx=0.0, vy=0.0, omega=0.0, is_blocking=False)
            print(f"   🔓 [完成] {action_name} 动作物理执行完毕。")

    def _set_motion_state(self, action_id, action_name, vx, vy, omega, is_blocking):
        with self._state_lock:
            self._runtime_state.update(
                {
                    "action_id": action_id,
                    "action_name": action_name,
                    "vx": float(vx),
                    "vy": float(vy),
                    "omega": float(omega),
                    "is_blocking": bool(is_blocking),
                    "updated_at": time.time(),
                }
            )

    def get_runtime_state(self):
        """返回可安全读取的当前动作与速度状态。"""
        with self._state_lock:
            return dict(self._runtime_state)

    def _log_speed_snapshot_after_action(self):
        """每个动作执行完成后输出一次速度快照。"""
        state = self.get_runtime_state()
        print(
            "   📡 [动作后速度] "
            f"动作:{state['action_name']}#{state['action_id']} | "
            f"vx={state['vx']:+.2f}  vy={state['vy']:+.2f}  ω={state['omega']:+.2f} | "
            f"blocking={'Y' if state['is_blocking'] else 'N'}"
        )

    def _ramp_speed_during_action(
        self,
        action_id,
        action_name,
        target_vx,
        target_vy,
        target_omega,
        steps=6,
        step_sleep=0.08,
    ):
        """连续动作执行时模拟速度渐变，并逐步打印速度变化。"""
        current = self.get_runtime_state()
        start_vx = current["vx"]
        start_vy = current["vy"]
        start_omega = current["omega"]

        for i in range(1, steps + 1):
            ratio = i / steps
            vx = start_vx + (target_vx - start_vx) * ratio
            vy = start_vy + (target_vy - start_vy) * ratio
            omega = start_omega + (target_omega - start_omega) * ratio
            self._set_motion_state(
                action_id,
                action_name,
                vx=vx,
                vy=vy,
                omega=omega,
                is_blocking=False,
            )
            print(
                "   📈 [执行中速度] "
                f"{action_name} step {i}/{steps} | "
                f"vx={vx:+.2f}  vy={vy:+.2f}  ω={omega:+.2f}"
            )
            time.sleep(step_sleep)