import time
import threading


class FSMGuardian(threading.Thread):
    """
    Layer 4: 硬件状态机守护者。

    负责从 TaskQueue 取出动作，先进行 FSM 状态转移安全验证，
    再进行模式分类与底层 SDK / dry-run 映射。

    This class is still hardware-independent in this project version:
    it simulates velocity / blocking states and records runtime state.
    """

    ACTION_NAMES = {
        0: "前进", 1: "完全静止", 2: "退后", 3: "左横移", 4: "右横移",
        5: "左转", 6: "右转", 7: "坐下", 8: "起立", 9: "伸懒腰",
        10: "打滚", 11: "摆姿势", 12: "拜年/作揖"
    }

    MODE_0_EMERGENCY = [1]
    MODE_1_CONTINUOUS = [0, 2, 3, 4, 5, 6]
    MODE_2_BLOCKING = [7, 8, 9, 10, 11, 12]

    MODE_1_SPEED_MAP = {
        0: {"vx": 0.2, "vy": 0.0, "omega": 0.0},
        2: {"vx": -0.3, "vy": 0.0, "omega": 0.0},
        3: {"vx": 0.0, "vy": 0.3, "omega": 0.0},
        4: {"vx": 0.0, "vy": -0.3, "omega": 0.0},
        5: {"vx": 0.0, "vy": 0.0, "omega": 0.5},
        6: {"vx": 0.0, "vy": 0.0, "omega": -0.5},
    }

    FSM_STATES = {
        "IDLE",
        "STANDING",
        "SITTING",
        "LOCOMOTION",
        "BLOCKING",
        "EMERGENCY_STOP",
    }

    def __init__(self, task_queue):
        super().__init__(daemon=True)
        self.queue = task_queue
        self.is_blocking = False

        self._state_lock = threading.Lock()
        self._fsm_state = "IDLE"
        self._rejection_log = []

        self._runtime_state = {
            "action_id": 1,
            "action_name": self.ACTION_NAMES[1],
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "is_blocking": False,
            "fsm_state": self._fsm_state,
            "updated_at": time.time(),
        }

        print("🛡️  [FSM Guardian] 底层运动控制线程已就绪。")

    def run(self):
        """线程主循环：后台等待并执行动作。"""
        print("⚙️  [FSM Guardian] 状态机开始监听队列...")
        while True:
            task = self.queue.pop_action()

            action_id = task.get("action_id", 1)
            duration = task.get("duration", 0)

            result = self._execute_single_action(action_id, duration)

            if result.get("accepted", False):
                self._log_speed_snapshot_after_action()
            else:
                print(f"   ⛔ [FSM拒绝] {result.get('reason')}")

            self.queue.mark_done()

    def validate_transition(self, action_id):
        """
        Validate whether the requested action is safe under the current FSM state.

        Returns:
            (accepted: bool, reason: str)
        """
        try:
            action_id = int(action_id)
        except (TypeError, ValueError):
            return False, "action_id is not an integer"

        with self._state_lock:
            state = self._fsm_state
            is_blocking = self.is_blocking

        if action_id not in self.ACTION_NAMES:
            return False, f"unknown action_id={action_id}"

        # Emergency stop is always allowed.
        if action_id == 1:
            return True, "emergency/full stop is always allowed"

        # During blocking actions, only stop is allowed.
        if is_blocking or state == "BLOCKING":
            return False, "only stop is allowed during blocking action"

        # After emergency stop, require stand_up before normal actions.
        if state == "EMERGENCY_STOP":
            if action_id == 8:
                return True, "stand_up allowed after emergency stop"
            return False, "must stand_up before executing other actions after emergency stop"

        # Stand up is allowed from IDLE, SITTING, STANDING.
        if action_id == 8:
            if state in {"IDLE", "SITTING", "STANDING"}:
                return True, "stand_up allowed"
            return False, f"stand_up not allowed from state={state}"

        # Sit down is allowed from IDLE or STANDING.
        if action_id == 7:
            if state in {"IDLE", "STANDING"}:
                return True, "sit_down allowed from idle/standing"
            return False, f"sit_down not allowed from state={state}"

        # Continuous locomotion requires STANDING or LOCOMOTION.
        if action_id in self.MODE_1_CONTINUOUS:
            if state in {"STANDING", "LOCOMOTION"}:
                return True, "continuous motion allowed from standing/locomotion"
            return False, f"continuous motion requires standing, current state={state}"

        # Blocking skills except sit/stand require STANDING.
        if action_id in {9, 10, 11, 12}:
            if state == "STANDING":
                return True, "blocking skill allowed from standing"
            return False, f"blocking skill requires standing, current state={state}"

        return False, f"unhandled transition action_id={action_id}, state={state}"

    def _apply_state_after_accept(self, action_id):
        """Update FSM state after an accepted action."""
        with self._state_lock:
            if action_id == 1:
                self._fsm_state = "EMERGENCY_STOP"
            elif action_id == 7:
                self._fsm_state = "SITTING"
            elif action_id == 8:
                self._fsm_state = "STANDING"
            elif action_id in self.MODE_1_CONTINUOUS:
                self._fsm_state = "LOCOMOTION"
            elif action_id in {9, 10, 11, 12}:
                self._fsm_state = "BLOCKING"
            else:
                self._fsm_state = "IDLE"

            self._runtime_state["fsm_state"] = self._fsm_state
            self._runtime_state["updated_at"] = time.time()

    def _record_rejection(self, action_id, duration, reason):
        with self._state_lock:
            entry = {
                "action_id": action_id,
                "duration": duration,
                "reason": reason,
                "fsm_state": self._fsm_state,
                "updated_at": time.time(),
            }
            self._rejection_log.append(entry)
        return entry

    def get_rejection_log(self):
        with self._state_lock:
            return list(self._rejection_log)

    def get_fsm_state(self):
        with self._state_lock:
            return self._fsm_state

    def set_fsm_state_for_test(self, state):
        """Only for testing. Allows deterministic FSM transition tests."""
        if state not in self.FSM_STATES:
            raise ValueError(f"Unknown FSM state: {state}")
        with self._state_lock:
            self._fsm_state = state
            self.is_blocking = state == "BLOCKING"
            self._runtime_state["fsm_state"] = state
            self._runtime_state["is_blocking"] = self.is_blocking
            self._runtime_state["updated_at"] = time.time()

    def _execute_single_action(self, action_id, duration):
        """解析、验证并执行单个动作。"""
        try:
            action_id = int(action_id)
        except (TypeError, ValueError):
            action_id = -1

        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 0.0

        accepted, reason = self.validate_transition(action_id)

        if not accepted:
            rejection = self._record_rejection(action_id, duration, reason)
            return {
                "accepted": False,
                "executed_action_id": None,
                "reason": reason,
                "rejection": rejection,
            }

        action_name = self.ACTION_NAMES.get(action_id, "完全静止")

        # === 底盘分发逻辑 ===
        if action_id in self.MODE_0_EMERGENCY:
            self._set_motion_state(
                action_id,
                action_name,
                vx=0.0,
                vy=0.0,
                omega=0.0,
                is_blocking=False,
            )
            self._apply_state_after_accept(action_id)
            print(f"   🛑 [执行] 紧急制动 ({action_name}) -> 速度全量归零。")

        elif action_id in self.MODE_1_CONTINUOUS:
            speed = self.MODE_1_SPEED_MAP.get(
                action_id,
                {"vx": 0.0, "vy": 0.0, "omega": 0.0},
            )
            print(f"   🔄 [执行] 连续运动 ({action_name}) -> 改变底盘速度矢量。")
            self._apply_state_after_accept(action_id)
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
            safe_duration = max(1.0, min(float(duration), 10.0))
            self.is_blocking = True
            self._apply_state_after_accept(action_id)
            self._set_motion_state(
                action_id,
                action_name,
                vx=0.0,
                vy=0.0,
                omega=0.0,
                is_blocking=True,
            )

            print(f"   🔒 [执行] 高层动作 ({action_name}) -> 劫持底盘，物理阻塞 {safe_duration} 秒...")
            time.sleep(safe_duration)

            self.is_blocking = False

            # Blocking action finished.
            # Sit down leaves SITTING; stand up leaves STANDING; other blocking skills return to STANDING.
            with self._state_lock:
                if action_id == 7:
                    self._fsm_state = "SITTING"
                else:
                    self._fsm_state = "STANDING"
                self._runtime_state["fsm_state"] = self._fsm_state

            self._set_motion_state(
                action_id,
                action_name,
                vx=0.0,
                vy=0.0,
                omega=0.0,
                is_blocking=False,
            )
            print(f"   🔓 [完成] {action_name} 动作物理执行完毕。")

        return {
            "accepted": True,
            "executed_action_id": action_id,
            "reason": reason,
            "fsm_state": self.get_fsm_state(),
        }

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
                    "fsm_state": self._fsm_state,
                    "updated_at": time.time(),
                }
            )

    def get_runtime_state(self):
        """返回可安全读取的当前动作、速度与 FSM 状态。"""
        with self._state_lock:
            return dict(self._runtime_state)

    def _log_speed_snapshot_after_action(self):
        state = self.get_runtime_state()
        print(
            "   📡 [动作后速度] "
            f"动作:{state['action_name']}#{state['action_id']} | "
            f"vx={state['vx']:+.2f}  vy={state['vy']:+.2f}  ω={state['omega']:+.2f} | "
            f"blocking={'Y' if state['is_blocking'] else 'N'} | "
            f"fsm={state['fsm_state']}"
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