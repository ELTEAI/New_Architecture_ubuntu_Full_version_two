1. push 一组普通任务
2. 检查 pending_count 是否等于普通任务数
3. clear_queue()
4. 检查 pending_count 是否为 0
5. push emergency stop: {"action_id": 1, "duration": 0}
6. pop stop task
7. 调用 FSMGuardian._execute_single_action(1, 0)
8. 检查是否 accepted，并且 FSM state 是否为 EMERGENCY_STOP