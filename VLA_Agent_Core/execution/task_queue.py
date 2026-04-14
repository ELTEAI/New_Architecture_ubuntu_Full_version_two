import queue
import threading

class TaskQueue:
    """
    线程安全的动作缓冲池 (Thread-safe Action Buffer)
    连接上层快速的大模型和下层慢速的物理硬件。
    """
    def __init__(self, max_size=20):
        # 使用 Python 原生的线程安全队列
        self._queue = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        print(f"📥 [Task Queue] 动作缓冲池初始化成功 (容量: {max_size})")

    def push_sequence(self, actions: list):
        """将大脑编译好的动作序列压入队列"""
        with self._lock:
            for task in actions:
                if not self._queue.full():
                    self._queue.put(task)
                else:
                    print("⚠️ [Task Queue] 队列已满！丢弃后续动作以保护内存。")
                    break
        print(f"📦 [Task Queue] 成功压入 {len(actions)} 个动作，当前排队总数: {self._queue.qsize()}")

    def pop_action(self):
        """取出下一个动作 (如果队列为空则阻塞等待)"""
        return self._queue.get()

    def mark_done(self):
        """标记当前动作执行完毕"""
        self._queue.task_done()

    def clear_queue(self):
        """紧急清空队列 (用于被 Whisper 听到'停下'时瞬间打断当前计划)"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        print("🚨 [Task Queue] 警告：队列已被强行清空！")

    def pending_count(self) -> int:
        """返回当前等待执行的动作数量。"""
        return self._queue.qsize()

    def wait_until_all_done(self):
        """阻塞等待队列中已投递动作全部执行完成。"""
        self._queue.join()