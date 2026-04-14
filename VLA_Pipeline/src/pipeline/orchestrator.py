#cd /home/ubuntu/New_Architecture/VLA_Pipeline
#python3 -m src.pipeline.orchestrator
#
from __future__ import annotations


from pathlib import Path
import subprocess
import threading
import time
import queue
import shutil
from urllib.error import URLError
from urllib.request import urlopen
import yaml
import numpy as np
import cv2

from src.cognition.planner_client import PlannerClient
from src.cognition.prompt_router import route_text
from src.execution.fsm_adapter import FSMAdapter
from src.execution.task_queue_adapter import TaskQueueAdapter
from src.perception.gesture_classifier import GestureClassifier
from src.perception.mediapipe_stream import CameraConfig, MediaPipeStream
from src.perception.reflex_bridge import ReflexBridge
from src.pipeline.contracts import ActionTask, ErrorEvent, PerceptionEvent, SpeechEvent, SystemEvent
from src.pipeline.event_bus import EventBus
from src.pipeline.health import Heartbeat
from src.runtime.logger import log_info, log_warn
from src.runtime.metrics import Metrics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "pipeline.yaml"
DEFAULT_VLLM_SCRIPT = ROOT / "scripts" / "run_vllm_server.sh"
DEFAULT_QWEN_DOWNLOAD_SCRIPT = ROOT / "scripts" / "download_qwen35_4b.py"


def load_cfg(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_vllm_ready(base_url: str, timeout_sec: float = 1.0) -> bool:
    """探测 vLLM OpenAI 兼容服务是否可用。"""
    url = base_url.rstrip("/") + "/models"
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except URLError:
        return False
    except Exception:
        return False


class VLLMProcessManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None

    def maybe_start(self) -> None:
        vcfg = self.cfg.get("vllm", {})
        base_url = self.cfg["planner"]["base_url_for_healthcheck"]
        if _is_vllm_ready(base_url):
            log_info(f"vLLM already ready @ {base_url}")
            return

        if not bool(vcfg.get("autostart", False)):
            self._wait_ready_or_raise(base_url, float(vcfg.get("startup_timeout_sec", 60)))
            return

        script = Path(vcfg.get("script_path", str(DEFAULT_VLLM_SCRIPT)))
        if not script.is_file():
            raise FileNotFoundError(f"vLLM 启动脚本不存在: {script}")

        log_info(f"autostart vLLM: {script}")
        self.proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(script.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._wait_ready_or_raise(base_url, float(vcfg.get("startup_timeout_sec", 60)))
        log_info("vLLM autostart ready")

    @staticmethod
    def _wait_ready_or_raise(base_url: str, max_wait: float) -> None:
        log_info(f"waiting vLLM ready @ {base_url} (timeout={max_wait:.0f}s)")
        t0 = time.time()
        while time.time() - t0 < max_wait:
            if _is_vllm_ready(base_url):
                return
            time.sleep(1.0)
        raise TimeoutError(f"vLLM 未就绪，超时: {max_wait}s (url={base_url})")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            log_info("stopping autostarted vLLM process...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def _ensure_resources(cfg: dict) -> None:
    rcfg = cfg.get("resources", {})
    if not bool(rcfg.get("auto_pull", True)):
        log_info("resource auto pull disabled by config")
        return

    model_dir = Path(rcfg.get("model_dir", str(ROOT / "models")))
    model_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"[resources] 开始资源检查: model_dir={model_dir}")

    def _download_file(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        log_info(f"[resources] 正在下载: {target.name}")
        log_info(f"[resources] 下载地址: {url}")
        t0 = time.time()
        with urlopen(url, timeout=30) as resp, open(target, "wb") as f:
            shutil.copyfileobj(resp, f)
        elapsed = time.time() - t0
        size_mb = target.stat().st_size / (1024 * 1024)
        log_info(f"[resources] 下载完成: {target} ({size_mb:.2f} MB, {elapsed:.1f}s)")

    # 1) 手势分类权重
    gesture_file = model_dir / "best_mp_gesture_model.pth"
    if not gesture_file.is_file():
        log_warn(f"[resources] 缺少手势权重，准备下载: {gesture_file}")
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise RuntimeError(
                "缺少 huggingface_hub，无法自动下载手势权重。请先安装依赖。"
            ) from e

        log_info("[resources] 正在下载手势权重（Hugging Face）...")
        hf_hub_download(
            repo_id=rcfg.get("gesture_repo_id", "Xuhui0101/best_mp_gesture_model"),
            filename=rcfg.get("gesture_filename", "best_mp_gesture_model.pth"),
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=rcfg.get("hf_token"),
        )
        if not gesture_file.is_file():
            raise FileNotFoundError(f"手势权重下载后仍不存在: {gesture_file}")
        size_mb = gesture_file.stat().st_size / (1024 * 1024)
        log_info(f"[resources] 手势权重下载完成: {gesture_file} ({size_mb:.2f} MB)")
    else:
        size_mb = gesture_file.stat().st_size / (1024 * 1024)
        log_info(f"[resources] 手势权重已就绪: {gesture_file} ({size_mb:.2f} MB)")

    # 2) Qwen3.5-4B 模型目录
    qwen_dir = model_dir / "Qwen3.5-4B"
    qwen_check = qwen_dir / "config.json"
    if not qwen_check.is_file():
        download_script = Path(rcfg.get("qwen_download_script", str(DEFAULT_QWEN_DOWNLOAD_SCRIPT)))
        if not download_script.is_file():
            raise FileNotFoundError(f"Qwen 下载脚本不存在: {download_script}")
        log_warn(f"[resources] 缺少 Qwen3.5-4B，准备下载: {qwen_dir}")
        log_info(f"[resources] 执行下载脚本: {download_script}")
        subprocess.run(["python3", str(download_script)], cwd=str(download_script.parent), check=True)
        if not qwen_check.is_file():
            raise FileNotFoundError(f"Qwen 模型下载后仍不完整（缺少 config.json）: {qwen_dir}")
        log_info(f"[resources] Qwen3.5-4B 下载完成: {qwen_dir}")
    else:
        log_info(f"[resources] Qwen3.5-4B 已就绪: {qwen_dir}")

    # 3) MediaPipe .task 模型（Pose + Hands）
    mp_dir = model_dir / "MediaPipe_Models"
    pose_target = mp_dir / "pose_landmarker_lite.task"
    hand_target = mp_dir / "hand_landmarker.task"
    pose_url = rcfg.get(
        "pose_task_url",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    )
    hand_url = rcfg.get(
        "hand_task_url",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    )

    if not pose_target.is_file():
        log_warn(f"[resources] 缺少 pose task，准备下载: {pose_target}")
        _download_file(str(pose_url), pose_target)
    if not hand_target.is_file():
        log_warn(f"[resources] 缺少 hand task，准备下载: {hand_target}")
        _download_file(str(hand_url), hand_target)

    if not pose_target.is_file() or not hand_target.is_file():
        raise FileNotFoundError("MediaPipe .task 模型下载失败或不完整。")
    pose_mb = pose_target.stat().st_size / (1024 * 1024)
    hand_mb = hand_target.stat().st_size / (1024 * 1024)
    log_info(f"[resources] MediaPipe tasks 已就绪: {mp_dir} (pose={pose_mb:.2f} MB, hand={hand_mb:.2f} MB)")

    # 全量资源最终闸门：未完备则抛错，主流程不得继续。
    required_paths = [gesture_file, qwen_check, pose_target, hand_target]
    missing = [str(p) for p in required_paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"资源未完备，禁止继续启动流程。缺失项: {missing}")
    log_info("[resources] 全部资源完备，继续启动后续流程。")


class HolisticKeypointExtractor:
    """
    使用 MediaPipe Tasks 提取 (75,3) 关键点：
    pose(33) + left_hand(21) + right_hand(21)。
    """

    def __init__(
        self,
        pose_model_path: str,
        hand_model_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_hands: int = 2,
        use_gpu_delegate: bool = True,
    ):
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except Exception as e:
            raise RuntimeError("未安装 mediapipe，无法启用视觉感知链路。") from e

        self._mp = mp
        self._vision = vision
        self._ts_ms = 0

        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if use_gpu_delegate
            else mp_python.BaseOptions.Delegate.CPU
        )

        pose_base = mp_python.BaseOptions(model_asset_path=pose_model_path, delegate=delegate)
        hand_base = mp_python.BaseOptions(model_asset_path=hand_model_path, delegate=delegate)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        hand_opts = vision.HandLandmarkerOptions(
            base_options=hand_base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._pose_engine = vision.PoseLandmarker.create_from_options(pose_opts)
        self._hand_engine = vision.HandLandmarker.create_from_options(hand_opts)

    @staticmethod
    def _pose_to_array(pose_landmarks) -> np.ndarray:
        arr = np.zeros((33, 3), dtype=np.float32)
        if not pose_landmarks:
            return arr
        pts = pose_landmarks[0]
        n = min(len(pts), 33)
        for i in range(n):
            arr[i, 0] = float(pts[i].x)
            arr[i, 1] = float(pts[i].y)
            arr[i, 2] = float(getattr(pts[i], "visibility", 1.0))
        return arr

    @staticmethod
    def _hands_to_array(hand_landmarks, handedness) -> tuple[np.ndarray, np.ndarray]:
        left = np.zeros((21, 3), dtype=np.float32)
        right = np.zeros((21, 3), dtype=np.float32)
        if not hand_landmarks or not handedness:
            return left, right

        for i, landmarks in enumerate(hand_landmarks):
            side = "Right"
            try:
                side = handedness[i][0].category_name
            except Exception:
                pass

            target = left if side == "Left" else right
            n = min(len(landmarks), 21)
            for j in range(n):
                target[j, 0] = float(landmarks[j].x)
                target[j, 1] = float(landmarks[j].y)
                target[j, 2] = 1.0
        return left, right

    @staticmethod
    def _landmarks_to_array(landmarks, expected_num: int) -> np.ndarray:
        arr = np.zeros((expected_num, 3), dtype=np.float32)
        if landmarks is None:
            return arr
        pts = landmarks.landmark
        n = min(len(pts), expected_num)
        for i in range(n):
            arr[i, 0] = float(pts[i].x)
            arr[i, 1] = float(pts[i].y)
            arr[i, 2] = float(pts[i].z)
        return arr

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        p_res = self._pose_engine.detect_for_video(mp_img, self._ts_ms)
        h_res = self._hand_engine.detect_for_video(mp_img, self._ts_ms)
        self._ts_ms += 33

        pose = self._pose_to_array(getattr(p_res, "pose_landmarks", None))
        left, right = self._hands_to_array(
            getattr(h_res, "hand_landmarks", None),
            getattr(h_res, "handedness", None),
        )
        out = np.concatenate([pose, left, right], axis=0)

        # 三路都没有检测到时不触发分类，避免全零帧造成噪声预测。
        has_any = (
            len(getattr(p_res, "pose_landmarks", []) or []) > 0
            or len(getattr(h_res, "hand_landmarks", []) or []) > 0
        )
        if not has_any:
            return None
        return out

    def close(self) -> None:
        close_pose = getattr(self._pose_engine, "close", None)
        close_hand = getattr(self._hand_engine, "close", None)
        if callable(close_pose):
            close_pose()
        if callable(close_hand):
            close_hand()


def _start_perception_loop(
    cfg: dict,
    mode: str,
    bus: EventBus,
    metrics: Metrics,
):
    pcfg = cfg.get("perception", {})
    if not bool(pcfg.get("enabled", True)):
        log_info("perception loop disabled by config")
        return None, None, None, None

    if mode not in {"reflex_only", "hybrid"}:
        log_info(f"perception loop skipped in mode={mode}")
        return None, None, None, None

    camera_cfg = CameraConfig(
        width=int(pcfg.get("camera_width", 640)),
        height=int(pcfg.get("camera_height", 480)),
        fps=int(pcfg.get("camera_fps", 30)),
        local_camera_index=int(pcfg.get("local_camera_index", 0)),
        unitree_network_interface=pcfg.get("unitree_network_interface"),
        unitree_timeout_sec=float(pcfg.get("unitree_timeout_sec", 3.0)),
    )

    stream = MediaPipeStream(config=camera_cfg)
    extractor = HolisticKeypointExtractor(
        pose_model_path=str(
            pcfg.get(
                "pose_model_path",
                str(ROOT / "models" / "MediaPipe_Models" / "pose_landmarker_lite.task"),
            )
        ),
        hand_model_path=str(
            pcfg.get(
                "hand_model_path",
                str(ROOT / "models" / "MediaPipe_Models" / "hand_landmarker.task"),
            )
        ),
        min_detection_confidence=float(pcfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(pcfg.get("min_tracking_confidence", 0.5)),
        num_hands=int(pcfg.get("num_hands", 2)),
        use_gpu_delegate=bool(pcfg.get("use_gpu_delegate", True)),
    )
    classifier = GestureClassifier(
        weights_path=pcfg.get("gesture_weights_path"),
        sequence_len=int(pcfg.get("sequence_len", 100)),
        num_classes=int(pcfg.get("num_classes", 13)),
        confidence_threshold=float(pcfg.get("gesture_conf_threshold", 0.85)),
    )

    stream.start()
    log_info(f"perception stream started: source={stream.source_name}")

    stop_evt = threading.Event()

    def _loop():
        log_info("perception loop running")
        while not stop_evt.is_set():
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            feat = extractor.extract(frame)
            if feat is None:
                continue

            event = classifier.update_and_predict(feat)
            if event is None:
                continue

            if bus.publish(event):
                metrics.inc("perception_event_publish")
            else:
                metrics.inc("perception_event_drop")

    t = threading.Thread(target=_loop, daemon=True, name="perception-loop")
    t.start()
    return stop_evt, t, stream, extractor


def _start_event_dispatcher(
    mode: str,
    bus: EventBus,
    queue_adapter: TaskQueueAdapter,
    planner: PlannerClient,
    reflex: ReflexBridge,
    metrics: Metrics,
):
    stop_evt = threading.Event()

    def _loop():
        log_info("event dispatcher running")
        while not stop_evt.is_set():
            try:
                event = bus.consume(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                continue

            if isinstance(event, SpeechEvent):
                text = event.text.strip()
                route = route_text(text)
                if route == "empty":
                    continue
                if route == "emergency":
                    queue_adapter.clear()
                    queue_adapter.push_tasks([ActionTask(action_id=1, duration=0.0, source="emergency")])
                    queue_adapter.wait_all_done()
                    metrics.inc("emergency")
                    continue

                if mode in {"planner_only", "hybrid"}:
                    plan = planner.plan(text)
                    if plan.actions:
                        queue_adapter.push_tasks(plan.actions)
                        metrics.inc("planner_plan")
                    else:
                        log_warn(f"planner returned empty actions: seq={plan.sequence_name}")
                queue_adapter.wait_all_done()
                continue

            if isinstance(event, PerceptionEvent):
                if mode not in {"reflex_only", "hybrid"}:
                    continue
                task = reflex.to_task(event)
                if task is None:
                    continue
                queue_adapter.push_tasks([task])
                metrics.inc("reflex_emit")
                log_info(
                    f"reflex emit: pred={event.pred_id}, conf={event.confidence:.3f}, "
                    f"action={task.action_id}, pending={queue_adapter.pending_count()}"
                )
                continue

            if isinstance(event, SystemEvent):
                if event.level.lower() in {"warn", "warning"}:
                    log_warn(f"[{event.source}] {event.message}")
                else:
                    log_info(f"[{event.source}] {event.message}")
                metrics.inc("system_event")
                continue

            if isinstance(event, ErrorEvent):
                log_warn(
                    f"[{event.source}] {event.exc_type}: {event.message}"
                )
                metrics.inc("error_event")
                continue

    t = threading.Thread(target=_loop, daemon=True, name="event-dispatcher")
    t.start()
    return stop_evt, t


def run() -> None:
    cfg = load_cfg()
    mode = cfg["pipeline"].get("mode", "hybrid")
    queue_size = int(cfg["execution"].get("queue_size", 20))
    hb_sec = float(cfg["pipeline"].get("heartbeat_sec", 2.0))
    bus_maxsize = int(cfg["pipeline"].get("event_bus_size", 512))
    use_text_cli = bool(cfg["pipeline"].get("use_text_cli", True))

    # 启动前资源检查/自动拉取
    _ensure_resources(cfg)

    # 先确保大模型可用，再进行后续任何组件初始化
    vllm_mgr = VLLMProcessManager(cfg)
    vllm_mgr.maybe_start()

    metrics = Metrics()
    hb = Heartbeat(interval_sec=hb_sec)
    hb.start()

    bus = EventBus(maxsize=bus_maxsize)
    bus.publish(SystemEvent(level="info", source="pipeline", message="event bus initialized"))
    queue_adapter = TaskQueueAdapter(max_size=queue_size)
    fsm = FSMAdapter(queue_adapter.inner)
    fsm.start()
    bus.publish(SystemEvent(level="info", source="execution", message="fsm started"))

    planner = PlannerClient(config_path=cfg["planner"]["config_path"])
    reflex = ReflexBridge(
        conf_threshold=float(cfg["pipeline"].get("reflex_conf_threshold", 0.85)),
        cooldown_sec=float(cfg["pipeline"].get("reflex_cooldown_sec", 1.0)),
    )
    dispatcher_stop_evt, dispatcher_thread = _start_event_dispatcher(
        mode=mode,
        bus=bus,
        queue_adapter=queue_adapter,
        planner=planner,
        reflex=reflex,
        metrics=metrics,
    )
    perception_stop_evt = None
    perception_thread = None
    perception_stream = None
    perception_extractor = None
    try:
        (
            perception_stop_evt,
            perception_thread,
            perception_stream,
            perception_extractor,
        ) = _start_perception_loop(cfg, mode, bus, metrics)
        bus.publish(SystemEvent(level="info", source="perception", message="perception loop started"))
    except Exception as e:
        log_warn(f"perception loop unavailable: {e}")
        bus.publish(
            ErrorEvent(
                source="perception",
                message=str(e),
                exc_type=type(e).__name__,
            )
        )

    log_info(f"pipeline started, mode={mode}")
    if use_text_cli:
        log_info("input text; use 'exit' to quit, 'stop' for emergency")
    else:
        log_info("text cli disabled; pipeline running with background producers only")

    try:
        if use_text_cli:
            while True:
                text = input("\n🗣️ text> ").strip()
                if text.lower() in {"exit", "quit"}:
                    break
                if not bus.publish(SpeechEvent(text=text)):
                    log_warn("event bus full, drop speech event")
                    bus.publish(
                        ErrorEvent(
                            source="pipeline",
                            message="event bus full while publishing speech event",
                            exc_type="QueueFull",
                        )
                    )
                time.sleep(0.05)
        else:
            while True:
                time.sleep(1.0)
                state = fsm.runtime_state()
                log_info(
                    f"state action={state['action_name']}#{state['action_id']} "
                    f"v=({state['vx']:+.2f},{state['vy']:+.2f},{state['omega']:+.2f}) "
                    f"metrics={metrics.snapshot()}"
                )
    finally:
        bus.publish(SystemEvent(level="info", source="pipeline", message="shutting down"))
        dispatcher_stop_evt.set()
        dispatcher_thread.join(timeout=1.0)
        if perception_stop_evt is not None:
            perception_stop_evt.set()
        if perception_thread is not None:
            perception_thread.join(timeout=1.0)
        if perception_stream is not None:
            perception_stream.stop()
        if perception_extractor is not None:
            perception_extractor.close()
        hb.stop()
        vllm_mgr.stop()
        log_info("pipeline stopped")


if __name__ == "__main__":
    run()

