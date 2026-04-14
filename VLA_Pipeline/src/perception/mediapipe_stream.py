from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator
import threading

import cv2
import numpy as np


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    local_camera_index: int = 0
    unitree_network_interface: str | None = None
    unitree_timeout_sec: float = 3.0


class _UnitreeGo2Camera:
    """Wrap Unitree VideoClient to expose read() -> (ok, frame_bgr)."""

    def __init__(self, client, timeout_sec: float):
        self._client = client
        self._timeout_sec = timeout_sec
        self._lock = threading.Lock()

        set_timeout = getattr(self._client, "SetTimeout", None)
        if callable(set_timeout):
            set_timeout(timeout_sec)

        init_fn = getattr(self._client, "Init", None) or getattr(self._client, "init", None)
        if callable(init_fn):
            init_fn()
        else:
            raise RuntimeError("Unitree VideoClient 缺少 Init/init 方法。")

    def read(self):
        with self._lock:
            get_image = getattr(self._client, "GetImageSample", None)
            if not callable(get_image):
                return False, None
            code, data = get_image()

        if code != 0 or data is None:
            return False, None

        # Go2 SDK 返回 JPEG 字节流，这里解码成 OpenCV BGR 帧。
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        return True, frame

    def stop(self) -> None:
        close_fn = getattr(self._client, "Close", None) or getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


class MediaPipeStream:
    """
    统一摄像头流：
    1) 优先尝试 Unitree Go2 相机（SDK）
    2) 失败则自动回退到本机摄像头

    说明：
    - 由于 Unitree SDK 版本差异较大，这里采用“可插拔 provider”方式：
      你可传入 `unitree_provider` 返回一个对象，需实现 `read()` -> (ok, frame)
    - 若未传入 provider，则尝试几个常见路径导入；失败就走本机摄像头。
    """

    def __init__(
        self,
        config: CameraConfig | None = None,
        unitree_provider: Callable[[], object] | None = None,
    ):
        self.cfg = config or CameraConfig()
        self._unitree_provider = unitree_provider
        self._source_name = "unknown"
        self._stream = None
        self._local_cap: cv2.VideoCapture | None = None

    @property
    def source_name(self) -> str:
        return self._source_name

    def start(self) -> None:
        # 1) 先尝试 Unitree
        stream = self._try_start_unitree()
        if stream is not None:
            self._stream = stream
            self._source_name = "unitree_go2"
            return

        # 2) 回退本机
        cap = cv2.VideoCapture(self.cfg.local_camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头：Unitree 不可用且本机 camera[{self.cfg.local_camera_index}] 打不开。"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self._local_cap = cap
        self._source_name = "local_camera"

    def read(self):
        """
        返回 (ok, frame_bgr)。
        """
        if self._source_name == "unitree_go2":
            # 约定 provider 暴露 read() -> (ok, frame)
            return self._stream.read()

        if self._local_cap is None:
            return False, None
        return self._local_cap.read()

    def frames(self) -> Generator:
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    def stop(self) -> None:
        if self._source_name == "unitree_go2" and self._stream is not None:
            close_fn = getattr(self._stream, "close", None) or getattr(self._stream, "stop", None)
            if callable(close_fn):
                close_fn()
        if self._local_cap is not None:
            self._local_cap.release()
            self._local_cap = None
        self._stream = None
        self._source_name = "stopped"

    def _try_start_unitree(self):
        # A) 用户传入 provider（推荐：你按当前 Go2 SDK 写一个 provider）
        if self._unitree_provider is not None:
            try:
                stream = self._unitree_provider()
                if stream is not None and hasattr(stream, "read"):
                    return stream
            except Exception:
                pass

        # B) 对齐 unitree_sdk2_python 官方示例:
        #    ChannelFactoryInitialize(...) + VideoClient().GetImageSample()
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
            from unitree_sdk2py.go2.video.video_client import VideoClient  # type: ignore

            if self.cfg.unitree_network_interface:
                ChannelFactoryInitialize(0, self.cfg.unitree_network_interface)
            else:
                ChannelFactoryInitialize(0)

            client = VideoClient()
            return _UnitreeGo2Camera(client, self.cfg.unitree_timeout_sec)
        except Exception:
            return None

        return None

