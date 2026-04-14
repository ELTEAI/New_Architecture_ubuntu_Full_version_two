from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from src.pipeline.contracts import PerceptionEvent


class GestureClassifier:
    """
    1D-CNN+GRU 手势分类封装（最小可用版）。

    支持两种调用方式：
    1) `update_and_predict(frame_features)`：逐帧喂入 `(75,3)`，内部维护滑窗；
    2) `predict_from_sequence(sequence)`：直接输入序列 `(T,75,3)` 推理。
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        sequence_len: int = 100,
        num_classes: int = 13,
        device: str | None = None,
        confidence_threshold: float = 0.85,
    ):
        try:
            import torch
            import torch.nn as nn
        except Exception as e:  # pragma: no cover
            raise RuntimeError("GestureClassifier 需要安装 torch。") from e

        self.torch = torch
        self.nn = nn
        self.sequence_len = int(sequence_len)
        self.confidence_threshold = float(confidence_threshold)
        self.num_classes = int(num_classes)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._buffer: deque[np.ndarray] = deque(maxlen=self.sequence_len)

        self.model = self._build_model().to(self.device)
        weights = self._resolve_weights_path(weights_path)
        state = torch.load(str(weights), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _resolve_weights_path(weights_path: str | Path | None) -> Path:
        """
        权重路径优先级：
        1) 显式传入 weights_path
        2) 环境变量 VLA_GESTURE_WEIGHTS
        3) VLA_Pipeline/models/best_mp_gesture_model.pth
        4) 旧工程路径回退
        """
        if weights_path is not None:
            p = Path(weights_path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"手势模型权重不存在: {p}")

        env_path = Path(
            os.getenv("VLA_GESTURE_WEIGHTS", "")
        ) if os.getenv("VLA_GESTURE_WEIGHTS") else None
        if env_path is not None and env_path.is_file():
            return env_path

        pipeline_default = Path(__file__).resolve().parents[2] / "models" / "best_mp_gesture_model.pth"
        if pipeline_default.is_file():
            return pipeline_default

        legacy = Path("/home/ubuntu/New_Architecture/1D_CNN_GRU_mediapipe/trained_model/best_mp_gesture_model.pth")
        if legacy.is_file():
            return legacy

        raise FileNotFoundError(
            "未找到手势模型权重。请将权重放到 VLA_Pipeline/models/best_mp_gesture_model.pth，"
            "或通过参数 weights_path / 环境变量 VLA_GESTURE_WEIGHTS 指定。"
        )

    def _build_model(self):
        nn = self.nn
        num_classes = self.num_classes

        class RobotGestureClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(225, 128, 5, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(128, 256, 3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
                self.gru = nn.GRU(256, 128, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                x = x.permute(0, 2, 1)  # [B,225,100]
                x = self.cnn(x)         # [B,256,25]
                x = x.permute(0, 2, 1)  # [B,25,256]
                _, h_n = self.gru(x)
                return self.fc(h_n[-1])

        return RobotGestureClassifier()

    @staticmethod
    def _to_frame_array(frame_features: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
        arr = np.asarray(frame_features, dtype=np.float32)
        if arr.shape != (75, 3):
            raise ValueError(f"单帧特征必须是 (75,3)，当前: {arr.shape}")
        return arr

    def _prepare_input(self, sequence: np.ndarray) -> "torch.Tensor":
        """
        输入序列 shape: (T,75,3) -> 模型输入 (1,100,225)
        """
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[1:] != (75, 3):
            raise ValueError(f"序列必须为 (T,75,3)，当前: {seq.shape}")

        if len(seq) < self.sequence_len:
            pad = np.repeat(seq[:1], repeats=self.sequence_len - len(seq), axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        seq = seq[-self.sequence_len :]

        x = self.torch.from_numpy(seq).float().view(1, self.sequence_len, -1)
        return x.to(self.device)

    def predict_from_sequence(self, sequence: np.ndarray) -> PerceptionEvent | None:
        x = self._prepare_input(sequence)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0]
            pred_id = int(self.torch.argmax(probs).item())
            conf = float(probs[pred_id].item())

        if conf < self.confidence_threshold:
            return None
        return PerceptionEvent(pred_id=pred_id, confidence=conf)

    def update_and_predict(self, frame_features: np.ndarray | Iterable[Iterable[float]]) -> PerceptionEvent | None:
        frame = self._to_frame_array(frame_features)
        self._buffer.append(frame)
        if len(self._buffer) == 0:
            return None
        seq = np.stack(list(self._buffer), axis=0)
        return self.predict_from_sequence(seq)

