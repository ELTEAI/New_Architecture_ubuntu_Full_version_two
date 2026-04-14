from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from src.pipeline.contracts import SpeechEvent


class WhisperInput:
    """
    OpenAI Whisper 输入适配器（文件识别版）。

    目标：
    1) 统一读取 `.env` / 环境变量的 API Key；
    2) 提供稳定的音频文件转文本接口；
    3) 输出管道标准事件 `SpeechEvent`。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "whisper-1",
        language: str = "zh",
        repo_root: str | Path | None = None,
    ):
        self._repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self._env_file = self._repo_root / ".env"
        self._load_dotenv_file(self._env_file)

        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                f"未找到 OPENAI_API_KEY。请在环境变量或 {self._env_file} 中配置。"
            )

        self.client = OpenAI(api_key=resolved_key)
        self.model_name = model_name
        self.language = language

    @staticmethod
    def _load_dotenv_file(path: Path) -> None:
        """简单解析 .env（仅处理 KEY=VALUE），不覆盖已存在环境变量。"""
        if not path.is_file():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))

    def transcribe_file(self, audio_path: str | Path) -> SpeechEvent:
        """
        识别本地音频文件并返回 SpeechEvent。
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"音频文件不存在: {path}")

        with open(path, "rb") as audio_file:
            res = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
                language=self.language,
            )
        text = (res.text or "").strip()
        return SpeechEvent(text=text)

    def pull(self) -> SpeechEvent | None:
        """
        保留统一接口：当前版本不直接监听麦克风，返回 None。
        可在 orchestrator 中显式调用 `transcribe_file(...)`。
        """
        return None

