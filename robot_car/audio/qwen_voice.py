"""Qwen voice dialogue client.

This replaces the hard-coded API key and absolute audio path from the uploaded
main function.  Keys are read from .env, qwen.env, or OS environment variables.
"""

from __future__ import annotations

import ast
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from robot_car.config import AudioConfig, QwenConfig

logger = logging.getLogger(__name__)


@dataclass
class TextEmotionResult:
    text: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float = 0.0


@dataclass
class VoiceReply:
    text: str


class QwenVoiceClient:
    """Cloud ASR/chat/TTS client using DashScope OpenAI-compatible mode."""

    def __init__(self, qwen_config: QwenConfig, audio_config: AudioConfig):
        self.qwen_config = qwen_config
        self.audio_config = audio_config
        self.history: list[dict] = []
        self.client = None
        self.api_key = self._load_api_key()
        if self.api_key:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=qwen_config.base_url)
        else:
            logger.warning("No Qwen API key found; voice response is disabled")

    def _load_api_key(self) -> Optional[str]:
        for env_file in (self.qwen_config.env_file, self.qwen_config.fallback_env_file):
            if env_file.exists():
                try:
                    from dotenv import load_dotenv

                    load_dotenv(env_file)
                except Exception:
                    pass
        for name in self.qwen_config.api_key_env_names:
            value = os.getenv(name)
            if value:
                return value
        return None

    @staticmethod
    def _encode_audio(audio_path: Path) -> str:
        with audio_path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def transcribe_text_emotion(self, audio_path: Path) -> TextEmotionResult:
        """Optional DashScope transcription and text-emotion extraction.

        The main response call can still work without this.  If dashscope is not
        installed or the model response is malformed, the caller simply gets an
        empty result.
        """

        if not self.api_key:
            return TextEmotionResult()

        try:
            import dashscope
            from dashscope import MultiModalConversation
        except Exception as exc:
            logger.info("DashScope transcription unavailable: %s", exc)
            return TextEmotionResult()

        dashscope.api_key = self.api_key
        prompt = (
            "请将用户上传的语音内容识别为文字，并提供对应的情绪标签和置信度。"
            "情绪从 ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt'] 中选择。"
            "只返回 Python 字典格式，例如 {'text':'内容','emotion':'neutral','confidence':'0.8'}。"
        )
        messages = [
            {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"audio": audio_path.resolve().as_uri()}, {"text": prompt}]},
        ]
        try:
            response = MultiModalConversation.call(model=self.qwen_config.transcription_model, messages=messages)
            raw_text = response["output"]["choices"][0]["message"]["content"][0]["text"]
            parsed = ast.literal_eval(raw_text)
            return TextEmotionResult(
                text=parsed.get("text"),
                emotion=parsed.get("emotion"),
                confidence=float(parsed.get("confidence", 0.0)),
            )
        except Exception as exc:
            logger.warning("Text emotion transcription failed: %s", exc)
            return TextEmotionResult()

    def chat_with_audio(self, audio_path: Path, *, facial_emotion: Optional[str], speech_emotion: Optional[str]) -> VoiceReply:
        if self.client is None:
            return VoiceReply(text="")

        system_prompt = self._build_system_prompt(facial_emotion, speech_emotion)
        audio_b64 = self._encode_audio(audio_path)
        message = {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:audio/wav;base64,{audio_b64}",
                        "format": "wav",
                    },
                }
            ],
        }
        self.history = self.history[-5:] + [message]
        completion = self.client.chat.completions.create(
            model=self.qwen_config.model,
            messages=[{"role": "system", "content": system_prompt}] + self.history[-6:],
            modalities=["text", "audio"],
            audio={"voice": self.qwen_config.voice, "format": "wav"},
            stream=True,
        )

        import pyaudio

        player = pyaudio.PyAudio()
        stream = player.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_config.playback_sample_rate,
            output=True,
            output_device_index=self.audio_config.output_device,
        )
        result_text = ""
        try:
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                audio_delta = getattr(delta, "audio", None)
                if not audio_delta:
                    continue
                transcript = audio_delta.get("transcript")
                audio_string = audio_delta.get("data")
                if transcript:
                    print(transcript, end="", flush=True)
                    result_text += transcript
                if audio_string:
                    stream.write(base64.b64decode(audio_string))
        finally:
            stream.stop_stream()
            stream.close()
            player.terminate()

        if result_text.strip():
            self.history.append({"role": "assistant", "content": result_text.strip()})
        return VoiceReply(text=result_text.strip())

    def _build_system_prompt(self, facial_emotion: Optional[str], speech_emotion: Optional[str]) -> str:
        parts = [self.qwen_config.system_prompt]
        if facial_emotion:
            parts.append(f"视觉情绪识别结果为 {facial_emotion}。")
        if speech_emotion:
            parts.append(f"语音情绪识别结果为 {speech_emotion}。")
        parts.append("请始终使用中文回复，语气自然，不要解释内部识别流程。")
        return "".join(parts)
