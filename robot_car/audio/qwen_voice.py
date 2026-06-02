"""本地 Qwen / RKLLM 语音对话客户端。

这一版不再依赖 DashScope、OpenAI SDK 或 API Key。一次语音回合拆成三段：
    1. sherpa-onnx + SenseVoice ONNX 本地语音转写
    2. 本机 RKLLM Server 调用 `.rkllm` 大语言模型
    3. sherpa-onnx + MeloTTS/VITS ONNX 本地合成并播放语音

RKLLM Server 需要单独启动；如果它不可用，主程序只记录日志并跳过回复，不会影响
视觉、跟随和避障流程。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from robot_car.config import AudioConfig, ProjectPaths, QwenConfig

logger = logging.getLogger(__name__)


@dataclass
class TextEmotionResult:
    """本地语音转写结果。

    目前没有本地文本情绪模型，所以 emotion/confidence 保留为空值，供 Web/状态层兼容。
    """

    text: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float = 0.0


@dataclass
class VoiceReply:
    """本地大模型回复文本。"""

    text: str


class LocalSpeechRecognizer:
    """使用 sherpa-onnx SenseVoice 模型做本地 ASR。"""

    def __init__(self, paths: ProjectPaths, config: QwenConfig):
        if not paths.sense_voice_model_path.exists():
            raise FileNotFoundError(paths.sense_voice_model_path)
        if not paths.sense_voice_tokens_path.exists():
            raise FileNotFoundError(paths.sense_voice_tokens_path)

        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(paths.sense_voice_model_path),
            tokens=str(paths.sense_voice_tokens_path),
            num_threads=config.local_num_threads,
            sample_rate=16000,
            language=config.asr_language,
            use_itn=config.asr_use_itn,
            provider=config.local_provider,
        )
        self.sample_rate = 16000
        logger.info("Loaded local ASR model: %s", paths.sense_voice_model_path)

    def transcribe(self, audio_path: Path) -> str:
        """识别 WAV 文件并返回文本。"""

        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if sample_rate != self.sample_rate:
            import librosa

            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=self.sample_rate)
            sample_rate = self.sample_rate
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, np.ascontiguousarray(samples, dtype=np.float32))
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


class LocalTextToSpeech:
    """使用 sherpa-onnx VITS/MeloTTS 模型做本地语音合成和播放。"""

    def __init__(self, paths: ProjectPaths, qwen_config: QwenConfig, audio_config: AudioConfig):
        for path in (paths.local_tts_model_path, paths.local_tts_lexicon_path, paths.local_tts_tokens_path):
            if not path.exists():
                raise FileNotFoundError(path)

        import sherpa_onnx

        self.config = qwen_config
        self.audio_config = audio_config
        self.tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(paths.local_tts_model_path),
                        lexicon=str(paths.local_tts_lexicon_path),
                        tokens=str(paths.local_tts_tokens_path),
                        data_dir=str(paths.local_tts_dir),
                    ),
                    num_threads=qwen_config.local_num_threads,
                    provider=qwen_config.local_provider,
                ),
                max_num_sentences=1,
            )
        )
        logger.info("Loaded local TTS model: %s", paths.local_tts_model_path)

    def speak(self, text: str) -> None:
        """合成并播放文本。"""

        if not text.strip():
            return

        audio = self.tts.generate(
            text,
            sid=self.config.tts_speaker_id,
            speed=self.config.tts_speed,
        )

        import pyaudio

        player = pyaudio.PyAudio()
        stream = player.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=audio.sample_rate,
            output=True,
            output_device_index=self.audio_config.output_device,
        )
        try:
            samples = np.asarray(audio.samples, dtype=np.float32)
            stream.write(samples.tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            player.terminate()


class QwenVoiceClient:
    """本地语音对话客户端，保留旧类名以兼容主程序。"""

    def __init__(self, qwen_config: QwenConfig, audio_config: AudioConfig, paths: Optional[ProjectPaths] = None):
        self.qwen_config = qwen_config
        self.audio_config = audio_config
        self.paths = paths or ProjectPaths()
        self.history: list[tuple[str, str]] = []
        self.session = requests.Session()
        self._last_audio_path: Optional[Path] = None
        self._last_transcription = TextEmotionResult()

        self.asr: Optional[LocalSpeechRecognizer] = None
        if qwen_config.asr_enabled:
            try:
                self.asr = LocalSpeechRecognizer(self.paths, qwen_config)
            except Exception as exc:
                logger.warning("Local ASR disabled: %s", exc)

        self.tts: Optional[LocalTextToSpeech] = None
        if qwen_config.tts_enabled:
            try:
                self.tts = LocalTextToSpeech(self.paths, qwen_config, audio_config)
            except Exception as exc:
                logger.warning("Local TTS disabled: %s", exc)

    def transcribe_text_emotion(self, audio_path: Path) -> TextEmotionResult:
        """本地语音转文字。

        函数名保留旧语义，但不再调用云端文本情绪模型。
        """

        if self.asr is None:
            return TextEmotionResult()

        try:
            text = self.asr.transcribe(audio_path)
            result = TextEmotionResult(text=text or None)
            self._last_audio_path = audio_path
            self._last_transcription = result
            if text:
                logger.info("Local ASR: %s", text)
            return result
        except Exception as exc:
            logger.warning("Local ASR failed: %s", exc)
            return TextEmotionResult()

    def chat_with_audio(self, audio_path: Path, *, facial_emotion: Optional[str], speech_emotion: Optional[str]) -> VoiceReply:
        """把本地转写文本发送到 RKLLM Server，并播放本地 TTS 回复。"""

        transcription = self._last_transcription if self._last_audio_path == audio_path else self.transcribe_text_emotion(audio_path)
        user_text = (transcription.text or "").strip()
        if not user_text:
            logger.info("No local ASR text; skipping RKLLM chat")
            return VoiceReply(text="")

        reply_text = self._chat_with_rkllm(user_text, facial_emotion=facial_emotion, speech_emotion=speech_emotion)
        if not reply_text:
            return VoiceReply(text="")

        if self.tts is not None:
            try:
                self.tts.speak(reply_text)
            except Exception as exc:
                logger.warning("Local TTS playback failed: %s", exc)

        self.history.append((user_text, reply_text))
        self.history = self.history[-5:]
        return VoiceReply(text=reply_text)

    def _chat_with_rkllm(
        self,
        user_text: str,
        *,
        facial_emotion: Optional[str],
        speech_emotion: Optional[str],
    ) -> str:
        """调用本机 RKLLM Server 的 OpenAI-like `/rkllm_chat` 接口。"""

        prompt = self._build_prompt(user_text, facial_emotion, speech_emotion)
        payload = {
            "model": self.qwen_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "enable_thinking": False,
        }
        try:
            response = self.session.post(
                self.qwen_config.rkllm_server_url,
                json=payload,
                timeout=self.qwen_config.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][-1]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("RKLLM server unavailable or failed: %s", exc)
            return ""

    def _build_prompt(self, user_text: str, facial_emotion: Optional[str], speech_emotion: Optional[str]) -> str:
        """把系统提示词、情绪上下文和短历史合成单轮 prompt。

        RKLLM 示例 server 对多条 messages 的处理较简单，因此这里把上下文压成一条
        user 消息，避免 system/user 多消息时被示例 server 跳过。
        """

        parts = [self.qwen_config.system_prompt]
        if facial_emotion:
            parts.append(f"视觉情绪识别结果：{facial_emotion}。")
        if speech_emotion:
            parts.append(f"语音情绪识别结果：{speech_emotion}。")
        if self.history:
            parts.append("最近对话：")
            for user, assistant in self.history[-3:]:
                parts.append(f"用户：{user}")
                parts.append(f"助手：{assistant}")
        parts.append("请始终使用中文回复，语气自然，回答简短，不要解释内部识别流程。")
        parts.append(f"用户：{user_text}")
        parts.append("助手：")
        return "\n".join(parts)
