"""可选的 TFLite 语音情绪识别。

旧主函数使用 `SER.tflite` 进行语音情绪识别。当前 Orange Pi 环境导入 TensorFlow
会崩溃，因此这里只依赖 `tflite_runtime`，并手写了序列 padding 逻辑。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeechEmotionRecognizer:
    """从录音文件中识别语音情绪。"""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError as exc:
            raise RuntimeError("tflite_runtime is required for speech emotion recognition") from exc

        self._librosa = __import__("librosa")
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = ["neutral", "happy", "surprise", "unpleasant"]
        self.sample_rate = 16000
        self.max_len = 47

    def predict_file(self, audio_path: Path) -> tuple[Optional[str], float]:
        """读取 WAV 文件，提取 MFCC，并输出情绪标签和置信度。"""
        audio_data, _ = self._librosa.load(str(audio_path), sr=self.sample_rate)
        mfcc = self._librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512,
        ).T
        padded = self._pad_features(mfcc)
        self.interpreter.set_tensor(self.input_details[0]["index"], padded)
        self.interpreter.invoke()
        probabilities = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        emotion_id = int(np.argmax(probabilities))
        return self.labels[emotion_id], float(probabilities[emotion_id])

    def _pad_features(self, features: np.ndarray) -> np.ndarray:
        """把 MFCC 序列补齐或截断到模型固定长度。

        这样可以替代 `tensorflow.keras.preprocessing.sequence.pad_sequences`，
        避免在 Orange Pi 上导入 TensorFlow。
        """

        output = np.zeros((1, self.max_len, features.shape[1]), dtype=np.float32)
        usable = min(self.max_len, features.shape[0])
        output[0, :usable, :] = features[:usable].astype(np.float32)
        return output


def maybe_create_speech_emotion(model_path: Path) -> Optional[SpeechEmotionRecognizer]:
    """尝试创建语音情绪识别器。

    模型或依赖不存在时返回 None，让主流程自动跳过 SER 功能。
    """
    if not model_path.exists():
        logger.warning("Speech emotion model missing, SER disabled: %s", model_path)
        return None
    try:
        recognizer = SpeechEmotionRecognizer(model_path)
        logger.info("Loaded speech emotion model: %s", model_path)
        return recognizer
    except Exception as exc:
        logger.warning("Speech emotion recognizer disabled: %s", exc)
        return None
