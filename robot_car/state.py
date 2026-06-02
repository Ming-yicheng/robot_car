"""线程安全的运行状态。

主程序同时有视频线程、语音线程、底盘控制循环和 Web 上传逻辑。它们都需要读写
人脸位置、情绪结果、语音状态等共享数据。为了避免旧主函数里大量全局变量造成的
竞态问题，统一用 `RobotState` 管理，并用可重入锁保护。
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RobotState:
    """小车运行时共享状态。

    所有字段都保持简单类型，方便序列化上传到 Web，也方便调试时打印 snapshot。
    写入状态必须通过下面的 update 方法，读取状态使用 `snapshot()` 获取一份稳定拷贝。
    """

    lock: threading.RLock = field(default_factory=threading.RLock)
    face_detected: bool = False
    face_error_pan: float = 0.0
    face_width: float = 0.0
    pan: float = 70.0
    tilt: float = 0.0
    facial_emotion_label: Optional[str] = None
    facial_emotion_confidence: float = 0.0
    speech_emotion_label: Optional[str] = None
    speech_emotion_confidence: float = 0.0
    input_text: Optional[str] = None
    text_emotion: Optional[str] = None
    text_emotion_confidence: float = 0.0
    output_text: Optional[str] = None
    voice_active: bool = True
    listening_for_voice: bool = False
    speaking: bool = False
    emotion_sequence: list[tuple[str, float]] = field(default_factory=list)

    def update_gimbal(self, pan: float, tilt: float) -> None:
        """更新云台当前角度。"""
        with self.lock:
            self.pan = pan
            self.tilt = tilt

    def update_face(self, detected: bool, error_pan: float = 0.0, width: float = 0.0) -> None:
        """更新人脸检测结果。

        `error_pan` 表示人脸中心相对画面中心的水平偏差，单位是像素；`width` 表示
        人脸框宽度，用来粗略估计人与小车之间的距离。
        """
        with self.lock:
            self.face_detected = detected
            self.face_error_pan = error_pan
            self.face_width = width
            if not detected:
                self.facial_emotion_label = None
                self.facial_emotion_confidence = 0.0

    def update_facial_emotion(self, label: str, confidence: float, collect: bool) -> None:
        """更新面部情绪。

        语音录制期间可以把每帧情绪加入序列，等一次对话结束后聚合成更稳定的结果。
        """
        with self.lock:
            self.facial_emotion_label = label
            self.facial_emotion_confidence = float(confidence)
            if collect:
                self.emotion_sequence.append((label, float(confidence)))

    def aggregate_facial_emotion(self) -> tuple[Optional[str], float]:
        """聚合一次对话期间收集到的面部情绪序列。"""

        with self.lock:
            if not self.emotion_sequence:
                self.facial_emotion_label = None
                self.facial_emotion_confidence = 0.0
                return None, 0.0

            labels = [item[0] for item in self.emotion_sequence]
            dominant_label, count = collections.Counter(labels).most_common(1)[0]
            confidence = count / len(self.emotion_sequence)
            self.facial_emotion_label = dominant_label
            self.facial_emotion_confidence = confidence
            self.emotion_sequence.clear()
            return dominant_label, confidence

    def set_voice_flags(
        self,
        *,
        listening: Optional[bool] = None,
        speaking: Optional[bool] = None,
        active: Optional[bool] = None,
    ) -> None:
        """设置语音状态标志。

        `listening_for_voice` 和 `speaking` 会影响底盘速度：语音交互期间底盘会降速，
        防止小车移动影响录音，也降低用户靠近时的安全风险。
        """
        with self.lock:
            if listening is not None:
                self.listening_for_voice = listening
            if speaking is not None:
                self.speaking = speaking
            if active is not None:
                self.voice_active = active

    def update_speech_emotion(self, label: Optional[str], confidence: float) -> None:
        """更新语音情绪识别结果。"""
        with self.lock:
            self.speech_emotion_label = label
            self.speech_emotion_confidence = float(confidence or 0.0)

    def update_text_emotion(self, text: Optional[str], emotion: Optional[str], confidence: float) -> None:
        """更新云端转写文本及文本情绪。"""
        with self.lock:
            self.input_text = text
            self.text_emotion = emotion
            self.text_emotion_confidence = float(confidence or 0.0)

    def update_output_text(self, text: Optional[str]) -> None:
        """更新 AI 回复文本，供 Web 上传使用。"""
        with self.lock:
            self.output_text = text

    def snapshot(self) -> dict[str, Any]:
        """返回当前状态快照。

        返回的是普通 dict，调用者可以放心在锁外使用，不会被其他线程中途改动。
        """
        with self.lock:
            return {
                "face_detected": self.face_detected,
                "face_error_pan": self.face_error_pan,
                "face_width": self.face_width,
                "pan": self.pan,
                "tilt": self.tilt,
                "facial_emotion_label": self.facial_emotion_label,
                "facial_emotion_confidence": self.facial_emotion_confidence,
                "speech_emotion_label": self.speech_emotion_label,
                "speech_emotion_confidence": self.speech_emotion_confidence,
                "input_text": self.input_text,
                "text_emotion": self.text_emotion,
                "text_emotion_confidence": self.text_emotion_confidence,
                "output_text": self.output_text,
                "voice_active": self.voice_active,
                "listening_for_voice": self.listening_for_voice,
                "speaking": self.speaking,
            }
