"""Thread-safe runtime state shared by camera, voice, web, and wheel loops."""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RobotState:
    """Small mutable state container protected by a lock.

    The previous main script used many module-level globals.  Keeping the same
    data in one object makes thread ownership clearer and avoids subtle races.
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
        with self.lock:
            self.pan = pan
            self.tilt = tilt

    def update_face(self, detected: bool, error_pan: float = 0.0, width: float = 0.0) -> None:
        with self.lock:
            self.face_detected = detected
            self.face_error_pan = error_pan
            self.face_width = width
            if not detected:
                self.facial_emotion_label = None
                self.facial_emotion_confidence = 0.0

    def update_facial_emotion(self, label: str, confidence: float, collect: bool) -> None:
        with self.lock:
            self.facial_emotion_label = label
            self.facial_emotion_confidence = float(confidence)
            if collect:
                self.emotion_sequence.append((label, float(confidence)))

    def aggregate_facial_emotion(self) -> tuple[Optional[str], float]:
        """Return the dominant facial emotion collected during one voice turn."""

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
        with self.lock:
            if listening is not None:
                self.listening_for_voice = listening
            if speaking is not None:
                self.speaking = speaking
            if active is not None:
                self.voice_active = active

    def update_speech_emotion(self, label: Optional[str], confidence: float) -> None:
        with self.lock:
            self.speech_emotion_label = label
            self.speech_emotion_confidence = float(confidence or 0.0)

    def update_text_emotion(self, text: Optional[str], emotion: Optional[str], confidence: float) -> None:
        with self.lock:
            self.input_text = text
            self.text_emotion = emotion
            self.text_emotion_confidence = float(confidence or 0.0)

    def update_output_text(self, text: Optional[str]) -> None:
        with self.lock:
            self.output_text = text

    def snapshot(self) -> dict[str, Any]:
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
