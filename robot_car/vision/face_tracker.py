"""Face tracking, gimbal control, and optional facial emotion recognition."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from robot_car.config import AppConfig
from robot_car.hardware.robot import RobotCar
from robot_car.state import RobotState

logger = logging.getLogger(__name__)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=0)


class FacialEmotionRecognizer:
    """ONNX FER+ emotion classifier used by the uploaded main function."""

    def __init__(self, model_path: Path, labels: tuple[str, ...], input_size: tuple[int, int]):
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels
        self.input_size = input_size

    def predict(self, gray_face) -> tuple[str, float]:
        resized = cv2.resize(gray_face, self.input_size)
        model_input = np.expand_dims(np.expand_dims(resized, axis=0), axis=0).astype(np.float32)
        logits = self.session.run(None, {self.input_name: model_input})[0][0]
        probabilities = softmax(logits)
        idx = int(np.argmax(probabilities))
        return self.labels[idx], float(probabilities[idx])


class FaceTracker:
    """Detect the largest face, move the pan/tilt servos, and update state."""

    def __init__(self, config: AppConfig, state: RobotState, robot: RobotCar):
        self.config = config
        self.state = state
        self.robot = robot
        self.pan = config.servos.initial_pan
        self.tilt = config.servos.initial_tilt
        self.smoothed_error_pan = 0.0
        self.smoothed_error_tilt = 0.0
        self.face_cascade = self._load_cascade(config.paths.face_cascade_path, "haarcascade_frontalface_default.xml")
        self.emotion = self._load_emotion_recognizer()

    def _load_cascade(self, configured_path: Path, fallback_name: str):
        cascade_path = configured_path if configured_path.exists() else Path(cv2.data.haarcascades) / fallback_name
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")
        logger.info("Loaded face cascade: %s", cascade_path)
        return cascade

    def _load_emotion_recognizer(self):
        model_path = self.config.paths.facial_emotion_model_path
        if not model_path.exists():
            logger.warning("Facial emotion model missing, emotion detection disabled: %s", model_path)
            return None
        try:
            recognizer = FacialEmotionRecognizer(
                model_path,
                self.config.face_tracking.emotion_labels,
                self.config.face_tracking.emotion_input_size,
            )
            logger.info("Loaded facial emotion model: %s", model_path)
            return recognizer
        except Exception as exc:
            logger.warning("Facial emotion model failed to load: %s", exc)
            return None

    def process_frame(self, frame) -> np.ndarray:
        """Return an annotated frame and update shared state."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.face_tracking.scale_factor,
            minNeighbors=self.config.face_tracking.min_neighbors,
            minSize=self.config.face_tracking.min_face_size,
        )

        if len(faces) == 0:
            self.state.update_face(False)
            return frame

        x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        error_pan = (x + w / 2.0) - self.config.camera.width / 2.0
        error_tilt = (y + h / 2.0) - self.config.camera.height / 2.0
        smoothing = self.config.face_tracking.smoothing_factor
        self.smoothed_error_pan = smoothing * self.smoothed_error_pan + (1 - smoothing) * error_pan
        self.smoothed_error_tilt = smoothing * self.smoothed_error_tilt + (1 - smoothing) * error_tilt

        if abs(self.smoothed_error_pan) > self.config.face_tracking.pan_deadband_px:
            self.pan += self.smoothed_error_pan / self.config.face_tracking.pan_gain_divisor
        if abs(self.smoothed_error_tilt) > self.config.face_tracking.tilt_deadband_px:
            self.tilt -= self.smoothed_error_tilt / self.config.face_tracking.tilt_gain_divisor

        self.pan = max(min(self.pan, self.config.servos.pan_max), self.config.servos.pan_min)
        self.tilt = max(min(self.tilt, self.config.servos.tilt_max), self.config.servos.tilt_min)
        self.robot.set_servo_angle(self.config.servos.pan_channel, int(self.pan))
        self.robot.set_servo_angle(self.config.servos.tilt_channel, int(self.tilt))
        self.state.update_gimbal(self.pan, self.tilt)
        self.state.update_face(True, self.smoothed_error_pan, float(w))

        if self.emotion is not None:
            face_roi_gray = gray[y : y + h, x : x + w]
            if face_roi_gray.size:
                label, confidence = self.emotion.predict(face_roi_gray)
                snapshot = self.state.snapshot()
                self.state.update_facial_emotion(label, confidence, collect=snapshot["listening_for_voice"])
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (x, max(15, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return frame
