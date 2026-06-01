"""Socket.IO telemetry and video streaming."""

from __future__ import annotations

import base64
import datetime as dt
import logging
from typing import Optional

import cv2

from robot_car.config import WebConfig
from robot_car.state import RobotState

logger = logging.getLogger(__name__)


class TelemetryClient:
    """Non-fatal Socket.IO wrapper.

    If the web server is down, robot control should continue.  All networking
    failures are logged and swallowed.
    """

    def __init__(self, config: WebConfig):
        self.config = config
        self.sio = None
        self.connected = False
        if not config.enabled or not config.server_url:
            logger.info("Web telemetry disabled")
            return
        try:
            import socketio

            self.sio = socketio.Client()
            self._register_events()
        except Exception as exc:
            logger.warning("Socket.IO unavailable: %s", exc)

    def _register_events(self) -> None:
        @self.sio.event
        def connect():
            self.connected = True
            logger.info("Connected to web server")

        @self.sio.event
        def connect_error(data):
            self.connected = False
            logger.warning("Failed to connect to web server: %s", data)

        @self.sio.event
        def disconnect():
            self.connected = False
            logger.info("Disconnected from web server")

    def connect(self) -> None:
        if self.sio is None:
            return
        try:
            self.sio.connect(self.config.server_url)
        except Exception as exc:
            logger.warning("Web server connection skipped: %s", exc)

    def send_robot_data(self, state: RobotState) -> None:
        if not self._can_emit():
            return
        state.aggregate_facial_emotion()
        snapshot = state.snapshot()
        payload = {
            "user_id": self.config.user_id,
            "facial_emotion": snapshot["facial_emotion_label"],
            "facial_confidence": snapshot["facial_emotion_confidence"],
            "speech_emotion": snapshot["speech_emotion_label"],
            "speech_confidence": snapshot["speech_emotion_confidence"],
            "text_emotion": snapshot["text_emotion"],
            "text_confidence": snapshot["text_emotion_confidence"],
            "input_text": snapshot["input_text"],
            "output_text": snapshot["output_text"],
            "timestamp": dt.datetime.utcnow().isoformat(),
        }
        self._emit("robot_data", payload)

    def send_video_frame(self, frame, jpeg_quality: int = 80) -> None:
        if not self._can_emit():
            return
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        self._emit("video_frame", {"frame": frame_b64})

    def _can_emit(self) -> bool:
        return self.sio is not None and bool(self.connected)

    def _emit(self, event: str, payload: dict) -> None:
        try:
            self.sio.emit(event, payload)
        except Exception as exc:
            logger.warning("Failed to emit %s: %s", event, exc)

    def close(self) -> None:
        if self.sio is not None and self.connected:
            self.sio.disconnect()
