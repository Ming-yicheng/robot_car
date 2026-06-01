"""Camera abstraction.

OpenCV is the default on the Orange Pi project because /dev/video0 is present.
Picamera2 support is kept optional for compatibility with the uploaded main
function, but it is not required to import this module.
"""

from __future__ import annotations

import cv2

from robot_car.config import CameraConfig


class OpenCVCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {config.device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return _apply_flip(frame, self.config)

    def close(self) -> None:
        self.cap.release()


class Picamera2Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        from picamera2 import Picamera2

        self.camera = Picamera2()
        preview_config = self.camera.create_preview_configuration(
            main={"format": "RGB888", "size": (config.width, config.height)}
        )
        self.camera.configure(preview_config)
        self.camera.start()

    def read(self):
        frame = self.camera.capture_array()
        return _apply_flip(frame, self.config)

    def close(self) -> None:
        self.camera.stop()


def _apply_flip(frame, config: CameraConfig):
    if config.flip_vertical and config.flip_horizontal:
        return cv2.flip(frame, -1)
    if config.flip_vertical:
        return cv2.flip(frame, 0)
    if config.flip_horizontal:
        return cv2.flip(frame, 1)
    return frame


def create_camera(config: CameraConfig):
    backend = config.backend.lower()
    if backend == "picamera2":
        return Picamera2Camera(config)
    if backend == "opencv":
        return OpenCVCamera(config)
    raise ValueError(f"Unsupported camera backend: {config.backend}")
