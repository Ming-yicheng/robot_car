"""摄像头抽象。

当前 Orange Pi 项目默认使用 OpenCV 打开 `/dev/video0`，这是远端环境中已经存在的
设备。为了兼容用户提供的旧主函数，也保留 Picamera2 后端，但只有显式选择时才会
导入 Picamera2，避免普通 OpenCV 模式下因为缺少库而报错。
"""

from __future__ import annotations

import cv2

from robot_car.config import CameraConfig


class OpenCVCamera:
    """基于 OpenCV VideoCapture 的摄像头。"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {config.device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def read(self):
        """读取一帧图像。读取失败时返回 None。"""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return _apply_flip(frame, self.config)

    def close(self) -> None:
        self.cap.release()


class Picamera2Camera:
    """基于 Picamera2 的摄像头后端。"""

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
        """读取一帧图像。"""
        frame = self.camera.capture_array()
        return _apply_flip(frame, self.config)

    def close(self) -> None:
        self.camera.stop()


def _apply_flip(frame, config: CameraConfig):
    """根据配置翻转图像。

    旧主函数中摄像头设置了垂直翻转，这里用统一函数处理 OpenCV 和 Picamera2。
    """

    if config.flip_vertical and config.flip_horizontal:
        return cv2.flip(frame, -1)
    if config.flip_vertical:
        return cv2.flip(frame, 0)
    if config.flip_horizontal:
        return cv2.flip(frame, 1)
    return frame


def create_camera(config: CameraConfig):
    """按配置创建摄像头后端。"""

    backend = config.backend.lower()
    if backend == "picamera2":
        return Picamera2Camera(config)
    if backend == "opencv":
        return OpenCVCamera(config)
    raise ValueError(f"Unsupported camera backend: {config.backend}")
