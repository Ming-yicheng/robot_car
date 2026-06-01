"""Central configuration for the robot car.

The old project spread pin numbers, model paths, thresholds, and API settings
across several scripts.  This module keeps them in one place so hardware
changes can be made without editing control logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value, 0)


def _optional_int_env(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    return default if value in (None, "") else int(value, 0)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


@dataclass
class ProjectPaths:
    root: Path = PROJECT_ROOT
    assets_dir: Path = PROJECT_ROOT / "assets"
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def face_cascade_path(self) -> Path:
        return self.assets_dir / "image" / "haarcascade_frontalface_default.xml"

    @property
    def eye_cascade_path(self) -> Path:
        return self.assets_dir / "image" / "haarcascade_eye.xml"

    @property
    def facial_emotion_model_path(self) -> Path:
        return self.models_dir / "emotion-ferplus-8.onnx"

    @property
    def speech_emotion_model_path(self) -> Path:
        return self.models_dir / "SER.tflite"

    @property
    def output_wav_path(self) -> Path:
        return self.data_dir / "output.wav"


@dataclass
class MotorPins:
    """TB6612FNG four-motor wiring.

    These defaults are copied from robust_code/finish module/tb6612_abcd_test.py.
    Periphery uses Linux GPIO line offsets, not Raspberry Pi BCM numbering.
    """

    pwma: int = 36
    ain1: int = 39
    ain2: int = 40
    pwmb: int = 42
    bin1: int = 43
    bin2: int = 41
    pwmc: int = 33
    cin1: int = 97
    cin2: int = 32
    pwmd: int = 109
    din1: int = 110
    din2: int = 34
    pwm_frequency_hz: float = 50.0
    control_slice_seconds: float = 0.05


@dataclass
class ServoConfig:
    """PCA9685 servo board configuration."""

    i2c_bus: int = field(default_factory=lambda: _int_env("ROBOT_SERVO_I2C_BUS", 2))
    address: int = field(default_factory=lambda: _int_env("ROBOT_SERVO_I2C_ADDRESS", 0x40))
    frequency_hz: int = 50
    pan_channel: int = 10
    tilt_channel: int = 9
    initial_pan: float = 70.0
    initial_tilt: float = 0.0
    pan_min: float = -90.0
    pan_max: float = 180.0
    tilt_min: float = -10.0
    tilt_max: float = 90.0
    min_count: int = 200
    max_count: int = 500


@dataclass
class SensorConfig:
    """Sensors and indicator pins.

    Button and LED defaults are None because the old main program used
    gpiozero BCM numbers from a Raspberry Pi-style script.  Set them only
    after confirming the Orange Pi GPIO line offsets.
    """

    ultrasonic_trig_pin: int = 100
    ultrasonic_echo_pin: int = 99
    ultrasonic_timeout_seconds: float = 0.03
    infrared_left_pin: int = 101
    infrared_right_pin: int = 35
    infrared_active_low: bool = True
    follow_button_pin: Optional[int] = field(
        default_factory=lambda: _optional_int_env("ROBOT_FOLLOW_BUTTON_PIN", None)
    )
    button_active_low: bool = True
    green_led_pin: Optional[int] = field(default_factory=lambda: _optional_int_env("ROBOT_GREEN_LED_PIN", None))
    red_led_pin: Optional[int] = field(default_factory=lambda: _optional_int_env("ROBOT_RED_LED_PIN", None))
    blue_led_pin: Optional[int] = field(default_factory=lambda: _optional_int_env("ROBOT_BLUE_LED_PIN", None))


@dataclass
class CameraConfig:
    backend: str = field(default_factory=lambda: os.getenv("ROBOT_CAMERA_BACKEND", "opencv"))
    device_index: int = field(default_factory=lambda: _int_env("ROBOT_CAMERA_DEVICE", 0))
    width: int = 320
    height: int = 240
    flip_vertical: bool = True
    flip_horizontal: bool = False
    jpeg_quality: int = 80


@dataclass
class FaceTrackingConfig:
    min_face_size: tuple[int, int] = (30, 30)
    scale_factor: float = 1.3
    min_neighbors: int = 5
    pan_deadband_px: float = 20.0
    tilt_deadband_px: float = 20.0
    pan_gain_divisor: float = 75.0
    tilt_gain_divisor: float = 75.0
    smoothing_factor: float = 0.8
    emotion_input_size: tuple[int, int] = (64, 64)
    emotion_labels: tuple[str, ...] = (
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    )


@dataclass
class AudioConfig:
    sample_format_name: str = "paInt16"
    channels: int = 1
    sample_rate: int = 48000
    chunk_size: int = 1024
    base_threshold: float = 10000.0
    record_seconds: float = 7.0
    silence_duration_ms: int = 500
    input_device: Optional[int] = field(default_factory=lambda: _optional_int_env("ROBOT_AUDIO_INPUT_DEVICE", None))
    output_device: Optional[int] = field(default_factory=lambda: _optional_int_env("ROBOT_AUDIO_OUTPUT_DEVICE", None))
    playback_sample_rate: int = 24000
    calibration_seconds: float = 2.0


@dataclass
class QwenConfig:
    env_file: Path = PROJECT_ROOT / ".env"
    fallback_env_file: Path = PROJECT_ROOT / "qwen.env"
    api_key_env_names: tuple[str, ...] = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY", "key")
    base_url: str = field(
        default_factory=lambda: os.getenv("ROBOT_QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    model: str = field(default_factory=lambda: os.getenv("ROBOT_QWEN_MODEL", "qwen2.5-omni-7b"))
    transcription_model: str = "qwen-audio-turbo-latest"
    voice: str = field(default_factory=lambda: os.getenv("ROBOT_QWEN_VOICE", "Chelsie"))
    system_prompt: str = "你是一个情感智能语音助手，擅长根据用户的情绪用中文进行自然、温和的沟通。"


@dataclass
class WebConfig:
    server_url: str = field(default_factory=lambda: os.getenv("ROBOT_WEB_SERVER_URL", "http://192.168.1.102:5000"))
    enabled: bool = True
    video_interval_seconds: float = 0.1
    user_id: int = 1


@dataclass
class FollowConfig:
    enabled_on_start: bool = field(default_factory=lambda: _bool_env("ROBOT_FOLLOW_ENABLED_ON_START", False))
    gimbal_assist_range: float = 50.0
    wheel_assist_error_threshold: float = 10.0
    face_width_far_threshold: float = 50.0
    face_width_close_threshold: float = 120.0
    centered_error_pan_threshold: float = 35.0
    turn_speed_assist: int = 20
    turn_speed_follow: int = 25
    forward_speed_approach: int = 28
    infrared_turn_speed: int = 30
    infrared_back_speed: int = 30
    ultrasonic_forward_obstacle_cm: float = 20.0
    approach_clearance_factor: float = 1.5
    control_interval_seconds: float = 0.05


@dataclass
class AppConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    motors: MotorPins = field(default_factory=MotorPins)
    servos: ServoConfig = field(default_factory=ServoConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_tracking: FaceTrackingConfig = field(default_factory=FaceTrackingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    web: WebConfig = field(default_factory=WebConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)


def load_config() -> AppConfig:
    """Load environment variables first, then return a typed config object."""

    for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "qwen.env"):
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file)
            except Exception:
                # dotenv is convenience only. The program can still read OS env.
                pass
    return AppConfig()
