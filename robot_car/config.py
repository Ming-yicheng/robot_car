"""智能小车集中配置。

旧项目把 GPIO 引脚、I2C 地址、模型路径、阈值、Web 地址和 API 配置分散在多个
脚本里。这样在换线、换舵机板地址、替换模型时很容易漏改。这个模块把所有可调
参数集中到一处，其他代码只读取配置对象，不直接写死硬件参数。

使用方式：
    from robot_car.config import load_config
    config = load_config()

环境变量优先级：
    1. 先加载项目根目录下的 `.env` 或 `qwen.env`
    2. 再用系统环境变量覆盖默认值
    3. 没配置时使用 dataclass 中的安全默认值
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool) -> bool:
    """读取布尔型环境变量。

    支持 1/true/yes/on 这类常见写法。没有配置时返回默认值。
    """

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量。

    `int(value, 0)` 支持十进制和十六进制，例如 `0x40`。
    """

    value = os.getenv(name)
    return default if value in (None, "") else int(value, 0)


def _optional_int_env(name: str, default: Optional[int]) -> Optional[int]:
    """读取可选整数环境变量。

    用于按键、LED 等可接可不接的硬件。如果环境变量为空，返回 None。
    """

    value = os.getenv(name)
    return default if value in (None, "") else int(value, 0)


def _float_env(name: str, default: float) -> float:
    """读取浮点型环境变量。"""

    value = os.getenv(name)
    return default if value in (None, "") else float(value)


@dataclass
class ProjectPaths:
    """项目目录和模型资源路径。

    这里的路径都以项目根目录为基准。模型文件不会上传到 GitHub，但运行时会从这些
    路径读取；如果文件不存在，相关功能会优雅降级或禁用。
    """

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
    """TB6612FNG 四电机接线。

    默认值来自旧项目 `robust_code/finish module/tb6612_abcd_test.py`。
    注意：`periphery.GPIO` 使用 Linux GPIO line offset，不是树莓派 BCM 编号，
    也不是物理针脚编号。改线时要先确认 Orange Pi 的 GPIO 映射。
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
    """PCA9685 舵机控制板配置。

    `i2c_bus` 和 `address` 是最容易因为硬件接线而变化的参数。运行前可用
    `ls /dev/i2c-*` 和 `i2cdetect -y 2` 检查实际总线和地址。
    """

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
    """传感器、按键和指示灯配置。

    红外和超声波使用旧项目中已经验证过的 GPIO line offset。按键和 LED 默认是
    None，因为旧主函数里使用的是 Raspberry Pi 风格的 gpiozero BCM 编号，不能
    直接套到 Orange Pi 上。确认线序后再通过环境变量配置。
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
    """摄像头配置。

    默认使用 OpenCV 打开 `/dev/video0`。如果未来换成 Picamera2，可通过
    `ROBOT_CAMERA_BACKEND=picamera2` 或命令行参数覆盖。
    """

    backend: str = field(default_factory=lambda: os.getenv("ROBOT_CAMERA_BACKEND", "opencv"))
    device_index: int = field(default_factory=lambda: _int_env("ROBOT_CAMERA_DEVICE", 0))
    width: int = 320
    height: int = 240
    flip_vertical: bool = True
    flip_horizontal: bool = False
    jpeg_quality: int = 80


@dataclass
class FaceTrackingConfig:
    """人脸追踪和云台控制参数。

    死区、增益和平滑系数共同决定云台是否抖动。死区越大越稳但响应慢；增益越大
    越灵敏但可能来回摆动；平滑系数越接近 1 越平滑。
    """

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
    """录音和播放参数。

    录音使用能量阈值进行简单语音活动检测。`calibration_seconds` 用于启动时测量
    环境噪声，并动态调整阈值，避免风扇声或环境声触发录音。
    """

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
    """Qwen / DashScope 云端语音对话配置。

    API Key 不写入代码，运行时从 `.env`、`qwen.env` 或系统环境变量读取。
    """

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
    """Web 前端/看板上传配置。

    通过 Socket.IO 上传识别结果和视频帧。Web 服务不可用时不会阻塞小车控制。
    """

    server_url: str = field(default_factory=lambda: os.getenv("ROBOT_WEB_SERVER_URL", "http://192.168.1.102:5000"))
    enabled: bool = True
    video_interval_seconds: float = 0.1
    user_id: int = 1


@dataclass
class FollowConfig:
    """底盘跟随和避障控制参数。

    参数基本来自旧主函数的 `wheel_control_loop`。这里保留原策略：红外避障优先，
    然后用云台角度辅助底盘转向，再根据人脸框宽度判断是否靠近或停止。
    """

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
    """完整应用配置对象。

    主程序只传递这个对象，避免函数签名里散落大量独立参数。
    """

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
    """加载环境变量并返回配置对象。

    `.env` 和 `qwen.env` 都是可选文件。没有这些文件时，程序仍然可以用默认配置
    运行基础视觉/硬件流程；需要云端语音功能时再补 API Key。
    """

    for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "qwen.env"):
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file)
            except Exception:
                # dotenv is convenience only. The program can still read OS env.
                pass
    return AppConfig()
