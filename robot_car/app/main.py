"""重构后智能小车主入口。

主程序负责把各个模块接起来：
    - 视频线程：读取摄像头，检测人脸，控制云台，并上传视频帧
    - 语音线程：录音、情绪识别、Qwen 回复、上传对话结果
    - 底盘循环：读取红外/超声波/人脸状态，控制小车移动

注意：这个文件只负责流程编排，不直接写 GPIO 或模型推理细节。
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

import cv2

from robot_car.app.follow_controller import FollowController
from robot_car.audio.qwen_voice import QwenVoiceClient
from robot_car.audio.recorder import VoiceRecorder
from robot_car.audio.speech_emotion import maybe_create_speech_emotion
from robot_car.config import load_config
from robot_car.hardware.robot import RobotCar
from robot_car.hardware.sensors import FollowButton, InfraredPair, LedIndicators, UltrasonicSensor
from robot_car.state import RobotState
from robot_car.utils.logging import configure_logging
from robot_car.vision.camera import create_camera
from robot_car.vision.face_tracker import FaceTracker
from robot_car.web.telemetry import TelemetryClient

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这些参数用于调试不同子系统，例如只跑视觉和底盘，或在没有实体按键时直接启用
    跟随模式。
    """

    parser = argparse.ArgumentParser(description="Run the Orange Pi robot car.")
    parser.add_argument("--follow", action="store_true", help="Enable follow mode without a physical button.")
    parser.add_argument("--no-follow", action="store_true", help="Disable wheel follow loop.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice recording and Qwen reply.")
    parser.add_argument("--no-web", action="store_true", help="Disable Socket.IO upload.")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip microphone noise calibration.")
    parser.add_argument("--camera-backend", choices=["opencv", "picamera2"], help="Override camera backend.")
    return parser.parse_args()


def draw_status(frame, snapshot: dict) -> None:
    """在视频帧左上角叠加当前语音状态。"""

    status = []
    if snapshot["voice_active"]:
        status.append("Voice: ON")
    if snapshot["listening_for_voice"]:
        status.append("Listening")
    if snapshot["speaking"]:
        status.append("Speaking")
    for index, text in enumerate(status):
        cv2.putText(frame, text, (10, 20 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def run_video_loop(stop_event: threading.Event, config, state: RobotState, robot: RobotCar, telemetry: TelemetryClient) -> None:
    """视频线程入口。

    该线程持续读取摄像头画面，交给 FaceTracker 处理。FaceTracker 会更新共享状态
    并控制云台；这里再负责按频率上传 JPEG 帧。
    """

    camera = create_camera(config.camera)
    tracker = FaceTracker(config, state, robot)
    last_sent_at = 0.0
    logger.info("Video loop started")
    try:
        while not stop_event.is_set():
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            annotated = tracker.process_frame(frame)
            draw_status(annotated, state.snapshot())
            now = time.time()
            if now - last_sent_at >= config.web.video_interval_seconds:
                telemetry.send_video_frame(annotated, jpeg_quality=config.camera.jpeg_quality)
                last_sent_at = now
            time.sleep(0.02)
    except Exception:
        logger.exception("Video loop stopped by error")
        stop_event.set()
    finally:
        camera.close()
        logger.info("Video loop stopped")


def blink_blue_led(stop_event: threading.Event, leds: LedIndicators) -> None:
    """语音处理期间闪烁蓝灯，提示用户系统正在工作。"""

    while not stop_event.is_set():
        leds.blue.on()
        time.sleep(0.15)
        leds.blue.off()
        time.sleep(0.15)
    leds.blue.off()


def run_voice_loop(
    stop_event: threading.Event,
    config,
    state: RobotState,
    recorder: VoiceRecorder,
    speech_emotion,
    voice_client: QwenVoiceClient,
    telemetry: TelemetryClient,
    leds: LedIndicators,
) -> None:
    """语音线程入口。

    一次语音回合包括：录音、可选语音情绪识别、可选云端文本情绪提取、Qwen 音频
    回复、Web 上传。任何一步失败都只记录日志，不让主程序退出。
    """

    logger.info("Voice loop started")

    def handle_listening_state(active: bool) -> None:
        """同步录音状态和蓝色指示灯。

        单独写成函数而不是 lambda，是为了让类型检查器知道这里没有返回值。
        """

        state.set_voice_flags(listening=active)
        leds.set_listening(active)

    while not stop_event.is_set():
        snapshot = state.snapshot()
        if not snapshot["voice_active"] or snapshot["listening_for_voice"] or snapshot["speaking"]:
            time.sleep(0.1)
            continue

        audio_path = recorder.record_once(on_listening=handle_listening_state)
        if audio_path is None:
            time.sleep(0.2)
            continue

        blink_stop = threading.Event()
        blink_thread = threading.Thread(target=blink_blue_led, args=(blink_stop, leds), daemon=True)
        blink_thread.start()
        try:
            if speech_emotion is not None:
                label, confidence = speech_emotion.predict_file(audio_path)
                state.update_speech_emotion(label, confidence)

            text_emotion = voice_client.transcribe_text_emotion(audio_path)
            state.update_text_emotion(text_emotion.text, text_emotion.emotion, text_emotion.confidence)

            state.set_voice_flags(speaking=True)
            current = state.snapshot()
            reply = voice_client.chat_with_audio(
                audio_path,
                facial_emotion=current["facial_emotion_label"],
                speech_emotion=current["speech_emotion_label"],
            )
            state.update_output_text(reply.text)
            telemetry.send_robot_data(state)
        except Exception:
            logger.exception("Voice turn failed")
        finally:
            state.set_voice_flags(speaking=False)
            blink_stop.set()
            blink_thread.join(timeout=2)
        time.sleep(0.5)
    logger.info("Voice loop stopped")


def main() -> None:
    """程序主入口。

    这里负责创建配置、硬件对象、共享状态和各工作线程，并在退出时统一释放资源。
    """

    args = parse_args()
    config = load_config()
    if args.camera_backend:
        config.camera.backend = args.camera_backend
    if args.no_web:
        config.web.enabled = False
    if args.follow:
        config.follow.enabled_on_start = True
    if args.no_follow:
        config.follow.enabled_on_start = False

    configure_logging(config.paths.root / "logs")
    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):
        """SIGINT/SIGTERM 信号处理函数。"""
        logger.info("Stop requested")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    state = RobotState(pan=config.servos.initial_pan, tilt=config.servos.initial_tilt)
    telemetry = TelemetryClient(config.web)
    robot = RobotCar.from_config(config)
    infrared = InfraredPair.from_config(config.sensors)
    ultrasonic = UltrasonicSensor(
        config.sensors.ultrasonic_trig_pin,
        config.sensors.ultrasonic_echo_pin,
        config.sensors.ultrasonic_timeout_seconds,
    )
    follow_button = FollowButton(
        config.sensors.follow_button_pin,
        active_low=config.sensors.button_active_low,
        default_enabled=config.follow.enabled_on_start,
    )
    leds = LedIndicators(config.sensors.green_led_pin, config.sensors.red_led_pin, config.sensors.blue_led_pin)

    video_thread = None
    voice_thread = None
    try:
        telemetry.connect()
        leds.set_idle()
        robot.initialize_gimbal()

        video_thread = threading.Thread(
            target=run_video_loop,
            args=(stop_event, config, state, robot, telemetry),
            name="VideoLoop",
            daemon=True,
        )
        video_thread.start()

        if not args.no_voice:
            recorder = VoiceRecorder(config.audio, config.paths.output_wav_path)
            if not args.skip_calibration:
                recorder.calibrate_noise()
            speech_emotion = maybe_create_speech_emotion(config.paths.speech_emotion_model_path)
            voice_client = QwenVoiceClient(config.qwen, config.audio, config.paths)
            voice_thread = threading.Thread(
                target=run_voice_loop,
                args=(stop_event, config, state, recorder, speech_emotion, voice_client, telemetry, leds),
                name="VoiceLoop",
                daemon=True,
            )
            voice_thread.start()
        else:
            state.set_voice_flags(active=False)

        if args.no_follow:
            logger.info("Follow loop disabled; waiting until stopped")
            while not stop_event.is_set():
                time.sleep(0.5)
        else:
            controller = FollowController(config, state, robot, infrared, ultrasonic, follow_button, leds)
            controller.run(stop_event)
    finally:
        stop_event.set()
        if video_thread:
            video_thread.join(timeout=3)
        if voice_thread:
            voice_thread.join(timeout=3)
        telemetry.close()
        for device in (infrared, ultrasonic, follow_button, leds):
            try:
                device.close()
            except Exception:
                logger.exception("Failed to close %s", device)
        robot.close()
        logger.info("Program terminated")


if __name__ == "__main__":
    main()
