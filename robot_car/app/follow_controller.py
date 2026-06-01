"""Wheel follow and obstacle-avoidance strategy.

This is the wheel_control_loop from the uploaded main function, rewritten as a
class with explicit dependencies.  The behavior is intentionally close to the
original: infrared avoidance has priority, then gimbal-assisted face following,
then approach/hold decisions based on face width.
"""

from __future__ import annotations

import logging
import math
import threading
import time

from robot_car.config import AppConfig
from robot_car.hardware.robot import RobotCar
from robot_car.hardware.sensors import FollowButton, InfraredPair, LedIndicators, UltrasonicSensor
from robot_car.state import RobotState

logger = logging.getLogger(__name__)


class FollowController:
    def __init__(
        self,
        config: AppConfig,
        state: RobotState,
        robot: RobotCar,
        infrared: InfraredPair,
        ultrasonic: UltrasonicSensor,
        follow_button: FollowButton,
        leds: LedIndicators,
    ):
        self.config = config
        self.state = state
        self.robot = robot
        self.infrared = infrared
        self.ultrasonic = ultrasonic
        self.follow_button = follow_button
        self.leds = leds
        self._last_action = None

    def run(self, stop_event: threading.Event) -> None:
        logger.info("Wheel follow loop started")
        try:
            while not stop_event.is_set():
                self.step()
                time.sleep(self.config.follow.control_interval_seconds)
        finally:
            self.robot.t_stop(0.1)
            self.leds.set_idle()
            logger.info("Wheel follow loop stopped")

    def step(self) -> None:
        follow_enabled = self.follow_button.is_enabled
        if follow_enabled:
            self.leds.set_following()
        else:
            self.robot.t_stop(0.05)
            self.leds.set_idle()
            return

        snapshot = self.state.snapshot()
        speed_factor = 0.5 if (snapshot["listening_for_voice"] or snapshot["speaking"]) else 1.0
        left_ir = self.infrared.left_triggered
        right_ir = self.infrared.right_triggered
        distance_cm = self.ultrasonic.read_distance_cm()
        if distance_cm is None:
            distance_cm = math.inf

        if self._handle_infrared(left_ir, right_ir, speed_factor):
            return
        if snapshot["face_detected"]:
            if self._handle_face_tracking(snapshot, distance_cm, speed_factor):
                return

        self._act("no_face_stop", "未检测到可跟随人脸，停止。", lambda: self.robot.t_stop(0.05))

    def _handle_infrared(self, left_ir: bool, right_ir: bool, speed_factor: float) -> bool:
        cfg = self.config.follow
        if left_ir and right_ir:
            self._act(
                "ir_both_back",
                "红外避障：左右两侧都有障碍物，后退。",
                lambda: self.robot.t_down(int(cfg.infrared_back_speed * speed_factor), 0),
            )
            return True
        if left_ir:
            self._act(
                "ir_left_turn_left",
                "红外避障：左侧触发，按原策略左转。",
                lambda: self.robot.turnLeft(int(cfg.infrared_turn_speed * speed_factor), 0),
            )
            return True
        if right_ir:
            self._act(
                "ir_right_turn_right",
                "红外避障：右侧触发，按原策略右转。",
                lambda: self.robot.turnRight(int(cfg.infrared_turn_speed * speed_factor), 0),
            )
            return True
        return False

    def _handle_face_tracking(self, snapshot: dict, distance_cm: float, speed_factor: float) -> bool:
        cfg = self.config.follow
        pan = snapshot["pan"]
        error_pan = snapshot["face_error_pan"]
        face_width = snapshot["face_width"]
        assist_low = self.config.servos.initial_pan - cfg.gimbal_assist_range
        assist_high = self.config.servos.initial_pan + cfg.gimbal_assist_range

        if pan > assist_high and error_pan > cfg.wheel_assist_error_threshold:
            if distance_cm >= cfg.ultrasonic_forward_obstacle_cm:
                self._act(
                    "assist_turn_left",
                    "轮子辅助云台居中：目标在右侧，底盘左转。",
                    lambda: self.robot.turnLeft(int(cfg.turn_speed_assist * speed_factor), 0),
                )
            else:
                self._act("assist_blocked_stop", "轮子辅助转向被超声波障碍物阻挡，停止。", lambda: self.robot.t_stop(0.05))
            return True

        if pan < assist_low and error_pan < -cfg.wheel_assist_error_threshold:
            if distance_cm >= cfg.ultrasonic_forward_obstacle_cm:
                self._act(
                    "assist_turn_right",
                    "轮子辅助云台居中：目标在左侧，底盘右转。",
                    lambda: self.robot.turnRight(int(cfg.turn_speed_assist * speed_factor), 0),
                )
            else:
                self._act("assist_blocked_stop", "轮子辅助转向被超声波障碍物阻挡，停止。", lambda: self.robot.t_stop(0.05))
            return True

        if 0 < face_width < cfg.face_width_far_threshold:
            return self._approach_far_face(error_pan, distance_cm, speed_factor)

        return self._hold_or_turn(error_pan, face_width, speed_factor)

    def _approach_far_face(self, error_pan: float, distance_cm: float, speed_factor: float) -> bool:
        cfg = self.config.follow
        if abs(error_pan) < cfg.centered_error_pan_threshold:
            clear_distance = cfg.ultrasonic_forward_obstacle_cm * cfg.approach_clearance_factor
            if distance_cm >= clear_distance:
                self._act(
                    "approach_forward",
                    "人脸较远且已居中，前方路径清晰，前进靠近。",
                    lambda: self.robot.t_up(int(cfg.forward_speed_approach * speed_factor), 0),
                )
            else:
                self._act("approach_blocked_stop", "人脸较远但前方距离不足，停止。", lambda: self.robot.t_stop(0.05))
            return True

        if error_pan < -cfg.centered_error_pan_threshold:
            self._act(
                "far_face_turn_right",
                "人脸较远且在左侧，底盘右转。",
                lambda: self.robot.turnRight(int(cfg.turn_speed_follow * speed_factor), 0),
            )
        else:
            self._act(
                "far_face_turn_left",
                "人脸较远且在右侧，底盘左转。",
                lambda: self.robot.turnLeft(int(cfg.turn_speed_follow * speed_factor), 0),
            )
        return True

    def _hold_or_turn(self, error_pan: float, face_width: float, speed_factor: float) -> bool:
        cfg = self.config.follow
        if abs(error_pan) > cfg.centered_error_pan_threshold:
            if error_pan < 0:
                self._act(
                    "follow_turn_right",
                    f"人脸未居中，位于左侧，底盘右转。宽度={face_width:.0f}",
                    lambda: self.robot.turnRight(int(cfg.turn_speed_follow * speed_factor), 0),
                )
            else:
                self._act(
                    "follow_turn_left",
                    f"人脸未居中，位于右侧，底盘左转。宽度={face_width:.0f}",
                    lambda: self.robot.turnLeft(int(cfg.turn_speed_follow * speed_factor), 0),
                )
            return True

        self._act("face_centered_stop", f"人脸已居中，保持当前位置。宽度={face_width:.0f}", lambda: self.robot.t_stop(0.05))
        return True

    def _act(self, action_id: str, message: str, command) -> None:
        if action_id != self._last_action:
            logger.info(message)
            self._last_action = action_id
        command()
