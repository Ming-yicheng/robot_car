"""底盘跟随和避障策略。

这个类来自用户提供的 `main_decoupling_web_sequence.py` 中的 `wheel_control_loop`，
但改成了显式依赖注入：需要的状态、传感器、机器人对象都从构造函数传入。

控制优先级保持和旧主函数一致：
    1. 红外避障最高优先级，检测到近距离障碍立即转向或后退
    2. 云台角度偏离过大时，让底盘辅助转向，避免云台转到极限
    3. 根据人脸框宽度判断距离：太远则靠近，适中或太近则停止
    4. 没有人脸或没有匹配动作时，底盘停止
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
    """把传感器和视觉状态转换成底盘动作。"""

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
        """持续运行底盘控制循环，直到收到停止事件。"""
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
        """执行一次控制决策。

        主循环每 50ms 左右调用一次。这里不做长时间阻塞，避免底盘响应迟钝。
        """
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
        """处理红外避障。

        返回 True 表示本轮已经采取动作，后续人脸跟随逻辑不再执行。
        """
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
        """处理人脸跟随逻辑。

        `snapshot` 是线程安全状态快照，避免在一次决策过程中被视频线程改动。
        """
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
        """人脸较远时的靠近策略。"""
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
        """人脸距离适中或较近时，执行居中转向或停止保持。"""
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
        """执行动作并限制重复日志。

        底盘循环频率较高，如果每次都打印会刷屏；只有动作类型变化时才记录日志。
        """
        if action_id != self._last_action:
            logger.info(message)
            self._last_action = action_id
        command()
