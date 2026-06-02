"""应用层使用的高层机器人对象。

`RobotCar` 把底盘和云台组合成一个对象，让主程序只关心“前进、后退、转向、设置舵机”
这些语义动作，不直接操作 GPIO、I2C 或 PWM。
"""

from __future__ import annotations

import logging
from time import sleep

from robot_car.config import AppConfig
from robot_car.hardware.motors import FourWheelDrive
from robot_car.hardware.servo import ServoController

logger = logging.getLogger(__name__)


class RobotCar:
    """底盘和云台的组合门面。

    这里刻意保留 `t_up`、`t_down`、`turnLeft` 等旧方法名，方便旧主函数中的控制
    逻辑迁移。后续如果想统一成 snake_case，可以在应用层逐步替换。
    """

    def __init__(self, wheels: FourWheelDrive, servos: ServoController, config: AppConfig):
        self.wheels = wheels
        self.servos = servos
        self.config = config

    @classmethod
    def from_config(cls, config: AppConfig) -> "RobotCar":
        wheels = FourWheelDrive.from_config(config.motors)
        servos = ServoController(
            bus_num=config.servos.i2c_bus,
            address=config.servos.address,
            frequency=config.servos.frequency_hz,
        )
        return cls(wheels, servos, config)

    def initialize_gimbal(self) -> None:
        """把云台移动到初始角度。"""
        self.set_servo_angle(self.config.servos.pan_channel, self.config.servos.initial_pan)
        self.set_servo_angle(self.config.servos.tilt_channel, self.config.servos.initial_tilt)
        sleep(0.2)

    def set_servo_angle(self, channel: int, angle: float) -> int:
        """设置某个舵机通道的角度。"""
        return self.servos.set_angle(
            channel,
            angle,
            min_count=self.config.servos.min_count,
            max_count=self.config.servos.max_count,
        )

    def t_up(self, speed: float, t_time: float = 0.0) -> None:
        self.wheels.t_up(speed, t_time)

    def t_down(self, speed: float, t_time: float = 0.0) -> None:
        self.wheels.t_down(speed, t_time)

    def moveLeft(self, speed: float, t_time: float = 0.0) -> None:
        self.wheels.moveLeft(speed, t_time)

    def moveRight(self, speed: float, t_time: float = 0.0) -> None:
        self.wheels.moveRight(speed, t_time)

    def turnLeft(self, speed: float, t_time: float = 0.0) -> None:
        self.wheels.turnLeft(speed, t_time)

    def turnRight(self, speed: float, t_time: float = 0.0) -> None:
        self.wheels.turnRight(speed, t_time)

    def t_stop(self, t_time: float = 0.0) -> None:
        self.wheels.t_stop(t_time)

    def close(self) -> None:
        """停止底盘并释放硬件资源。"""
        logger.info("Stopping robot and releasing hardware")
        try:
            self.t_stop(0.1)
        finally:
            self.wheels.close()
            self.servos.close()
