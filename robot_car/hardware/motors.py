"""Orange Pi 上的 TB6612FNG 四轮底盘控制。

旧项目里“完成版电机测试脚本”已经解决了一个关键问题：如果四个电机分别进入
自己的阻塞式软件 PWM 循环，电机实际上会轮流得到 PWM，表现为明显卡顿。这里保留
“同步 PWM”的思想：在同一个时间循环里同时拉高四个 PWM 引脚，再按各自占空比依次
拉低，尽量保证多电机动作一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep, time
from typing import Iterable

from periphery import GPIO

from robot_car.config import MotorPins


def clamp_percent(speed: float) -> float:
    """把速度百分比限制在 0..100。"""

    return max(0.0, min(100.0, float(speed)))


def clamp_duty(duty: float) -> float:
    """把 PWM 占空比限制在 0..1。"""

    return max(0.0, min(1.0, float(duty)))


@dataclass
class MotorVector:
    """单个电机的有符号运动命令。

    正数表示前进，负数表示后退，0 表示停止。绝对值是 PWM 占空比。
    """

    duty: float

    @property
    def is_stopped(self) -> bool:
        return abs(self.duty) <= 1e-6

    @property
    def forward(self) -> bool:
        return self.duty >= 0

    @property
    def abs_duty(self) -> float:
        return clamp_duty(abs(self.duty))


class Tb6612Motor:
    """一个 TB6612 电机通道。

    每个电机有一个 PWM 引脚和两个方向引脚。这里不单独提供长时间运行方法，
    避免单电机阻塞影响多电机同步。
    """

    def __init__(self, pwm_pin: int, in1_pin: int, in2_pin: int, name: str):
        self.name = name
        self.pwm = GPIO(pwm_pin, "out")
        self.in1 = GPIO(in1_pin, "out")
        self.in2 = GPIO(in2_pin, "out")
        self.stop()

    def set_direction(self, forward: bool) -> None:
        """设置电机方向。"""
        self.in1.write(bool(forward))
        self.in2.write(not bool(forward))

    def stop(self) -> None:
        """关闭 PWM 和方向脚，让电机停止。"""
        self.pwm.write(False)
        self.in1.write(False)
        self.in2.write(False)

    def close(self) -> None:
        self.stop()
        self.pwm.close()
        self.in1.close()
        self.in2.close()


class FourWheelDrive:
    """四轮底盘控制器，并保留旧 LOBOROBOT 风格方法名。

    上层应用仍然使用百分比速度，内部会转换成 0..1 占空比。保留 `t_up`、
    `turnLeft` 等旧方法名，是为了让从旧主函数迁移来的逻辑更容易对应。
    """

    def __init__(
        self,
        motors: Iterable[Tb6612Motor],
        *,
        pwm_frequency_hz: float = 50.0,
        control_slice_seconds: float = 0.05,
    ):
        self.motors = list(motors)
        if len(self.motors) != 4:
            raise ValueError("FourWheelDrive requires exactly four motors")
        self.pwm_frequency_hz = pwm_frequency_hz
        self.control_slice_seconds = control_slice_seconds

    @classmethod
    def from_config(cls, pins: MotorPins) -> "FourWheelDrive":
        motors = [
            Tb6612Motor(pins.pwma, pins.ain1, pins.ain2, "motor A"),
            Tb6612Motor(pins.pwmb, pins.bin1, pins.bin2, "motor B"),
            Tb6612Motor(pins.pwmc, pins.cin1, pins.cin2, "motor C"),
            Tb6612Motor(pins.pwmd, pins.din1, pins.din2, "motor D"),
        ]
        return cls(
            motors,
            pwm_frequency_hz=pins.pwm_frequency_hz,
            control_slice_seconds=pins.control_slice_seconds,
        )

    def run_vectors(self, vectors: Iterable[float], duration: float | None = None) -> None:
        """按四个有符号占空比运行底盘。

        `vectors` 顺序对应 A/B/C/D 四个电机。duration 为 0 或 None 时，只运行一个
        控制切片，适合主循环不断刷新命令；duration 大于 0 时会阻塞运行指定秒数。
        """

        commands = [MotorVector(duty) for duty in vectors]
        if len(commands) != 4:
            raise ValueError("expected four motor duty values")

        active = [(motor, cmd) for motor, cmd in zip(self.motors, commands) if not cmd.is_stopped]
        if not active:
            self.stop()
            return

        for motor, cmd in active:
            motor.set_direction(cmd.forward)

        run_seconds = self.control_slice_seconds if not duration or duration <= 0 else duration
        self._synchronous_pwm(active, run_seconds)

    def _synchronous_pwm(self, active: list[tuple[Tb6612Motor, MotorVector]], duration: float) -> None:
        """同步软件 PWM。

        先同时拉高所有需要工作的 PWM 引脚，再按照各自高电平时间排序拉低。
        这样四个电机在一个 PWM 周期内共享同一个时间基准。
        """

        period = 1.0 / self.pwm_frequency_hz
        end_time = time() + max(0.0, duration)

        while time() < end_time:
            cycle_start = time()
            for motor, command in active:
                motor.pwm.write(command.abs_duty > 0)

            off_schedule = sorted(
                (cycle_start + period * command.abs_duty, motor)
                for motor, command in active
                if command.abs_duty < 1.0
            )
            for off_time, motor in off_schedule:
                delay = off_time - time()
                if delay > 0:
                    sleep(delay)
                motor.pwm.write(False)

            remaining = period - (time() - cycle_start)
            if remaining > 0:
                sleep(remaining)

        for motor, _ in active:
            motor.pwm.write(False)

    def _speed_to_duty(self, speed_percent: float) -> float:
        """把百分比速度转换为占空比。"""
        return clamp_percent(speed_percent) / 100.0

    def forward(self, speed: float, duration: float = 0.0) -> None:
        duty = self._speed_to_duty(speed)
        self.run_vectors([duty, duty, duty, duty], duration)

    def backward(self, speed: float, duration: float = 0.0) -> None:
        duty = self._speed_to_duty(speed)
        self.run_vectors([-duty, -duty, -duty, -duty], duration)

    def strafe_left(self, speed: float, duration: float = 0.0) -> None:
        duty = self._speed_to_duty(speed)
        self.run_vectors([-duty, duty, duty, -duty], duration)

    def strafe_right(self, speed: float, duration: float = 0.0) -> None:
        duty = self._speed_to_duty(speed)
        self.run_vectors([duty, -duty, -duty, duty], duration)

    def turn_left(self, speed: float, duration: float = 0.0) -> None:
        duty = self._speed_to_duty(speed)
        self.run_vectors([-duty, duty, -duty, duty], duration)

    def turn_right(self, speed: float, duration: float = 0.0) -> None:
        duty = self._speed_to_duty(speed)
        self.run_vectors([duty, -duty, duty, -duty], duration)

    def stop(self, duration: float = 0.0) -> None:
        for motor in self.motors:
            motor.stop()
        if duration and duration > 0:
            sleep(duration)

    def close(self) -> None:
        for motor in self.motors:
            motor.close()

    # Legacy names kept so old main-function logic maps cleanly.
    def t_up(self, speed: float, t_time: float = 0.0) -> None:
        self.forward(speed, t_time)

    def t_down(self, speed: float, t_time: float = 0.0) -> None:
        self.backward(speed, t_time)

    def moveLeft(self, speed: float, t_time: float = 0.0) -> None:
        self.strafe_left(speed, t_time)

    def moveRight(self, speed: float, t_time: float = 0.0) -> None:
        self.strafe_right(speed, t_time)

    def turnLeft(self, speed: float, t_time: float = 0.0) -> None:
        self.turn_left(speed, t_time)

    def turnRight(self, speed: float, t_time: float = 0.0) -> None:
        self.turn_right(speed, t_time)

    def t_stop(self, t_time: float = 0.0) -> None:
        self.stop(t_time)
