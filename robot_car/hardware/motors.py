"""TB6612FNG four-wheel motor control for Orange Pi.

The original finished motor script already solved the important problem:
when four motors use software PWM, all PWM pins must be toggled in one timing
loop.  If each motor runs its own blocking PWM loop, the motors visibly stutter.
This module keeps that synchronous PWM approach and exposes clearer movement
methods for the application layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep, time
from typing import Iterable

from periphery import GPIO

from robot_car.config import MotorPins


def clamp_percent(speed: float) -> float:
    """Clamp a 0..100 speed percentage."""

    return max(0.0, min(100.0, float(speed)))


def clamp_duty(duty: float) -> float:
    """Clamp a 0..1 PWM duty value."""

    return max(0.0, min(1.0, float(duty)))


@dataclass
class MotorVector:
    """Signed motor command.

    Positive duty means forward, negative duty means backward, and zero means
    stopped.  Magnitude is in 0..1 duty units.
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
    """One TB6612 motor channel."""

    def __init__(self, pwm_pin: int, in1_pin: int, in2_pin: int, name: str):
        self.name = name
        self.pwm = GPIO(pwm_pin, "out")
        self.in1 = GPIO(in1_pin, "out")
        self.in2 = GPIO(in2_pin, "out")
        self.stop()

    def set_direction(self, forward: bool) -> None:
        self.in1.write(bool(forward))
        self.in2.write(not bool(forward))

    def stop(self) -> None:
        self.pwm.write(False)
        self.in1.write(False)
        self.in2.write(False)

    def close(self) -> None:
        self.stop()
        self.pwm.close()
        self.in1.close()
        self.in2.close()


class FourWheelDrive:
    """Four-motor chassis with legacy movement aliases.

    The application uses percentage speeds because the old LOBOROBOT class did.
    Internally the software PWM loop converts them to 0..1 duty values.
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
        """Run signed motor duties for one slice or a fixed duration."""

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
        """Toggle all active PWM pins in the same timing loop."""

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
