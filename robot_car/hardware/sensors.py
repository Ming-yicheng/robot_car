"""GPIO sensors and indicators."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from periphery import GPIO

from robot_car.config import SensorConfig

logger = logging.getLogger(__name__)


class GPIOInput:
    """Small active-low aware input wrapper."""

    def __init__(self, pin: int, *, active_low: bool = False):
        self.pin = pin
        self.active_low = active_low
        self.gpio = GPIO(pin, "in")

    @property
    def is_active(self) -> bool:
        value = bool(self.gpio.read())
        return not value if self.active_low else value

    def close(self) -> None:
        self.gpio.close()


class GPIOOutput:
    """Output wrapper used for status LEDs."""

    def __init__(self, pin: int):
        self.pin = pin
        self.gpio = GPIO(pin, "out")
        self.off()

    def on(self) -> None:
        self.gpio.write(True)

    def off(self) -> None:
        self.gpio.write(False)

    def close(self) -> None:
        self.off()
        self.gpio.close()


class NullLed:
    """No-op LED used when a pin is not configured."""

    def on(self) -> None:
        pass

    def off(self) -> None:
        pass

    def close(self) -> None:
        pass


class UltrasonicSensor:
    """HC-SR04-style ultrasonic distance sensor.

    Timeouts are important: the old script waited forever if ECHO never changed.
    A timeout returns None, letting the caller decide whether to stop or ignore.
    """

    def __init__(self, trig_pin: int, echo_pin: int, timeout_seconds: float = 0.03):
        self.trig = GPIO(trig_pin, "out")
        self.echo = GPIO(echo_pin, "in")
        self.timeout_seconds = timeout_seconds
        self.trig.write(False)

    def read_distance_cm(self) -> Optional[float]:
        self.trig.write(True)
        time.sleep(0.00001)
        self.trig.write(False)

        deadline = time.time() + self.timeout_seconds
        while not self.echo.read():
            if time.time() > deadline:
                return None
        pulse_start = time.time()

        deadline = pulse_start + self.timeout_seconds
        while self.echo.read():
            if time.time() > deadline:
                return None
        pulse_end = time.time()

        return round((pulse_end - pulse_start) * 17150.0, 2)

    def close(self) -> None:
        self.trig.close()
        self.echo.close()


@dataclass
class InfraredPair:
    left: GPIOInput
    right: GPIOInput

    @classmethod
    def from_config(cls, config: SensorConfig) -> "InfraredPair":
        return cls(
            GPIOInput(config.infrared_left_pin, active_low=config.infrared_active_low),
            GPIOInput(config.infrared_right_pin, active_low=config.infrared_active_low),
        )

    @property
    def left_triggered(self) -> bool:
        return self.left.is_active

    @property
    def right_triggered(self) -> bool:
        return self.right.is_active

    def close(self) -> None:
        self.left.close()
        self.right.close()


class FollowButton:
    """Optional physical button for enabling follow mode."""

    def __init__(self, pin: Optional[int], *, active_low: bool, default_enabled: bool):
        self.default_enabled = default_enabled
        self.input: Optional[GPIOInput] = None
        if pin is not None:
            self.input = GPIOInput(pin, active_low=active_low)

    @property
    def is_enabled(self) -> bool:
        return self.input.is_active if self.input is not None else self.default_enabled

    def close(self) -> None:
        if self.input is not None:
            self.input.close()


class LedIndicators:
    """Green/red/blue status LEDs; each pin is optional."""

    def __init__(self, green_pin: Optional[int], red_pin: Optional[int], blue_pin: Optional[int]):
        self.green = self._make_led(green_pin, "green")
        self.red = self._make_led(red_pin, "red")
        self.blue = self._make_led(blue_pin, "blue")

    @staticmethod
    def _make_led(pin: Optional[int], name: str):
        if pin is None:
            return NullLed()
        try:
            return GPIOOutput(pin)
        except Exception as exc:
            logger.warning("Failed to open %s LED on GPIO %s: %s", name, pin, exc)
            return NullLed()

    def set_idle(self) -> None:
        self.green.on()
        self.red.off()
        self.blue.off()

    def set_following(self) -> None:
        self.green.off()
        self.red.on()
        self.blue.off()

    def set_listening(self, listening: bool) -> None:
        if listening:
            self.blue.on()
        else:
            self.blue.off()

    def close(self) -> None:
        for led in (self.green, self.red, self.blue):
            led.close()
