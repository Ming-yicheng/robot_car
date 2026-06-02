"""GPIO 传感器和指示灯封装。

这里统一使用 `periphery.GPIO`，避免旧主函数中 gpiozero BCM 编号和 Orange Pi
GPIO line offset 混用的问题。所有输入都支持 active_low，用于低电平触发模块。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from periphery import GPIO

from robot_car.config import SensorConfig

logger = logging.getLogger(__name__)


class GPIOInput:
    """支持 active-low 的输入引脚包装类。"""

    def __init__(self, pin: int, *, active_low: bool = False):
        self.pin = pin
        self.active_low = active_low
        self.gpio = GPIO(pin, "in")

    @property
    def is_active(self) -> bool:
        """返回“是否触发”的逻辑状态，而不是原始电平。"""
        value = bool(self.gpio.read())
        return not value if self.active_low else value

    def close(self) -> None:
        self.gpio.close()


class GPIOOutput:
    """输出引脚包装类，主要用于状态 LED。"""

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
    """空 LED。

    当 LED 引脚没有配置或打开失败时使用，调用 on/off 不会产生任何硬件动作。
    这样主程序不需要到处判断 LED 是否存在。
    """

    def on(self) -> None:
        pass

    def off(self) -> None:
        pass

    def close(self) -> None:
        pass


class UltrasonicSensor:
    """HC-SR04 风格超声波测距模块。

    旧脚本在等待 ECHO 电平变化时没有超时保护，如果传感器断线或未响应，会一直卡住。
    这里加入超时：超时返回 None，由底盘控制策略决定停止或忽略。
    """

    def __init__(self, trig_pin: int, echo_pin: int, timeout_seconds: float = 0.03):
        self.trig = GPIO(trig_pin, "out")
        self.echo = GPIO(echo_pin, "in")
        self.timeout_seconds = timeout_seconds
        self.trig.write(False)

    def read_distance_cm(self) -> Optional[float]:
        """测量距离，单位厘米。超时或异常响应时返回 None。"""
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
    """左右两个红外避障传感器。"""

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
    """可选的跟随模式按键。

    如果没有配置实体按键，则使用 `default_enabled`。因此调试时可以直接通过
    `--follow` 启用跟随模式。
    """

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
    """绿/红/蓝状态 LED 组合。

    绿灯表示空闲，红灯表示跟随/运动，蓝灯表示正在听或处理语音。每个引脚都可以
    不配置。
    """

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
