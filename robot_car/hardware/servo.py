"""PCA9685 舵机控制器。

PCA9685 是 16 路 PWM 控制板，常用于云台舵机。它通过 I2C 控制，输出的是 12 位
PWM 计数值。上层代码使用角度，底层再转换为计数值。
"""

from __future__ import annotations

from time import sleep

try:
    from smbus2 import SMBus
except ImportError:  # pragma: no cover - fallback for older images
    from smbus import SMBus


MODE1 = 0x00
MODE2 = 0x01
PRESCALE = 0xFE
LED0_ON_L = 0x06
RESTART = 0x80
AI = 0x20
SLEEP = 0x10
ALLCALL = 0x01
OUTDRV = 0x04


class ServoController:
    """通过 PCA9685 控制舵机。

    `set_angle()` 接收 0..180 度，默认映射到 200..500 的 PCA9685 计数范围。
    不同舵机的极限脉宽不完全一致，如果出现嗡鸣或卡死，应缩小 min/max count。
    """

    def __init__(self, bus_num: int = 2, address: int = 0x40, frequency: int = 50):
        self.bus_num = bus_num
        self.address = address
        self.frequency = frequency
        self.bus = SMBus(bus_num)
        self._init_board()

    def _write8(self, reg: int, value: int) -> None:
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def _read8(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def _init_board(self) -> None:
        """初始化 PCA9685 工作模式和 PWM 频率。"""
        self._write8(MODE1, ALLCALL | AI)
        self._write8(MODE2, OUTDRV)
        self.set_frequency(self.frequency)

    def set_frequency(self, frequency: int) -> None:
        """设置 PWM 频率。普通模拟舵机通常使用 50Hz。"""
        prescale = int(25_000_000.0 / 4096.0 / frequency - 1.0 + 0.5)
        old_mode = self._read8(MODE1)
        self._write8(MODE1, (old_mode & 0x7F) | SLEEP)
        self._write8(PRESCALE, prescale)
        self._write8(MODE1, old_mode)
        sleep(0.005)
        self._write8(MODE1, old_mode | RESTART | AI)
        self.frequency = frequency

    def set_count(self, channel: int, count: int) -> None:
        """直接设置某一路 PWM 的 12 位计数值。"""
        self._check_channel(channel)
        count = self._clamp(int(count), 0, 4095)
        self.bus.write_i2c_block_data(
            self.address,
            LED0_ON_L + 4 * channel,
            [0x00, 0x00, count & 0xFF, (count >> 8) & 0x0F],
        )

    def set_angle(self, channel: int, angle: float, min_count: int = 200, max_count: int = 500) -> int:
        """按角度控制舵机，并返回实际写入的 count。"""
        angle = self._clamp(float(angle), 0.0, 180.0)
        count = min_count + (max_count - min_count) * angle / 180.0
        rounded = round(count)
        self.set_count(channel, rounded)
        return rounded

    def release(self, channel: int) -> None:
        """释放某一路 PWM 输出。"""
        self._check_channel(channel)
        self.bus.write_i2c_block_data(self.address, LED0_ON_L + 4 * channel, [0x00, 0x00, 0x00, 0x10])

    def release_all(self) -> None:
        for channel in range(16):
            self.release(channel)

    def close(self) -> None:
        self.bus.close()

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _check_channel(channel: int) -> None:
        if not 0 <= channel <= 15:
            raise ValueError("channel must be 0..15")
