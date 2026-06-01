"""PCA9685 servo controller."""

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
    """Control servos through a PCA9685 board.

    The public method accepts degrees because the app thinks in pan/tilt
    angles.  The board itself receives 12-bit PWM counts.
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
        self._write8(MODE1, ALLCALL | AI)
        self._write8(MODE2, OUTDRV)
        self.set_frequency(self.frequency)

    def set_frequency(self, frequency: int) -> None:
        prescale = int(25_000_000.0 / 4096.0 / frequency - 1.0 + 0.5)
        old_mode = self._read8(MODE1)
        self._write8(MODE1, (old_mode & 0x7F) | SLEEP)
        self._write8(PRESCALE, prescale)
        self._write8(MODE1, old_mode)
        sleep(0.005)
        self._write8(MODE1, old_mode | RESTART | AI)
        self.frequency = frequency

    def set_count(self, channel: int, count: int) -> None:
        self._check_channel(channel)
        count = self._clamp(int(count), 0, 4095)
        self.bus.write_i2c_block_data(
            self.address,
            LED0_ON_L + 4 * channel,
            [0x00, 0x00, count & 0xFF, (count >> 8) & 0x0F],
        )

    def set_angle(self, channel: int, angle: float, min_count: int = 200, max_count: int = 500) -> int:
        angle = self._clamp(float(angle), 0.0, 180.0)
        count = min_count + (max_count - min_count) * angle / 180.0
        rounded = round(count)
        self.set_count(channel, rounded)
        return rounded

    def release(self, channel: int) -> None:
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
