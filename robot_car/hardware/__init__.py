"""硬件驱动层。

这一层封装 Orange Pi 直接连接的硬件，包括电机、舵机和传感器。应用层只调用
`RobotCar` 这样的高层对象，不直接接触 GPIO/I2C 细节。
"""

from .robot import RobotCar

__all__ = ["RobotCar"]
