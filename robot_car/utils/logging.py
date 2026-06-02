"""日志配置工具。

主程序启动时调用 `configure_logging()`，同时把日志输出到终端和 `logs/robot_car.log`。
日志目录已被 `.gitignore` 忽略，避免运行日志进入仓库。
"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path | None = None) -> None:
    """配置全局日志。

    参数:
        log_dir: 如果传入目录，则额外写入 `robot_car.log`；如果为 None，则只输出到终端。
    """

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "robot_car.log", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )
