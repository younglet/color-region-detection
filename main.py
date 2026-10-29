#!/usr/bin/env python
"""
color-region-detection/main.py — 入口程序

在本文件夹（color-region-detection/）下直接运行即可，不需要安装、不需要父目录在 PYTHONPATH：

    python main.py                          摄像头
    python main.py --source photo_test.png  图片
    python main.py --print-bands            打印颜色判定区间表

源码在 src/color_region_detection/（标准 src 布局），这里只负责把 src/ 接上并转发给 cli。
装成命令行工具后（pip install -e .）也可以直接 `python -m color_region_detection`。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from color_region_detection.cli import main      # noqa: E402

if __name__ == "__main__":
    sys.exit(main() or 0)
