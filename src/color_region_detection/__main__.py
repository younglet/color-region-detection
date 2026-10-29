"""包入口：python -m color_region_detection 等价于 python main.py"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main() or 0)
