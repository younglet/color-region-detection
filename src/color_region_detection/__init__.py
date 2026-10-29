"""
color_region_detection — 精简版色彩区域检测（3 个可调参数的独立包）

自包含：本包不依赖项目根目录的任何文件，可整个文件夹拷走单独运行。

    from color_region_detection.detect import Params, detect_scaled
    from color_region_detection.colors import HUE_BANDS, classify

命令行入口：
    python main.py                     # 在项目根目录（color-region-detection/）下直接跑
    python -m color_region_detection             # 安装后（pip install -e .）任意目录可跑
    python -m color_region_detection --print-bands  # 打印颜色判定区间表
"""

__version__ = "3.0"
__all__ = ["colors", "detect", "ui", "cli"]
