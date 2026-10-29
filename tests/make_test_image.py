"""
color-region-detection/tests/make_test_image.py — 生成形状测试图

已知颜色 + 已知尺寸的色彩区域，用来验证尺寸阈值是否准确。
输出：color-region-detection/tests/data/test.png

运行：python color-region-detection/tests/make_test_image.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # color-region-detection/（项目根）
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from color_region_detection.colors import pure_bgr                    # noqa: E402

DATA = os.path.join(HERE, "data")
W, H = 960, 540
img = np.full((H, W, 3), 128, dtype=np.uint8)   # 中灰背景（让白色彩区域可区分）

blocks = [
    # (name, x, y, w, h)  —— 互不重叠，用于精确验证尺寸
    ("red",     40,  40, 120, 120),   # 大区域：应命中
    ("green",  200,  40,  90,  70),   # 中块：命中
    ("blue",   330,  40,  30,  25),   # 小块：命中
    ("yellow", 400,  40,  16,  16),   # 极小块：应被 min_side 过滤掉
    ("cyan",    40, 220, 200,  60),   # 扁长条
    ("orange", 300, 220,  70, 160),   # 高竖条
    ("purple",  420, 220, 100, 100),
    ("pink",    560, 220,  50,  50),
    ("white",   650, 220,  80,  40),   # 中灰底上可区分
    ("black",   770, 220,  60,  60),
    ("red",     650,  40,  40,  40),   # 与第一个红色彩区域同色不同位置
    ("green",   700,  40, 140,  40),   # 细长条：fill 高
    ("blue",    860,  40,  60, 140),   # 竖条
]

for name, x, y, w, h in blocks:
    cv2.rectangle(img, (x, y), (x + w, y + h), pure_bgr(name), -1)


# ---- 圆形物体：外接矩形与形状无关 ----
# 球体带明暗渐变：暗部落进 black/gray，不参与识别
def draw_shaded_ball(img, cx, cy, r, name):
    base = np.array(pure_bgr(name), dtype=np.float32)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = d <= r
    # 光照：越远越暗，最暗处叠加环境光（降低饱和度）
    light = np.clip(1.15 - 0.9 * (d / r), 0.0, 1.0)[:, :, None]
    ambient = np.array([255, 255, 255], dtype=np.float32) * 0.35
    shaded = base[None, None, :] * light + ambient * (1 - light)
    img[inside] = np.clip(shaded[inside], 0, 255).astype(np.uint8)


draw_shaded_ball(img, 480, 400, 58, "red")

# 葫芦：两个圆叠在一起（同色且相邻 -> 连成一个连通域）
for cxy, rr in (((680, 372), 42), ((700, 438), 32)):
    cv2.circle(img, cxy, rr, pure_bgr("green"), -1)

# 不加标题文字：文字会被当成色彩区域检出，干扰真值
os.makedirs(DATA, exist_ok=True)
out = os.path.join(DATA, "test.png")
cv2.imwrite(out, img)
print(f"{out} written: {W}x{H}, {len(blocks)} rect blocks + 1 ball + 1 gourd")
for b in blocks:
    print("  ", b)
