"""
color-region-detection/tests/make_purity_test.py — 生成干扰物测试图

左侧是纯色目标（应被识别），右侧是易误报的干扰物。
用于验证三组阈值（大小 / 纯度 / 明度差）的有效性。
输出：color-region-detection/tests/data/purity_test.png

运行：python color-region-detection/tests/make_purity_test.py
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
W, H = 960, 480
img = np.full((H, W, 3), 128, dtype=np.uint8)   # 中灰底


def hsv_rect(img, x, y, w, h, hsv):
    """按给定 HSV 直接填充矩形"""
    px = np.uint8([[list(hsv)]])            # 形状 (1,1,3)
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in bgr), -1)


# ---------- 左半：真目标（高纯度、纯净、够大） ----------
TARGETS = [
    ("red",     40,  60, 110, 110, (0, 255, 255)),    # H0   red   (0-7)
    ("green",  180,  60, 110, 110, (45, 255, 255)),   # H45  green (45-82)
    ("blue",   320,  60, 110, 110, (100, 255, 255)),  # H100 blue  (97-127)
    ("yellow", 460,  60, 110, 110, (30, 255, 255)),   # H30  yellow(22-45)
]
for name, x, y, w, h, hsv in TARGETS:
    hsv_rect(img, x, y, w, h, hsv)

# ---------- 右半：干扰物 ----------
# D1 淡粉：饱和度很低（只按色相判定会当成 red）
hsv_rect(img, 620, 60, 110, 110, (0, 60, 240))
# D2 淡橙：饱和度低
hsv_rect(img, 760, 60, 110, 110, (15, 70, 240))
# D3 红蓝竖条纹：局部颜色极不均匀（卷积方差门会剔除）
for i in range(10):
    x = 620 + i * 11
    col = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
    cv2.rectangle(img, (x, 230), (x + 10, 340), col, -1)
# D4 明暗渐变红条：区内明度差很大（区内明度差上限会剔除）
for i in range(110):
    v = 60 + i * 1.7
    hsv_rect(img, 760, 230, 1, 110, (0, 255, int(min(v, 255))))
# D5 带阴影的球：明暗渐变
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt((xx - 100) ** 2 + (yy - 320) ** 2)
ball = d <= 62
light = np.clip(1.25 - 1.05 * (d / 62.0), 0, 1)[:, :, None]
base = np.array((30, 30, 240), dtype=np.float32)      # 红色 BGR
amb = np.array((255, 255, 255), dtype=np.float32) * 0.3
shaded = base[None, None, :] * light + amb * (1 - light)
img[ball] = np.clip(shaded[ball], 0, 255).astype(np.uint8)
# D6 小红点噪声（面积太小）
for i in range(40):
    x = 480 + (i * 37) % 40
    y = 250 + (i * 53) % 60
    cv2.rectangle(img, (x, y), (x + 2, y + 2), (0, 0, 255), -1)

# 标线：上排=真目标，下排=干扰物
cv2.line(img, (600, 190), (930, 190), (80, 80, 80), 1)
cv2.line(img, (20, 190), (610, 190), (80, 80, 80), 1)

os.makedirs(DATA, exist_ok=True)
out = os.path.join(DATA, "purity_test.png")
cv2.imwrite(out, img)
print(f"{out}: 4 个真目标 + 6 类干扰物")
print("  目标: red/green/blue/yellow 纯色方块 (110x110)")
print("  干扰: D1淡粉 D2淡橙 D3红蓝条纹 D4明暗渐变 D5阴影球 D6小红点噪声")
