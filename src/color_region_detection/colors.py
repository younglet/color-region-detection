"""
color_region_detection/colors.py — 颜色常量与 HSV 分段表

OpenCV 的 H 取值 0~179，是真实色相 0~360° 的一半。
分界线按实测色相标定（tests/data/photo_test.png 绿卡 H≈72、蓝卡 H≈100），
非理论色相中点。

本表是全项目唯一的判定区间：调整某个颜色改它对应的 lo/hi；
改完用 python main.py --print-bands 打印当前表确认。

顺序从 0 递增，最后一段要覆盖 H 的回绕区间（red 165~180）。

需要重新标定（换场地 / 换灯 / 换批次色卡）时：
    python src/color_region_detection/calibrate.py --apply
"""

import numpy as np

# ============ 颜色常量表（判定顺序） ============
COLOR_NAMES = (
    "black",     # 黑
    "white",     # 白
    "gray",      # 灰
    "red",       # 红
    "orange",    # 橙
    "yellow",    # 黄
    "green",     # 绿
    "cyan",      # 青
    "blue",      # 蓝
    "purple",    # 紫
    "pink",      # 粉
)

COLOR_CN = {
    "black": "黑", "white": "白", "gray": "灰",
    "red": "红", "orange": "橙", "yellow": "黄",
    "green": "绿", "cyan": "青", "blue": "蓝",
    "purple": "紫", "pink": "粉",
}

# 名字 -> 整数 ID（用于掩膜比对，比字符串快得多）
COLOR_ID = {name: i for i, name in enumerate(COLOR_NAMES)}
NUM_COLORS = len(COLOR_NAMES)

# 无彩色：黑/白/灰，永远不参与识别
ACHROMATIC_NAMES = ("black", "white", "gray")

# 绘制用 BGR
COLOR_BGR = {
    "black": (40, 40, 40),
    "white": (255, 255, 255),
    "gray": (160, 160, 160),
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 230, 255),
    "green": (0, 220, 0),
    "cyan": (255, 220, 0),
    "blue": (255, 80, 0),
    "purple": (190, 0, 190),
    "pink": (255, 130, 205),
}

# ============ 判定区间：唯一的真值来源 ============
# (颜色名, H 下限含, H 上限不含)
HUE_BANDS = (
    ("red",    0,    7),      # 0~14°      红色
    ("orange", 7,   22),      # 14~44°     橙色
    ("yellow", 22,  45),      # 44~90°     黄色
    ("green",  45,  82),      # 90~164°    绿色
    ("cyan",   82,  97),      # 164~194°   青色
    ("blue",   97, 127),      # 194~254°   蓝色
    ("purple", 127, 142),     # 254~284°   紫色
    ("pink",   142, 165),     # 284~330°   粉色
    ("red",   165, 180),      # 330~360°   回绕到红色
)

BLACK_V_MAX = 46       # V < 46 判黑
RED_WRAP_MIN = 165     # H >= 165 归红

# 只有这 8 种高纯度彩色参与识别
DETECTABLE = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")
DET_IDS = [COLOR_ID[n] for n in DETECTABLE]


# ============ 纯色区域（BGR），画示意图 / 生成测试图用 ============
def pure_bgr(name, v=255):
    """按颜色名生成高纯度 BGR 色彩区域（V 可调）"""
    import cv2

    if name in ("black", "gray", "white"):
        gray_val = {"black": 20, "gray": 128, "white": 245}[name]
        hsv = np.uint8([[[0, 0, gray_val]]])
    else:
        # 用该色相分段的中点，避免落在边界上；red 取回绕段中点
        h_center = {"red": (RED_WRAP_MIN + 180) // 2 % 180}
        for nm, lo, hi in HUE_BANDS:
            if nm == name:
                h_center[name] = (lo + hi) // 2
                break
        hsv = np.uint8([[[h_center[name], 255, v]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


# ============ 逐像素分类 ============
def classify(hsv, sat_floor):
    """
    把每个像素标成颜色 ID；非目标像素（黑/白/灰/低饱和/过暗）标 0。
    sat_floor: 饱和度下限，低于此值一律 0（即纯度旋钮 level 推导出来的）。
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.zeros(h.shape, np.uint8)

    valid = (s >= sat_floor) & (v >= BLACK_V_MAX)
    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        return out

    hv = h.ravel()[idx]
    res = np.zeros(idx.size, np.uint8)
    for name, lo, hi in HUE_BANDS:
        sel = (hv >= lo) & (hv < hi)
        if sel.any():
            res[sel] = COLOR_ID[name]
    out.ravel()[idx] = res
    return out


def name_of(cid):
    """颜色 ID -> 颜色名"""
    return next(n for n, i in COLOR_ID.items() if i == cid)


def print_bands():
    """打印当前颜色判定区间表"""
    print("颜色判定区间（OpenCV H 0~179，真实色相 = H*2）:")
    print(f"{'颜色':9s} {'H 区间':>14s} {'真实色相':>14s}")
    for name, lo, hi in HUE_BANDS:
        deg = f"{lo*2}~{min(hi,180)*2}°"
        print(f"{name:9s} {f'[{lo}, {hi})':>14s} {deg:>14s}")
    print(f"\n无彩色门槛: V < {BLACK_V_MAX} 判黑；S < 纯度阈值 判无彩色（都不识别）")
