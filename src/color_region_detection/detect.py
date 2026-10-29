"""color_region_detection/detect.py — 核心检测：只有三个可调参数"""
from dataclasses import dataclass

import cv2
import numpy as np

from .colors import COLOR_NAMES, DET_IDS, DETECTABLE, classify

BLUR = 3          # 中值滤波核，压传感器噪点


# ============ 参数：界面上只有这三个 ============
@dataclass
class Params:
    """
    min_size_pct  最小尺寸 = 画面短边的百分比（1~100）
    max_size_pct  最大尺寸 = 画面短边的百分比（1~100，100 = 不限）
    purity        纯度最低需求 0~100，越大越严

    min_fill 最小填充率，拦碎片/稀疏轮廓：
    圆形填充率约 0.785，葫芦这类非凸形状约 0.57，取 0.35；取 0.6 会误杀葫芦。
    """
    min_size_pct: int = 20
    max_size_pct: int = 100
    purity: int = 50
    min_fill: float = 0.35

    def __post_init__(self):
        self.min_size_pct = int(np.clip(int(self.min_size_pct), 1, 100))
        self.max_size_pct = int(np.clip(int(self.max_size_pct), 1, 100))
        self.purity = int(np.clip(int(self.purity), 0, 100))
        self.min_fill = float(np.clip(float(self.min_fill), 0.0, 1.0))

    # ---- 百分比 -> 像素（参照画面短边）----
    def min_px(self, frame_wh):
        return max(1, int(round(min(frame_wh) * self.min_size_pct / 100.0)))

    def max_px(self, frame_wh):
        if self.max_size_pct >= 100:
            return 0
        return max(1, int(round(min(frame_wh) * self.max_size_pct / 100.0)))

    # ---- 纯度 level -> 两条约束（标定依据见根 README.md）----
    @property
    def min_s(self):
        """饱和度下限：level 0 -> 50，50 -> 110，100 -> 170"""
        return int(round(50 + 1.2 * self.purity))

    @property
    def max_std_v(self):
        """区内明度差上限：level 0 -> 60，50 -> 30，100 -> 6"""
        return max(6.0, 60 - 0.6 * self.purity)

    def accept(self, w, h, area, min_px, max_px):
        if w < min_px or h < min_px:
            return False
        if max_px and (w > max_px or h > max_px):
            return False
        return area >= self.min_fill * w * h

    def describe(self, frame_wh=None):
        if frame_wh:
            return (f"size {self.min_size_pct}%~{self.max_size_pct}% "
                    f"({self.min_px(frame_wh)}~"
                    f"{self.max_px(frame_wh) or '不限'}px)  "
                    f"purity>={self.purity}")
        return f"size {self.min_size_pct}%~{self.max_size_pct}%  purity>={self.purity}"


# ============ 区内纯度：只算 S 和 V ============
def region_sv(hsv_roi, mask):
    """连通域内的 (mean_s, std_v)。测不到返回 None（宁可丢弃）。"""
    if int(mask.sum()) == 0:
        return None
    s = hsv_roi[..., 1][mask]
    v = hsv_roi[..., 2][mask]
    mean_s, std_v = float(s.mean()), float(v.std())
    if not (np.isfinite(mean_s) and np.isfinite(std_v)):
        return None
    return mean_s, std_v


def border_touches(labels):
    """每个连通域触达了几条画面边界（>=3 视为背景）"""
    cnt = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for line in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        for lab in np.unique(line):
            if lab:
                cnt[int(lab)] += 1
    return cnt


# ============ 检测 ============
def detect(color_map, hsv, prm, frame_wh, ignore_bg=True):
    """
    返回 [(name, x, y, w, h, area, fill, cx, cy), ...]，坐标在传入分辨率下。
    """
    min_px, max_px = prm.min_px(frame_wh), prm.max_px(frame_wh)
    min_s, max_std_v = prm.min_s, prm.max_std_v
    counts = np.bincount(color_map.ravel(), minlength=len(COLOR_NAMES))
    # 该色总像素 < 达标块的面积下限 -> 必无达标块，直接跳过（提速）
    skip_below = max(16, int(min_px * min_px * prm.min_fill))

    out = []
    for cid in DET_IDS:
        if counts[cid] < skip_below:
            continue
        # bool 比较结果本来就是 1 字节，view 一下零拷贝喂给 CC
        mask = (color_map == cid).view(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            continue
        bt = border_touches(labels) if ignore_bg else None

        for i in range(1, n):
            if bt is not None and bt[i] >= 3:
                continue                    # 触达 >=3 条边界 = 背景
            x, y, w, h, area = (int(stats[i, cv2.CC_STAT_LEFT]),
                                int(stats[i, cv2.CC_STAT_TOP]),
                                int(stats[i, cv2.CC_STAT_WIDTH]),
                                int(stats[i, cv2.CC_STAT_HEIGHT]),
                                int(stats[i, cv2.CC_STAT_AREA]))
            if not prm.accept(w, h, area, min_px, max_px):
                continue
            sv = region_sv(hsv[y:y + h, x:x + w], labels[y:y + h, x:x + w] == i)
            if sv is None or sv[0] < min_s or sv[1] > max_std_v:
                continue
            out.append((COLOR_NAMES[cid], x, y, w, h, area,
                        area / float(max(w * h, 1)),
                        float(cents[i][0]), float(cents[i][1])))
    out.sort(key=lambda b: b[5], reverse=True)
    return out


def process(frame, prm, ignore_bg=True):
    """在传入分辨率上检测"""
    src = cv2.medianBlur(frame, BLUR) if BLUR >= 3 else frame
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cmap = classify(hsv, prm.min_s)
    return detect(cmap, hsv, prm, (frame.shape[1], frame.shape[0]), ignore_bg)


def detect_scaled(frame, prm, proc=None, ignore_bg=True):
    """低分辨率检测 + 坐标回缩到原帧"""
    fh, fw = frame.shape[:2]
    if not proc or proc[0] <= 0:
        proc = (fw, fh)
    elif proc[1] == 0:
        proc = (proc[0], max(2, int(round(proc[0] * fh / float(fw))) // 2 * 2))
    small = frame if proc == (fw, fh) else cv2.resize(frame, proc,
                                                      interpolation=cv2.INTER_AREA)
    blocks = process(small, prm, ignore_bg)
    if small is frame:
        return blocks, (fw, fh)
    sx, sy = fw / float(small.shape[1]), fh / float(small.shape[0])
    out = [(n, int(round(x * sx)), int(round(y * sy)),
            max(1, int(round(w * sx))), max(1, int(round(h * sy))),
            int(round(a * sx * sy)), f, cx * sx, cy * sy)
           for n, x, y, w, h, a, f, cx, cy in blocks]
    return out, (fw, fh)


__all__ = ["Params", "detect", "detect_scaled", "process", "region_sv",
           "border_touches", "DETECTABLE", "BLUR"]