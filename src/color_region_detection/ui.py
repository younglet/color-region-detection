"""color_region_detection/ui.py — 画框、信息栏、滑动条（只有三条）"""
import cv2
import numpy as np

from .colors import COLOR_BGR, COLOR_ID, DETECTABLE, pure_bgr


# ============ 画框 + 标注 ============
def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw(frame, blocks, single=False):
    """画外接框 + 中心点 + 单行 ASCII 标注（自动避让重叠）"""
    disp = frame.copy()
    H, W = disp.shape[:2]
    placed = []

    for name, x, y, w, h, area, fill, cx, cy in blocks:
        bgr = COLOR_BGR[name]
        # 框色固定白/亮黄：随区域色会与区域本身冲突（如黄区域配黄框）
        box = (255, 255, 0) if single else (255, 255, 255)
        cv2.rectangle(disp, (x, y), (x + w, y + h), box, 2 if w >= 40 else 1)
        cv2.circle(disp, (int(cx), int(cy)), 3, box, -1)

        label = f"{name} {w}x{h} {area}px"
        fs, th = 0.42, 1
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        for tx, ty in [(x, y - tht - 8), (x, y + h + tht + 8),
                       (x + w + 6, y + h // 2 + tht // 2),
                       (x - tw - 6, y + h // 2 + tht // 2),
                       (x + 40, y - tht - 8)]:
            tx = int(max(0, min(tx, W - tw - 4)))
            ty = int(max(tht + 4, min(ty, H - 4)))
            r = (tx - 3, ty - tht - 5, tx + tw + 3, ty + 5)
            if not any(_overlap(r, q) for q in placed):
                break
        placed.append(r)

        lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
        txt = (255, 255, 255) if lum < 110 else bgr
        cv2.rectangle(disp, (r[0], r[1]), (r[2], r[3]), (25, 25, 25), -1)
        cv2.rectangle(disp, (r[0], r[1]), (r[2], r[3]), box, 1)
        cv2.putText(disp, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, txt, th, cv2.LINE_AA)
    return disp


def draw_bar(disp, prm, blocks, fps, proc, proc_wh):
    """底部信息栏：三个旋钮的当前值 + 各颜色命中数"""
    h, w = disp.shape[:2]
    bar = np.zeros((44, w, 3), np.uint8)
    mp = prm.min_px(proc_wh)
    xp = prm.max_px(proc_wh) or "MAX"
    info = (f"[det {proc[0]}x{proc[1]}]  size {prm.min_size_pct}%~{prm.max_size_pct}%"
            f" ({mp}~{xp}px)  purity>={prm.purity}  blocks:{len(blocks)}"
            f"  fps:{fps:4.1f}")
    cv2.putText(bar, info, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)

    cnt = {n: 0 for n in DETECTABLE}
    for b in blocks:
        cnt[b[0]] += 1
    step = max(38, (w - 16) // len(DETECTABLE))
    for i, n in enumerate(DETECTABLE):
        x = 8 + i * step
        on = cnt[n] > 0
        cv2.rectangle(bar, (x, 26), (x + 14, 38),
                      pure_bgr(n) if on else (50, 50, 50), -1)
        cv2.rectangle(bar, (x, 26), (x + 14, 38), (200, 200, 200), 1)
        cv2.putText(bar, str(cnt[n]), (x + 17, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255) if on else (110, 110, 110),
                    1, cv2.LINE_AA)
    return np.vstack([disp, bar])


# ============ 滑动条：只有三条 ============
SPECS = (("minSize%", "min_size_pct", 1, 100, 1),
         ("maxSize%", "max_size_pct", 1, 100, 1),
         ("purity",   "purity",       1, 100, 0))


class Trackbars:
    def __init__(self, win, prm):
        self.win, self.prm, self._g = win, prm, False
        for label, attr, scale, hi, lo in SPECS:
            cv2.createTrackbar(label, win, self._pos(attr, scale, hi, lo), hi,
                               self._cb(attr, scale))

    def _pos(self, attr, scale, hi, lo):
        return int(max(lo, min(hi, round(getattr(self.prm, attr) * scale))))

    def _cb(self, attr, scale):
        def cb(val):
            if self._g:
                return
            new = val / scale
            if float(getattr(self.prm, attr)) == float(new):
                return
            setattr(self.prm, attr, new)
            type(self.prm).__post_init__(self.prm)
        return cb

    def sync(self):
        self._g = True
        for label, attr, scale, hi, lo in SPECS:
            cv2.setTrackbarPos(label, self.win, self._pos(attr, scale, hi, lo))
        self._g = False


def handle_key(key, prm):
    """w/W 最小尺寸  s/S 最大尺寸  e/E 纯度  x 复位"""
    if key == ord('w'):
        prm.min_size_pct += 1
    elif key == ord('W'):
        prm.min_size_pct -= 1
    elif key == ord('s'):
        prm.max_size_pct += 5
    elif key == ord('S'):
        prm.max_size_pct -= 5
    elif key == ord('e'):
        prm.purity += 5
    elif key == ord('E'):
        prm.purity -= 5
    elif key == ord('x'):
        prm.__init__()                     # 全部复位
    type(prm).__post_init__(prm)