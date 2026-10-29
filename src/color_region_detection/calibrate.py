"""
calibrate.py — 色卡参数更新：实测色卡 → 重算 HUE_BANDS → 重画色条图

做什么
------
拍一张色卡照片，量出每张卡在当前光照下的真实色相，重算颜色分界（HUE_BANDS），
用新参数重画「渐变色条 · 颜色分段参考」图，并把新图与原图并排放到桌面比对。

为什么需要
----------
`colors.py` 的 HUE_BANDS 是某次实测标定的。换场地、换灯、换批次色卡之后边界会漂：
弱光下深绿的 H 往青蓝跑，蓝卡容易被判成 cyan，粉/紫边界也容易串。
本模块重新量一遍，把「当前参数 vs 实测」的差值直接画出来。

分界怎么算
----------
1. 每张卡测出主导色相 H（直方图众数附近 ±3 再取环形均值，抗噪）
2. 同色多张卡 → 环形均值合并成一个中心
3. 某颜色有实测 → 中心 = 实测角；没实测 → 中心沿用当前表的中心
4. 新分界 = 相邻两个颜色中心的中点（环形，red 的 0/360 回绕自动处理）
5. 结果必然连续、覆盖 0~180、无重叠无缝隙

用法
----
    python -m color_region_detection.calibrate                       # 默认图，产物到桌面
    python -m color_region_detection.calibrate --source <色卡照片>
    python -m color_region_detection.calibrate --labels yellow,cyan,green,red
    python -m color_region_detection.calibrate --set green=49:81     # 手动锁定某色
    python -m color_region_detection.calibrate --apply               # 写回 colors.py + 重画 color_bands.png

产物（默认在桌面）
------------------
    color_bands_new.png       新参数重画的色条（与原图对比的主体产物）
    color_bands_compare.png   上=原图 下=新图 + 分界对照
    card_params_compare.png   色卡照片实测标注（左原图 / 右标注）
    card_params_report.md     文字报告
    card_params_new.json      结构化参数
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:                                   # 包内导入
    from .colorbar import render as render_bands
    from .colors import COLOR_CN, HUE_BANDS
except ImportError:                    # 直接跑文件时补上 src 路径
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from color_region_detection.colorbar import render as render_bands
    from color_region_detection.colors import COLOR_CN, HUE_BANDS

HERE = Path(__file__).resolve().parent              # src/color_region_detection/
PROJECT = HERE.parents[1]                           # 项目根
DESKTOP = Path(r"C:\Users\young\Desktop")
DEFAULT_SOURCE = PROJECT / "tests" / "data" / "photo_test.png"
DEFAULT_BANDS_PNG = PROJECT / "color_bands.png"

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"
BG = (18, 20, 24)
TEXT = (236, 239, 244)
MUTED = (150, 157, 168)
OK = (120, 220, 130)
BAD = (255, 110, 110)

# 颜色在圆周上的固定顺序（red 首尾相接）
CIRCLE_ORDER = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]


# ============================ 小工具 ============================
def h2deg(h):
    """OpenCV H(0~179) -> 真实色相角(0~360)，支持标量/数组"""
    return np.asarray(h, dtype=np.float64) * 2.0


def deg2h(d):
    """真实色相角 -> OpenCV H(0~179)"""
    return np.asarray(d, dtype=np.float64) / 2.0


def circ_delta(a, b):
    """两个色相角的最短夹角（度，正数）"""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def circ_mean(angles):
    """环形均值（处理 0/360 回绕）"""
    a = np.asarray(angles, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    r = np.deg2rad(a)
    x, y = np.mean(np.cos(r)), np.mean(np.sin(r))
    return float(math.degrees(math.atan2(y, x)) % 360.0)


def band_of(h):
    """当前 HUE_BANDS 判定一个 OpenCV H 属于哪个颜色"""
    for name, lo, hi in HUE_BANDS:
        if lo <= h < hi:
            return name
    return HUE_BANDS[-1][0]


def current_center_deg(name):
    """当前表里某颜色的中心角（回绕色取两段圆心）"""
    segs = [(lo * 2, hi * 2) for n, lo, hi in HUE_BANDS if n == name]
    if not segs:
        return None
    return circ_mean([x for seg in segs for x in seg])


def fmt_h(v):
    """整数就写整数（保持 colors.py 原风格）"""
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def font(size, bold=False):
    p = FONT_B if bold else FONT
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def save_img(img, path):
    """保存图片；文件被占用时先写临时文件再改名（Windows 上偶发 PermissionError）"""
    path = Path(path)
    for _ in range(3):
        try:
            img.save(path)
            return path
        except OSError:
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                img.save(tmp)
                os.replace(tmp, path)
                return path
            except OSError:
                time.sleep(0.3)
    raise OSError(f"无法写入 {path}")


# ============================ 找色卡 ============================
def find_hue_peaks(hsv, sat_floor, min_frac=0.06, max_peaks=10, merge_h=4):
    """高饱和像素里的色相峰，每个峰 ≈ 一种卡片颜色"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s >= sat_floor) & (v >= 40)
    if not mask.any():
        return []

    hist = np.bincount(h[mask], minlength=180).astype(np.float64)
    k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), k, mode="same")[2:-2]
    thresh = max(smooth.sum() * min_frac / 100.0, 1.0)

    peaks = []
    for i in range(180):
        if smooth[i] >= smooth[(i - 1) % 180] and smooth[i] > smooth[(i + 1) % 180] \
                and smooth[i] >= thresh:
            peaks.append((i, smooth[i]))
    peaks.sort(key=lambda p: -p[1])
    kept = []
    for hpk, _cnt in peaks:
        if all(circ_delta(hpk * 2.0, k2 * 2.0) > merge_h * 2 for k2, _ in kept):
            kept.append((hpk, _cnt))
        if len(kept) >= max_peaks:
            break
    kept.sort(key=lambda p: p[0])
    return kept


def _merge_boxes(boxes, gap):
    """把互相贴近/重叠的框并成一张卡（阴影把卡切成两块的情况）"""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            ax, ay, aw, ah = boxes[i]
            bx, by, bw, bh = boxes[j]
            dx = max(0, max(ax - (bx + bw), bx - (ax + aw)))
            dy = max(0, max(ay - (by + bh), by - (ay + ah)))
            if max(dx, dy) <= gap:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def find_cards(img, sat_floor=100, min_area_frac=0.002, max_area_frac=0.35,
               hue_tol=14, min_fill=0.35, merge_gap=14):
    """
    定位色卡：按色相峰分组再取连通域，相邻不同色的卡不会粘成一坨；
    同一张卡被阴影切成几块时按 bbox 距离合并回来。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H, W = h.shape
    img_area = float(H * W)

    sat_mask = ((s >= sat_floor) & (v >= 40)).astype(np.uint8)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cards = []
    for peak_h, _cnt in find_hue_peaks(hsv, sat_floor):
        d = np.abs(h.astype(np.int16) - peak_h)
        d = np.minimum(d, 180 - d)                       # 环形距离
        mask = ((d <= hue_tol) & (sat_mask > 0)).astype(np.uint8)
        n, labels, stats, _c = cv2.connectedComponentsWithStats(mask, 8)

        comps = []
        for i in range(1, n):
            x, y, ww, hh, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
                                  int(stats[i, cv2.CC_STAT_AREA]))
            frac = area / img_area
            if frac < min_area_frac or frac > max_area_frac:
                continue
            if area / float(max(ww * hh, 1)) < min_fill:
                continue
            comps.append([i, x, y, ww, hh, area])

        boxes = [(c[1], c[2], c[3], c[4]) for c in comps]
        for grp in _merge_boxes(boxes, merge_gap):
            ids = [comps[i][0] for i in grp]
            x0 = min(comps[i][1] for i in grp)
            y0 = min(comps[i][2] for i in grp)
            x1 = max(comps[i][1] + comps[i][3] for i in grp)
            y1 = max(comps[i][2] + comps[i][4] for i in grp)
            sel = np.isin(labels, ids)
            hs, ss, vs = h[sel], s[sel], v[sel]
            if hs.size == 0:
                continue
            hh_hist = np.bincount(hs, minlength=180)
            dom = int(np.argmax(hh_hist))
            keep = np.abs(hs.astype(np.int16) - dom) <= 3
            if keep.sum() < max(10, hs.size * 0.2):
                keep = np.ones(hs.shape, bool)
            ang = circ_mean(h2deg(hs[keep]))
            cards.append(dict(
                x=x0, y=y0, w=x1 - x0, h=y1 - y0, area=int(hs.size),
                fill=round(hs.size / float(max((x1 - x0) * (y1 - y0), 1)), 3),
                peak=peak_h, hue=round(float(deg2h(ang)), 1), hue_deg=round(float(ang), 1),
                s=round(float(ss.mean()), 1), sd_s=round(float(ss.std()), 1),
                v=round(float(vs.mean()), 1), sd_v=round(float(vs.std()), 1),
                parts=len(ids),
            ))
    cards.sort(key=lambda c: -c["area"])
    return cards


def background_stats(img, cards):
    """卡片以外的像素的饱和度分布，用来定饱和度门槛"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    mask = np.ones(s.shape, bool)
    for c in cards:
        mask[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = False
    bg = s[mask]
    if bg.size == 0:
        return None
    return dict(p50=float(np.percentile(bg, 50)), p90=float(np.percentile(bg, 90)),
                p99=float(np.percentile(bg, 99)), max=int(bg.max()))


# ============================ 提议新参数 ============================
def to_int_table(rows):
    """浮点分段 -> 整数 H 分段（保证 0~180 连续、无重叠、无缝隙）"""
    rs = sorted(rows, key=lambda r: r[1])
    out = [[n, int(round(lo)), int(round(hi))] for n, lo, hi in rs]
    for i in range(1, len(out)):
        if out[i][1] < out[i - 1][2]:
            out[i][1] = out[i - 1][2]
        if out[i][2] <= out[i][1]:
            out[i][2] = out[i][1] + 1
    if out:
        out[0][1] = 0
        out[-1][2] = 180
    return [tuple(r) for r in out]


def check_table(rows):
    """检查分段表是否连续、无重叠、覆盖 0~180"""
    issues = []
    rows = sorted(rows, key=lambda r: r[1])
    for i in range(len(rows) - 1):
        if rows[i][2] > rows[i + 1][1] + 1e-6:
            issues.append(f"{rows[i][0]}({rows[i][2]:.1f}) 与 {rows[i+1][0]}({rows[i+1][1]:.1f}) 重叠")
        elif rows[i][2] < rows[i + 1][1] - 1e-6:
            issues.append(f"{rows[i][0]} 与 {rows[i+1][0]} 之间有缝隙 "
                          f"[{rows[i][2]:.1f},{rows[i+1][1]:.1f})")
    if rows and rows[0][1] > 1e-6:
        issues.append(f"起点不是 0（{rows[0][1]:.1f}）")
    if rows and rows[-1][2] < 180 - 1e-6:
        issues.append(f"终点不是 180（{rows[-1][2]:.1f}）")
    return issues


def propose_bands(cards, labels=None, overrides=None):
    """
    - 有实测的颜色：中心角 = 实测环形均值
    - 没实测的颜色：中心角 = 当前表的中心
    - 新分界 = 相邻颜色中心的中点（环形，red 回绕自动处理）
    overrides: {"green": (48, 82)} 手动锁定某个颜色的 H 区间
    """
    overrides = overrides or {}

    for i, c in enumerate(cards):
        if labels and i < len(labels):
            c["truth"], c["truth_src"] = labels[i], "manual"
        else:
            c["truth"], c["truth_src"] = band_of(int(round(c["hue"]))), "auto"
        c["current"] = band_of(int(round(c["hue"])))
        c["ok"] = (c["current"] == c["truth"])

    groups = {}
    for c in cards:
        groups.setdefault(c["truth"], []).append(c)
    measured = {name: circ_mean([x["hue_deg"] for x in cs]) for name, cs in groups.items()}

    centers = {}
    for name in CIRCLE_ORDER:
        if name in measured:
            centers[name] = measured[name]
        elif name in overrides:
            lo, hi = overrides[name]
            centers[name] = circ_mean([lo * 2, hi * 2])
        else:
            cc = current_center_deg(name)
            if cc is not None:
                centers[name] = cc

    order = [n for n in CIRCLE_ORDER if n in centers]
    n = len(order)

    bnd = {}
    for i, name in enumerate(order):
        nxt = order[(i + 1) % n]
        if name in overrides:
            bnd[(name, nxt)] = overrides[name][1] * 2.0        # 手动段的高边界
        elif nxt in overrides:
            bnd[(name, nxt)] = overrides[nxt][0] * 2.0        # 手动段的低边界
        else:
            a, b = centers[name], centers[nxt]
            bnd[(name, nxt)] = (a + (b - a) % 360.0 / 2.0) % 360.0

    rows = []
    for i, name in enumerate(order):
        prv = order[i - 1]
        lo_d = bnd[(prv, name)]
        hi_d = bnd[(name, order[(i + 1) % n])]
        lo_h, hi_h = float(deg2h(lo_d)), float(deg2h(hi_d))
        if lo_h <= hi_h:
            rows.append((name, lo_h, hi_h))
        else:                                                  # 跨 0 -> 拆两段
            rows.append((name, lo_h, 180.0))
            rows.append((name, 0.0, hi_h))
    rows.sort(key=lambda r: (r[1], r[2]))
    rows = to_int_table(rows)

    cur = {}
    for nme, lo, hi in HUE_BANDS:
        cur.setdefault(nme, []).append((lo, hi))
    diff = []
    for name in order:
        if name not in measured and name not in overrides:
            continue
        new = [r for r in rows if r[0] == name]
        if not new:
            continue
        nlo = min(r[1] for r in new)
        nhi = max(r[2] for r in new)
        old = cur.get(name, [(nlo, nhi)])
        olo = min(r[0] for r in old)
        ohi = max(r[1] for r in old)
        if abs(nlo - olo) > 0.5 or abs(nhi - ohi) > 0.5:
            diff.append(dict(name=name, old=[round(olo, 1), round(ohi, 1)],
                             new=[round(nlo, 1), round(nhi, 1)],
                             center=round(centers[name], 1),
                             reason="manual" if name in overrides else "measured"))

    return dict(rows=rows, centers=centers, measured=measured, diff=diff,
                groups=groups, issues=check_table(rows))


# ============================ 出图 ============================
def make_bands_compare(out_png, orig_png, new_png, diff_rows, note):
    """上=原图 下=新图 的并排对比"""
    o = Image.open(orig_png).convert("RGB")
    n = Image.open(new_png).convert("RGB")
    W = max(o.width, n.width)
    head, foot = 46, 30 + 20 * len(diff_rows) + 12
    canvas = Image.new("RGB", (W, head + o.height + n.height + foot), BG)
    d = ImageDraw.Draw(canvas)
    f_title, f_sub, f_row = font(24, True), font(15), font(14)

    d.text((14, 10), "原图 vs 新图 · 渐变色条", font=f_title, fill=TEXT)
    d.text((14 + 250, 16), note, font=f_sub, fill=MUTED)
    canvas.paste(o, (0, head))
    y = head + o.height + 6
    d.text((14, y), "新图（新 HUE_BANDS）", font=f_sub, fill=OK)
    canvas.paste(n, (0, y + 20))

    y += 20 + n.height + 8
    d.text((14, y), "分界对照：", font=f_sub, fill=MUTED)
    y += 20
    for r in diff_rows:
        d.text((14, y), r, font=f_row, fill=TEXT)
        y += 20
    save_img(canvas, out_png)
    return out_png


def make_card_overlay(out_png, source_img, cards, bg):
    """色卡照片的标注图（左原图 / 右标注）"""
    img_w = 880
    scale = img_w / source_img.shape[1]
    img_h = int(source_img.shape[0] * scale)
    left = Image.fromarray(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)).resize((img_w, img_h))
    right = left.copy()
    dr = ImageDraw.Draw(right)
    f1, f2 = font(15, True), font(14)
    for i, c in enumerate(cards):
        x, y = c["x"] * scale, c["y"] * scale
        w, h = c["w"] * scale, c["h"] * scale
        col = OK if c["ok"] else BAD
        dr.rectangle([x, y, x + w, y + h], outline=col, width=3)
        ty = y - 34 if y > 40 else y + h + 4
        dr.text((x, ty), f"#{i+1} H={c['hue']:.0f} S={c['s']:.0f} V={c['v']:.0f}", font=f1, fill=TEXT)
        dr.text((x, ty + 16), f"{c['truth']} (判:{c['current']})", font=f2, fill=col)

    gap, head, foot = 16, 40, 60
    canvas = Image.new("RGB", (img_w * 2 + gap, head + img_h + foot), BG)
    d = ImageDraw.Draw(canvas)
    d.text((12, 10), "原图", font=f1, fill=TEXT)
    d.text((img_w + gap + 12, 10), "实测标注（绿=当前判定正确，红=错判）", font=f1, fill=TEXT)
    canvas.paste(left, (0, head))
    canvas.paste(right, (img_w + gap, head))
    y, x = head + img_h + 8, 12
    for i, c in enumerate(cards):
        s = (f"#{i+1} H{c['hue']:.0f} S{c['s']:.0f}±{c['sd_s']:.0f} V{c['v']:.0f}±{c['sd_v']:.0f} "
             f"{c['truth']}{'' if c['ok'] else ' ←错判:' + c['current']}")
        d.text((x, y), s, font=f2, fill=OK if c["ok"] else BAD)
        x += d.textlength(s, font=f2) + 16
        if x > img_w * 2 + gap - 260:
            x = 12
            y += 20
    if bg:
        d.text((12, y + 4),
               f"背景饱和度 p50={bg['p50']:.0f} p90={bg['p90']:.0f} p99={bg['p99']:.0f}  |  "
               f"卡片实测 S 下限 {min((c['s'] for c in cards), default=0):.0f}",
               font=f2, fill=MUTED)
    save_img(canvas, out_png)
    return out_png


# ============================ 写回 colors.py ============================
def apply_bands(rows, colors_py):
    """把新的 HUE_BANDS 写回 colors.py（只替换 HUE_BANDS 那一段）"""
    lines = [f'    ("{n}",{fmt_h(lo)},{fmt_h(hi)}),' for n, lo, hi in rows]
    block = "HUE_BANDS = (\n" + "\n".join(lines) + "\n)"
    p = Path(colors_py)
    txt = p.read_text(encoding="utf-8")
    new, n = re.subn(r"HUE_BANDS = \(.*?\n\)", block, txt, count=1, flags=re.S)
    if not n:
        print(f"  未找到 HUE_BANDS 段：{p}")
        return False
    p.write_text(new, encoding="utf-8")
    print(f"  已更新：{p}")
    return True


# ============================ 主程序 ============================
def build_parser():
    ap = argparse.ArgumentParser(description="色卡参数更新：实测色卡 → 重算 HUE_BANDS → 重画色条图")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="色卡照片路径")
    ap.add_argument("--orig-bands", default=str(DEFAULT_BANDS_PNG), help="原色条图（对比用）")
    ap.add_argument("--out-dir", default=str(DESKTOP), help="产物输出目录（默认桌面）")
    ap.add_argument("--labels", default=None,
                    help="真实颜色名，按面积降序逗号分隔（不给则按当前表自动推断）")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="手动锁定某个颜色的 H 区间，格式 name=lo:hi，可重复")
    ap.add_argument("--sat-floor", type=int, default=100, help="找卡片的饱和度下限")
    ap.add_argument("--min-area-frac", type=float, default=0.002, help="最小面积占比")
    ap.add_argument("--width", type=int, default=1400, help="色条图宽度")
    ap.add_argument("--card", type=int, default=150, help="每色渐变图边长")
    ap.add_argument("--apply", action="store_true",
                    help="写回 colors.py 的 HUE_BANDS 并重画项目的 color_bands.png")
    ap.add_argument("--tag", default="", help="产物文件名前缀")
    return ap


def parse_sets(items):
    out = {}
    for it in items:
        try:
            name, rng = it.split("=", 1)
            lo, hi = rng.split(":", 1)
            out[name.strip()] = (float(lo), float(hi))
        except Exception:
            print(f"忽略无法解析的 --set：{it}（格式 name=lo:hi）")
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    src = Path(args.source)
    if not src.exists():
        print(f"找不到图片：{src}")
        return 1
    img = cv2.imread(str(src))
    if img is None:
        print(f"无法读取图片：{src}")
        return 1

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    overrides = parse_sets(args.sets)

    cards = find_cards(img, sat_floor=args.sat_floor, min_area_frac=args.min_area_frac)
    proposal = propose_bands(cards, labels, overrides)
    bg = background_stats(img, cards)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("_" + args.tag) if args.tag else ""
    new_bands_png = out_dir / f"color_bands_new{tag}.png"
    cmp_png = out_dir / f"color_bands_compare{tag}.png"
    overlay_png = out_dir / f"card_params_compare{tag}.png"
    rep = out_dir / f"card_params_report{tag}.md"
    js = out_dir / f"card_params_new{tag}.json"

    render_bands(proposal["rows"], str(new_bands_png), width=args.width, card=args.card)

    orig = Path(args.orig_bands)
    diff_rows = [f"{d['name']:8s}{COLOR_CN.get(d['name'], '')}  当前 [{d['old'][0]:.0f},{d['old'][1]:.0f})"
                 f"  ->  建议 [{d['new'][0]:.0f},{d['new'][1]:.0f})   "
                 f"实测中心 {d['center']:.1f}° ({d['reason']})" for d in proposal["diff"]]
    if not diff_rows:
        diff_rows = ["（无改动）"]
    if orig.exists():
        make_bands_compare(cmp_png, orig, new_bands_png, diff_rows,
                           f"原图 {orig.name}  |  改动 {len(proposal['diff'])} 处")
    make_card_overlay(overlay_png, img, cards, bg)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    js.write_text(json.dumps(dict(
        source=str(src), generated=now, cards=cards, background=bg,
        current_bands=[[n, lo, hi] for n, lo, hi in HUE_BANDS],
        proposed_bands=[[n, round(lo, 1), round(hi, 1)] for n, lo, hi in proposal["rows"]],
        changed=proposal["diff"],
        measured_centers={k: round(v, 1) for k, v in proposal["measured"].items()},
        issues=proposal["issues"],
        outputs=dict(new_bands=str(new_bands_png), compare=str(cmp_png),
                     overlay=str(overlay_png), report=str(rep)),
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# 色卡参数更新报告", "",
         f"- 色卡照片：`{src}`", f"- 原色条图：`{orig}`", f"- 时间：{now}",
         f"- 颜色名来源：{'手动指定 (--labels)' if labels else '按当前表自动推断（自证，需人工确认）'}",
         "", "## 逐卡实测", "",
         "| # | 面积(px) | 块数 | H | S | V | sdV | 真实 | 当前判定 | 结果 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cards):
        L.append(f"| {i+1} | {c['area']} | {c['parts']} | {c['hue']:.1f} | {c['s']:.0f}±{c['sd_s']:.0f} | "
                 f"{c['v']:.0f}±{c['sd_v']:.0f} | {c['sd_v']:.0f} | {c['truth']} | {c['current']} | "
                 f"{'OK' if c['ok'] else '错判'} |")
    L += ["", "## 颜色中心角：当前 vs 实测", "",
          "| 颜色 | 当前中心 | 实测中心 | 来源 |", "|---|---|---|---|"]
    for name in CIRCLE_ORDER:
        cur = current_center_deg(name)
        if cur is None:
            continue
        m = proposal["measured"].get(name)
        src_txt = "实测" if m is not None else ("手动" if name in overrides else "沿用当前")
        L.append(f"| {name} {COLOR_CN.get(name, '')} | {cur:.1f}° | "
                 f"{('%.1f°' % m) if m is not None else '—'} | {src_txt} |")
    L += ["", "## 建议后的 HUE_BANDS", "", "```python", "HUE_BANDS = ("]
    for n, lo, hi in proposal["rows"]:
        L.append(f'    ("{n}",{fmt_h(lo)},{fmt_h(hi)}),')
    L += ["```", ""]
    if proposal["diff"]:
        L += ["## 改动明细", ""]
        for d in proposal["diff"]:
            L.append(f"- **{d['name']}**：[{d['old'][0]:.0f},{d['old'][1]:.0f}) -> "
                     f"[{d['new'][0]:.0f},{d['new'][1]:.0f})（实测中心 {d['center']:.1f}°，{d['reason']}）")
        L.append("")
    if proposal["issues"]:
        L += ["## 表结构问题", ""] + [f"- {x}" for x in proposal["issues"]] + [""]
    if bg:
        L.append(f"背景饱和度：p50={bg['p50']:.0f} p90={bg['p90']:.0f} p99={bg['p99']:.0f}；"
                 f"卡片实测饱和度 {min((c['s'] for c in cards), default=0):.0f}~"
                 f"{max((c['s'] for c in cards), default=0):.0f}")
    L += ["", "## 产物", "",
          f"- 新色条：`{new_bands_png}`", f"- 并排对比：`{cmp_png}`",
          f"- 色卡标注：`{overlay_png}`", f"- 参数 JSON：`{js}`", ""]
    rep.write_text("\n".join(L), encoding="utf-8")

    print(f"色卡照片: {src}")
    print(f"找到 {len(cards)} 张色卡:")
    for i, c in enumerate(cards):
        print(f"  #{i+1} area={c['area']:6d} parts={c['parts']} H={c['hue']:5.1f} "
              f"S={c['s']:5.1f} V={c['v']:5.1f} sdV={c['sd_v']:4.1f} "
              f"真实={c['truth']:7s} 当前={c['current']:7s} {'OK' if c['ok'] else '错判'}")
    print("\n建议改动:")
    if proposal["diff"]:
        for d in proposal["diff"]:
            print(f"  {d['name']:8s} [{d['old'][0]:.0f},{d['old'][1]:.0f}) -> "
                  f"[{d['new'][0]:.0f},{d['new'][1]:.0f})  中心 {d['center']:.1f}° ({d['reason']})")
    else:
        print("  （无）")
    if proposal["issues"]:
        print("\n表结构问题:")
        for x in proposal["issues"]:
            print(f"  - {x}")
    print(f"\n新色条   : {new_bands_png}")
    if orig.exists():
        print(f"并排对比 : {cmp_png}")
    print(f"色卡标注 : {overlay_png}")
    print(f"报告     : {rep}")
    print(f"参数JSON : {js}")

    if args.apply:
        print("\n写回参数:")
        apply_bands(proposal["rows"], HERE / "colors.py")
        print("\n重画项目的 color_bands.png:")
        render_bands(proposal["rows"], str(DEFAULT_BANDS_PNG), width=args.width, card=args.card)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
