"""
colorbar.py — 渐变色条渲染（颜色分段参考图）

把 HUE_BANDS 画成一条 0~360° 的连续色相条，每种颜色下方挂一张该颜色区间的渐变图。
`calibrate.py` 用它重画更新后的色条。

    python -m color_region_detection.colorbar --out color_bands.png
"""
import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:                                   # 包内导入
    from .colors import COLOR_CN, HUE_BANDS
except ImportError:                    # 直接跑文件时补上 src 路径
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from color_region_detection.colors import COLOR_CN, HUE_BANDS

HUE_SPAN = 180          # OpenCV H 取值 0~179（真实色相 = H*2）

BG = (18, 20, 24)
TEXT = (236, 239, 244)
MUTED = (146, 153, 164)
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    p = FONT_B if bold else FONT
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


# ============================ 色条数据 ============================
def band_table(bands=None):
    """每个分段一条记录: name / lo / hi / deg_lo / deg_hi"""
    bands = HUE_BANDS if bands is None else bands
    return [dict(name=n, lo=lo, hi=hi, deg_lo=lo * 2, deg_hi=hi * 2) for n, lo, hi in bands]


def unique_bands(bands=None):
    """按出现顺序去重的颜色段（回绕的 red 合并成一条）"""
    merged, order = {}, []
    for b in band_table(bands):
        name = b["name"]
        if name not in merged:
            merged[name] = dict(name=name, spans=[], deg_lo=b["deg_lo"], deg_hi=b["deg_hi"])
            order.append(name)
        merged[name]["spans"].append((b["deg_lo"], b["deg_hi"]))
        merged[name]["deg_lo"] = min(merged[name]["deg_lo"], b["deg_lo"])
        merged[name]["deg_hi"] = max(merged[name]["deg_hi"], b["deg_hi"])
    out = []
    for n in order:
        m = merged[n]
        m["spans"] = sorted(m["spans"], key=lambda s: (0 if s[1] >= 360 else 1, s[0]))
        out.append(m)
    return out


def band_spans(bands, width):
    """每个分段在宽度 width 上的像素区间 [(name, x0, x1, lo, hi), ...]"""
    width = max(1, int(width))
    out = []
    for n, lo, hi in bands:
        x0 = int(round(lo / float(HUE_SPAN) * width))
        x1 = int(round(hi / float(HUE_SPAN) * width))
        out.append((n, x0, min(max(x1, x0 + 1), width), lo, hi))
    return out


def gradient(width, height):
    """色相渐变条（RGB uint8）。左端 0°(红)，右端 360°(红)"""
    width = max(2, int(width))
    x = np.arange(width, dtype=np.float32)
    h = np.rint(x * (HUE_SPAN - 1) / (width - 1)).astype(np.uint8)
    strip = np.empty((1, width, 3), np.uint8)
    strip[..., 0] = h
    strip[..., 1] = 255
    strip[..., 2] = 255
    rgb = cv2.cvtColor(strip, cv2.COLOR_HSV2RGB)
    return np.repeat(rgb, max(1, int(height)), axis=0)


def hue_rgb(hue_deg, s=255, v=255):
    """真实色相角度 -> (r, g, b)"""
    hsv = np.uint8([[[int(round(hue_deg / 2.0)) % HUE_SPAN, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0])


def swatch(band, size=256):
    """该颜色色相区间的渐变图（红是首尾两段）"""
    full = gradient(360, size)
    strip = np.hstack([full[:, lo:max(hi, lo + 1)] for lo, hi in band["spans"]])
    strip = cv2.resize(strip, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)


# ============================ 渲染 ============================
def render(bands=None, out="color_bands.png", width=1400, card=150):
    """画色条图并保存"""
    bands = HUE_BANDS if bands is None else bands
    bl = unique_bands(bands)

    W, margin = int(width), 36
    content = W - 2 * margin
    bar_y, bar_h = 104, 84
    tick_y = bar_y + bar_h + 8
    cards_y = tick_y + 34
    n, gap = len(bl), 14
    card = min(int(card), (content - gap * (n - 1)) // n)
    H = cards_y + card + 62 + 40

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_sub = font(30, True), font(15)
    f_band, f_tick = font(17, True), font(13)
    f_card, f_meta = font(19, True), font(13)

    d.text((margin, 26), "渐变色条 · 颜色分段参考", font=f_title, fill=TEXT)
    d.text((margin, 64), "基于HSV色彩空间的H值", font=f_sub, fill=MUTED)

    # ---- 色条 ----
    grad = Image.fromarray(gradient(content, bar_h))
    ov = Image.new("RGBA", (content, bar_h), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    spans = band_spans(bands, content)
    for _n, x0, x1, _lo, _hi in spans:
        od.line([(x0, 0), (x0, bar_h)], fill=(255, 255, 255, 160), width=2)
        od.line([(x1 - 1, 0), (x1 - 1, bar_h)], fill=(255, 255, 255, 160), width=2)
    img.paste(Image.alpha_composite(grad.convert("RGBA"), ov).convert("RGB"), (margin, bar_y))

    for b in bl:
        xs = [(margin + lo / 360.0 * content, margin + hi / 360.0 * content)
              for lo, hi in b["spans"]]
        k = int(np.argmax([e - s for s, e in xs]))
        label = f"{COLOR_CN.get(b['name'], b['name'])} {b['name']}"
        w = d.textbbox((0, 0), label, font=f_band)[2]
        if xs[k][1] - xs[k][0] >= w + 10:
            cx = (xs[k][0] + xs[k][1]) / 2
            d.text((cx - w / 2, bar_y + bar_h / 2 - 12), label, font=f_band, fill=(0, 0, 0))

    for _n, _x0, _x1, lo, hi in spans:
        for h in (lo, hi):
            x = margin + h / 180.0 * content
            d.line([(x, bar_y + bar_h), (x, bar_y + bar_h + 5)], fill=(120, 126, 136))
            t = f"{h * 2}°"
            w = d.textbbox((0, 0), t, font=f_tick)[2]
            d.text((min(max(x - w / 2, margin), W - margin - w), tick_y), t,
                   font=f_tick, fill=MUTED)

    # ---- 每色渐变图 ----
    for i, b in enumerate(bl):
        x = margin + i * (card + gap)
        cx = x + card / 2
        xs = [(margin + lo / 360.0 * content, margin + hi / 360.0 * content)
              for lo, hi in b["spans"]]
        k = int(np.argmin([abs((s + e) / 2 - cx) for s, e in xs]))
        bx = (xs[k][0] + xs[k][1]) / 2
        d.line([(bx, bar_y + bar_h + 6), (bx, bar_y + bar_h + 16),
                (cx, cards_y - 16), (cx, cards_y - 2)], fill=(92, 99, 110))
        d.ellipse([bx - 3, bar_y + bar_h + 3, bx + 3, bar_y + bar_h + 9],
                  fill=hue_rgb(b["spans"][0][0]))

        thumb = swatch(b, card)
        img.paste(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)), (int(x), cards_y))
        d.rectangle([x, cards_y, x + card, cards_y + card], outline=(70, 76, 86), width=2)

        name = f"{COLOR_CN.get(b['name'], b['name'])} {b['name']}"
        w = d.textbbox((0, 0), name, font=f_card)[2]
        d.text((cx - w / 2, cards_y + card + 12), name, font=f_card, fill=TEXT)
        rng = " / ".join(f"{lo}~{hi}°" for lo, hi in sorted(b["spans"]))
        t = f"H {rng}"
        w = d.textbbox((0, 0), t, font=f_meta)[2]
        d.text((cx - w / 2, cards_y + card + 38), t, font=f_meta, fill=MUTED)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        img.save(out)
    except OSError:                       # 文件被占用时先写临时文件再改名
        tmp = out.with_suffix(out.suffix + ".tmp")
        img.save(tmp)
        os.replace(tmp, out)
    print(f"已生成 {out}  ({W}x{H})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成渐变色条 + 每个颜色对应一张渐变图")
    ap.add_argument("--out", default="color_bands.png")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--card", type=int, default=150, help="渐变图边长")
    args = ap.parse_args(argv)
    render(HUE_BANDS, args.out, args.width, args.card)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
