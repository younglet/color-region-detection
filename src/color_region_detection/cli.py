"""
color-region-detection/src/color_region_detection/cli.py — 命令行入口（实现）

界面只有三条滑条：
    minSize%   最小尺寸（画面短边的百分比）
    maxSize%   最大尺寸（画面短边的百分比，100 = 不限）
    purity     纯度最低需求 0~100

想改颜色判定区间：编辑 color-region-detection/src/color_region_detection/colors.py 里的 HUE_BANDS。

运行（用项目根目录的 main.py，或安装后 python -m color_region_detection）：
    python main.py                                   摄像头
    python main.py --source photo_test.png           图片
    python main.py --print-bands                     打印当前颜色区间表
"""
import argparse
import sys

import cv2

from .colors import DETECTABLE, print_bands          # noqa: E402
from .detect import Params, detect_scaled            # noqa: E402
from .ui import SPECS, Trackbars, draw, draw_bar, handle_key   # noqa: E402

WIN = "Color Region Detector"


def parse_size(t):
    t = str(t).lower()
    return tuple(int(x) for x in t.split("x")) if "x" in t else (int(t), 0)


def build_parser():
    ap = argparse.ArgumentParser(description="color_region_detection：3 个参数的精简色彩区域检测")
    ap.add_argument("--source", default="0", help="摄像头序号或图片/视频路径")
    ap.add_argument("--proc-size", default="320x240",
                    help="检测分辨率 WxH，只给宽度则按源比例算高度，0=原分辨率")
    ap.add_argument("--min-size", type=int, default=20,
                    help="最小尺寸 %%（画面短边的百分比 1~100）")
    ap.add_argument("--max-size", type=int, default=100,
                    help="最大尺寸 %%（画面短边的百分比 1~100，100=不限）")
    ap.add_argument("--purity", type=int, default=50,
                    help="纯度最低需求 0~100，越大越严")
    ap.add_argument("--min-fill", type=float, default=0.35,
                    help="内部固定：最小填充率（拦碎片，圆形 0.79/葫芦 0.57）")
    ap.add_argument("--keep-bg", action="store_true", help="保留背景")
    ap.add_argument("--only", default=None, help="只识别指定颜色，逗号分隔")
    ap.add_argument("--print-bands", action="store_true", help="打印颜色区间表后退出")
    ap.add_argument("--out")
    ap.add_argument("--no-show", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.print_bands:
        print_bands()
        return 0

    prm = Params(args.min_size, args.max_size, args.purity, args.min_fill)
    src = args.source

    still = cap = None
    if not src.isdigit() and src.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
        still = cv2.imread(src)
        if still is None:
            print(f"无法读取图片: {src}")
            return 1
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            print(f"无法打开源: {src}")
            return 1

    src_wh = (still.shape[1], still.shape[0]) if still is not None else (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640,
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480)
    pw, ph = parse_size(args.proc_size)
    if pw <= 0:
        proc = None
    elif ph <= 0:
        proc = (pw, max(2, int(round(pw * src_wh[1] / float(src_wh[0]))) // 2 * 2))
    else:
        proc = (pw, ph)

    proc_wh = proc or src_wh
    tb = None
    if not args.no_show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, src_wh[0], src_wh[1] + 44 + 30 * len(SPECS) + 20)
        tb = Trackbars(WIN, prm)

    fps, prev, idx = 0.0, None, 0
    disp, blocks = None, []
    while True:
        if still is not None:
            frame = still
        else:
            ret, frame = cap.read()
            if not ret:
                break
            now = cv2.getTickCount()
            if prev is not None:
                dt = (now - prev) / cv2.getTickFrequency()
                if dt > 0:
                    fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            prev = now
        idx += 1

        blocks, _ = detect_scaled(frame, prm, proc, not args.keep_bg)
        disp = draw(frame, blocks, single=False)
        disp = draw_bar(disp, prm, blocks, fps, proc or proc_wh, proc_wh)

        if args.no_show:
            if still is not None:
                break
            continue

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1 if still is None else 20) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            path = f"color_region_detection_capture_{idx:05d}.png"
            cv2.imwrite(path, disp)
            print(f"已保存 {path}")
        elif k != -1:
            handle_key(k, prm)
            if tb:
                tb.sync()

    if cap is not None:
        cap.release()
    if still is not None:
        out = args.out or "color_region_detection_result.png"
        cv2.imwrite(out, disp)
        print(f"检测分辨率 {proc_wh[0]}x{proc_wh[1]}  显示 {disp.shape[1]}x{disp.shape[0]}")
        print(f"最小尺寸 {prm.min_size_pct}% ({prm.min_px(proc_wh)}px)  "
              f"最大尺寸 {prm.max_size_pct}% ({prm.max_px(proc_wh) or '不限'}px)  "
              f"纯度 {prm.purity} (minS>={prm.min_s} sdV<={prm.max_std_v:.0f})")
        print(f"识别到 {len(blocks)} 个色彩区域 -> {out}")
        for n, x, y, w, h, a, f, cx, cy in blocks:
            print(f"  {n:8s} {w:4d}x{h:4d} area={a:7d} fill={f:.2f} @({x},{y})")
    if not args.no_show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
