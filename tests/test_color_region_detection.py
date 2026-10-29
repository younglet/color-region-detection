"""color_region_detection 包的测试：颜色区间 / 三个百分比分档 / 三张图回归 / GUI / 性能"""
import os
import sys

import cv2
import numpy as np

# 让测试可以从任意工作目录运行：把源码目录（src/）塞进 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # color-region-detection/（项目根）
SRC = os.path.join(ROOT, "src")       # color-region-detection/src/（源码包所在）
DATA = os.path.join(HERE, "data")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import color_region_detection.cli as CR                                      # noqa: E402
import color_region_detection.colors as COL                                 # noqa: E402
import color_region_detection.ui as UI                                     # noqa: E402
from color_region_detection.colors import DETECTABLE, HUE_BANDS, classify   # noqa: E402
from color_region_detection.detect import Params, detect_scaled, process, region_sv  # noqa: E402

ok = True


def check(label, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"{label}: {'PASS' if cond else 'FAIL'} {extra}")


def img_path(name):
    return os.path.join(DATA, name)


def hit(blocks, cards):
    return sum(1 for cx, cy in cards.values()
               if any(x <= cx <= x + w and y <= cy <= y + h
                      for _n, x, y, w, h, *_ in blocks))


CARDS = {"yellow_top": (259, 174), "blue_top": (374, 167),
         "green_top": (482, 159), "red_left": (150, 323),
         "yellow_bot": (270, 310), "green_right": (488, 287),
         "blue_br": (477, 364)}
PROC = (320, 240)
IMGS = ("photo_test.png", "purity_test.png", "test.png")

# ============ 1. 颜色区间表 ============
check("1a. bands ascending & contiguous",
      all(HUE_BANDS[i][2] == HUE_BANDS[i + 1][1] for i in range(len(HUE_BANDS) - 1))
      and HUE_BANDS[0][1] == 0,
      " -> ".join(f"{n}[{lo},{hi})" for n, lo, hi in HUE_BANDS))
check("1b. last band wraps to 180", HUE_BANDS[-1][2] == 180)
check("1c. every detectable colour present",
      {n for n, _, _ in HUE_BANDS} >= set(DETECTABLE))
check("1d. no achromatic in DETECTABLE",
      not ({"black", "white", "gray"} & set(DETECTABLE)), str(len(DETECTABLE)))

# 绿卡必须落在 green 段（实测 H≈72）
check("1e. H=72 (weak-light green) -> green",
      COL.name_of(classify(np.uint8([[[72, 200, 120]]]), 0)[0, 0]) == "green",
      COL.name_of(classify(np.uint8([[[72, 200, 120]]]), 0)[0, 0]))
check("1f. H=100 (weak-light blue) -> blue",
      COL.name_of(classify(np.uint8([[[100, 200, 120]]]), 0)[0, 0]) == "blue")
for h_exp, want in [(0, "red"), (15, "orange"), (30, "yellow"), (60, "green"),
                    (90, "cyan"), (120, "blue"), (135, "purple"), (150, "pink"), (175, "red")]:
    got = COL.name_of(classify(np.uint8([[[h_exp, 200, 120]]]), 0)[0, 0])
    check(f"1g. canonical H={h_exp} -> {want}", got == want, f"got {got}")

# 低饱和 / 过暗 -> 0
check("1h. low saturation -> 0 (background)",
      classify(np.uint8([[[72, 50, 120]]]), 110)[0, 0] == 0)
check("1i. too dark -> 0 (background)",
      classify(np.uint8([[[72, 200, 20]]]), 110)[0, 0] == 0)

# 标定工具产出的分段表必须合法（连续、覆盖 0~180、无重叠）
import color_region_detection.calibrate as CAL                 # noqa: E402

_proposed = CAL.propose_bands([], None, None)
check("1j. calibrate 产出的分段表连续且覆盖 0~180",
      not _proposed["issues"] and _proposed["rows"][0][1] == 0
      and _proposed["rows"][-1][2] == 180,
      str(_proposed["issues"]))
check("1k. calibrate 保持 9 段结构（red 回绕两段）",
      len(_proposed["rows"]) == len(HUE_BANDS)
      and [r[0] for r in _proposed["rows"]] == [r[0] for r in HUE_BANDS],
      str([r[0] for r in _proposed["rows"]]))

# ============ 2. 三个参数都是百分比 ============
p = Params()
check("2a. defaults 20% ~ 100%", p.min_size_pct == 20 and p.max_size_pct == 100)
check("2b. 240x320 frame: 20% of short side = 48px",
      p.min_px((320, 240)) == 48, str(p.min_px((320, 240))))
check("2c. max 100% = unlimited", p.max_px((320, 240)) == 0)
check("2d. max 50% of short side = 120px", Params(max_size_pct=50).max_px((320, 240)) == 120)
check("2e. min<=max enforced? (accept still works)",
      p.accept(60, 60, 2000, 48, 0) and not p.accept(30, 60, 1000, 48, 0))
check("2f. non-convex shape accepted (fill 0.57)", p.accept(80, 130, 5930, 48, 0))
check("2g. __post_init__ clamps",
      1 <= Params(min_size_pct=-9).min_size_pct <= 100
      and Params(max_size_pct=999).max_size_pct == 100
      and Params(purity=-5).purity == 0 and Params(min_fill=9).min_fill == 1.0)
check("2h. purity -> minS/sdV (50 -> 110/30)",
      p.min_s == 110 and abs(p.max_std_v - 30) < 1e-6, f"{p.min_s}/{p.max_std_v}")
check("2i. purity monotonic",
      Params(purity=90).min_s > Params(purity=10).min_s
      and Params(purity=90).max_std_v < Params(purity=10).max_std_v)
check("2j. region_sv(None) on empty mask",
      region_sv(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool)) is None)

# ============ 3. 真实照片：绿卡必须判成 green ============
img = cv2.imread(img_path("photo_test.png"))
bl, _ = detect_scaled(img, Params(), PROC)
print(f"\n  photo_test: {len(bl)} 块 -> {[(b[0], b[3], b[4]) for b in bl]}")
names = sorted({b[0] for b in bl})
# 默认 min=20%（48px）会过滤掉最小的那张卡：它在 proc 下只有 34x25px，
# 低于 20% 阈值（预期行为；放宽到 10% 可命中全部 7 张）
check("3a. min=20%: 6/7 (7th card is 34x25px, below 20% threshold)",
      hit(bl, CARDS) == 6, f"{hit(bl, CARDS)}/7")
check("3a2. min=10%: all 7 cards found",
      hit(detect_scaled(img, Params(min_size_pct=10), PROC)[0], CARDS) == 7)
check("3b. GREEN cards labelled green (not cyan)",
      sum(1 for b in bl if b[0] == "green") >= 2,
      f"green={sum(1 for b in bl if b[0]=='green')} cyan={sum(1 for b in bl if b[0]=='cyan')}")
check("3c. no achromatic", not [b for b in bl if b[0] not in DETECTABLE])
check("3d. not flooded", len(bl) <= 12, f"{len(bl)}")

# 边界扫描：绿卡在哪个边界下会丢
print("  green|cyan 边界扫描:")
small = cv2.resize(img, PROC, interpolation=cv2.INTER_AREA)
for b in (75, 80, 85, 90):
    COL.HUE_BANDS = (("red", 0, 7), ("orange", 7, 22), ("yellow", 22, 45),
                    ("green", 45, b), ("cyan", b, 97), ("blue", 97, 127),
                    ("purple", 127, 142), ("pink", 142, 165), ("red", 165, 180))
    bb, _ = detect_scaled(img, Params(), PROC)
    ng = sum(1 for q in bb if q[0] == "green")
    print(f"    边界={b} -> green {ng} 块, 总 {len(bb)} 块")
# 还原表后再断言
COL.HUE_BANDS = HUE_BANDS
check("3e. green/cyan boundary = 82 after restore", COL.HUE_BANDS[4][1] == 82,
      str(COL.HUE_BANDS[4]))

# ============ 4. 三个参数单调性 ============
print()
for pct in (5, 10, 20, 30, 50):
    bb, _ = detect_scaled(img, Params(min_size_pct=pct), PROC)
    print(f"  minSize%={pct:3d} ({Params(min_size_pct=pct).min_px(PROC):3d}px) -> "
          f"{len(bb):2d} 块, 命中 {hit(bb, CARDS)}/7")
for pct in (100, 60, 40, 30, 20):
    bb, _ = detect_scaled(img, Params(max_size_pct=pct), PROC)
    print(f"  maxSize%={pct:3d} ({Params(max_size_pct=pct).max_px(PROC) or 'MAX':>4}px) -> "
          f"{len(bb):2d} 块, 命中 {hit(bb, CARDS)}/7")
for lv in (0, 25, 50, 75, 100):
    bb, _ = detect_scaled(img, Params(purity=lv), PROC)
    print(f"  purity={lv:3d} (minS>={Params(purity=lv).min_s:3d} "
          f"sdV<={Params(purity=lv).max_std_v:.0f}) -> {len(bb):2d} 块, "
          f"命中 {hit(bb, CARDS)}/7")

check("4a. minSize monotonic",
      len(detect_scaled(img, Params(min_size_pct=5), PROC)[0])
      >= len(detect_scaled(img, Params(min_size_pct=50), PROC)[0]))
check("4b. maxSize monotonic",
      len(detect_scaled(img, Params(max_size_pct=100), PROC)[0])
      >= len(detect_scaled(img, Params(max_size_pct=20), PROC)[0]))
check("4c. purity monotonic",
      len(detect_scaled(img, Params(purity=0), PROC)[0])
      >= len(detect_scaled(img, Params(purity=100), PROC)[0]))
check("4d. purity=50 + min=10% finds all 7 cards",
      hit(detect_scaled(img, Params(purity=50, min_size_pct=10), PROC)[0], CARDS) == 7)
check("4e. green cards survive purity sweep",
      all(sum(1 for b in detect_scaled(img, Params(purity=lv), PROC)[0]
              if b[0] == "green") >= 2 for lv in (0, 25, 50)),
      "弱光绿卡的饱和度只有 133 左右，purity<=50 时应始终稳定识别")

# ============ 5. 三张图回归 ============
print()
for f in IMGS:
    im = cv2.imread(img_path(f))
    bb, _ = detect_scaled(im, Params(), (320, 0))
    ex = f"  命中{hit(bb, CARDS)}/7" if f == "photo_test.png" else ""
    print(f"  {f:18s} -> {len(bb):2d} 块 {sorted({b[0] for b in bb})}{ex}")
    check(f"5. no achromatic in {f}", not [b for b in bb if b[0] not in DETECTABLE])

b_pu, _ = detect_scaled(cv2.imread(img_path("purity_test.png")), Params(), (320, 0))
check("5a. distractors rejected", len(b_pu) <= 5, f"{len(b_pu)}")
b_sh, _ = detect_scaled(cv2.imread(img_path("test.png")), Params(), (320, 180))
# test.png 的区域多为细长条（如 189x51、129x30），短边低于 20% 阈值（36px），
# 默认参数下只剩大区域；放宽到 5% 才能验证多区域与非凸形状。
b_sh5, _ = detect_scaled(cv2.imread(img_path("test.png")), Params(min_size_pct=5), (320, 180))
check("5b. test.png detects multiple at min=5%", len(b_sh5) >= 6, f"{len(b_sh5)}")
check("5c. gourd (non-convex) still detected at min=5%",
      any(b[0] == "green" and b[6] < 0.75 for b in b_sh5),
      str([(b[0], b[3], b[4], round(b[6], 2)) for b in b_sh5 if b[6] < 0.75]))

# ============ 6. GUI：只有三条滑条 ============
calls = {"namedWindow": 0, "createTrackbar": 0, "imshow": 0, "setTrackbarPos": 0}
wait_seq = [30, ord('w'), 30, ord('q')]
state = {}
smoke_out = os.path.join(HERE, "gui_smoke.png")
CR.cv2.namedWindow = lambda *a, **k: calls.update(namedWindow=calls["namedWindow"] + 1)
CR.cv2.resizeWindow = lambda *a, **k: None
CR.cv2.createTrackbar = lambda l, w, pos, hi, cb=None: (
    calls.update(createTrackbar=calls["createTrackbar"] + 1), state.update({l: pos}))
CR.cv2.setTrackbarPos = lambda l, w, pos: (
    calls.update(setTrackbarPos=calls["setTrackbarPos"] + 1), state.update({l: pos}))
CR.cv2.imshow = lambda *a, **k: calls.update(imshow=calls["imshow"] + 1)
CR.cv2.waitKey = lambda d=0: (wait_seq.pop(0) if wait_seq else ord('q'))
CR.cv2.destroyAllWindows = lambda *a, **k: None

argv = sys.argv
sys.argv = ["color_region_detection", "--source", img_path("test.png"), "--out", smoke_out]
try:
    rc = CR.main()
finally:
    sys.argv = argv
check("6a. main() rc=0", rc == 0, f"rc={rc}")
check("6b. exactly 3 sliders", calls["createTrackbar"] == 3, f"{calls['createTrackbar']}")
check("6c. slider names", [s[0] for s in UI.SPECS]
      == ["minSize%", "maxSize%", "purity"])
check("6d. keyboard synced to slider", state.get("minSize%") == 21,
      f"minSize%={state.get('minSize%')}")
check("6e. frames rendered", calls["imshow"] >= 3, f"{calls['imshow']}")
check("6f. output written", os.path.exists(smoke_out))

# --print-bands
check("6g. --print-bands works",
      CR.main(["--print-bands"]) == 0)

# ============ 7. 性能 ============
print()


def bench(fn, n=60):
    fn()
    t0 = cv2.getTickCount()
    for _ in range(n):
        fn()
    return (cv2.getTickCount() - t0) / cv2.getTickFrequency() / n * 1000


for W in (240, 320, 480):
    H = int(W * img.shape[0] / img.shape[1]) // 2 * 2
    im = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    print(f"  {W}x{H:<4d} {bench(lambda: process(im, Params())):6.2f} ms")
check("7. 320x240 under 5ms", bench(lambda: process(
    cv2.resize(img, PROC, interpolation=cv2.INTER_AREA), Params())) < 5.0)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
