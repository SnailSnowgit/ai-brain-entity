# -*- coding: utf-8 -*-
"""N-Omniglot 真实事件数据导出器（一次性转换工具，需要 pip install dv）。

从解压后的 dvs_background 目录读取 aedat4 + CSV 时间戳，把每个字符的
20 个书写样本切成独立事件段，按时间积分成 F 帧并降采样为 G×G 网格，
导出为纯 JSON（之后 nomniglot.py 零依赖加载，不再需要 dv/torch）。

用法:
    python export_nomniglot.py [--alphabet Latin] [--chars 13] [--samples 10]
        [--frames 4] [--grid 16] [--out data/nomniglot_latin.json]

数据: Li, Dong, Zhao, Zeng (2022) N-Omniglot, Scientific Data 9:746,
figshare doi:10.6084/m9.figshare.16821427 (CC BY 4.0)
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import types

# DAVIS346 相机参数
CAM_W, CAM_H = 346, 260
# BrainCog 预处理同款裁剪: y∈[4,254), x∈[54,304) -> 250×250 书写区
CROP_Y0, CROP_Y1, CROP_X0, CROP_X1 = 4, 254, 54, 304


def install_dv():
    """dv 包在 Python 3.12 下 import imp 失败，先打 shim 再导入"""
    if "imp" not in sys.modules:
        imp = types.ModuleType("imp")

        def find_module(name):
            if importlib.util.find_spec(name) is None:
                raise ImportError(name)

        imp.find_module = find_module
        sys.modules["imp"] = imp
    from dv import AedatFile
    return AedatFile


def load_segments(aedat_path, csv_path):
    """返回 20 段事件 [(t0, t1, [(t, x, y, p), ...]), ...]"""
    AedatFile = install_dv()
    with open(csv_path, newline="") as fp:
        rows = [r for r in csv.reader(fp) if r and r[0].isdigit()]
    bounds = [(int(r[1]), int(r[2])) for r in rows]
    ts, xs, ys, ps = [], [], [], []
    with AedatFile(aedat_path) as f:
        for e in f["events"]:
            ts.append(e.timestamp)
            xs.append(e.x)
            ys.append(e.y)
            ps.append(1 if e.polarity else -1)
    import bisect
    segments = []
    for t0, t1 in bounds:
        lo, hi = bisect.bisect_left(ts, t0), bisect.bisect_right(ts, t1)
        segments.append((t0, t1, list(zip(ts[lo:hi], xs[lo:hi],
                                          ys[lo:hi], ps[lo:hi]))))
    return segments


def integrate_frames(events, t0, t1, frames_num, grid):
    """把一段事件按时间等分为 F 帧，每帧落到 G×G 网格（ON-OFF 差值归一）"""
    frames = [[[0.0] * grid for _ in range(grid)] for _ in range(frames_num)]
    if not events:
        return frames
    span = max(t1 - t0, 1)
    gw = (CROP_X1 - CROP_X0) / grid
    gh = (CROP_Y1 - CROP_Y0) / grid
    for t, x, y, p in events:
        if not (CROP_X0 <= x < CROP_X1 and CROP_Y0 <= y < CROP_Y1):
            continue
        fi = min(int((t - t0) * frames_num / span), frames_num - 1)
        gx = min(int((x - CROP_X0) / gw), grid - 1)
        gy = min(int((y - CROP_Y0) / gh), grid - 1)
        frames[fi][gy][gx] += p  # ON=+1 OFF=-1，保留笔画方向信息
    for f in frames:
        peak = max((abs(v) for row in f for v in row), default=0.0)
        if peak > 0:
            for gy in range(grid):
                f[gy] = [round(v / peak, 3) for v in f[gy]]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nomniglot_raw/dvs_background_2")
    ap.add_argument("--alphabet", default="Latin")
    ap.add_argument("--chars", type=int, default=13)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--out", default="data/nomniglot_latin.json")
    args = ap.parse_args()

    alpha_dir = os.path.join(args.root, args.alphabet)
    char_dirs = sorted(d for d in os.listdir(alpha_dir)
                       if d.startswith("character"))[:args.chars]
    samples = []
    for ci, cdir in enumerate(char_dirs):
        cpath = os.path.join(alpha_dir, cdir)
        aedat = next(f for f in os.listdir(cpath) if f.endswith(".aedat4"))
        csvf = next(f for f in os.listdir(cpath) if f.endswith(".csv"))
        segs = load_segments(os.path.join(cpath, aedat),
                             os.path.join(cpath, csvf))
        for si, (t0, t1, ev) in enumerate(segs[:args.samples]):
            frames = integrate_frames(ev, t0, t1, args.frames, args.grid)
            samples.append({
                "class": ci,
                "alphabet": args.alphabet,
                "character": cdir,
                "sample": si,
                "n_events": len(ev),
                "duration_us": t1 - t0,
                "frames": frames,
            })
        print(f"{cdir}: {len(segs)} 段, 平均事件数 "
              f"{sum(len(s[2]) for s in segs)//max(len(segs),1)}")

    meta = {
        "dataset": "N-Omniglot",
        "source": "figshare doi:10.6084/m9.figshare.16821427 (CC BY 4.0)",
        "paper": "Li, Dong, Zhao, Zeng, Scientific Data 9:746 (2022)",
        "camera": "iniVation DAVIS346 346x260",
        "alphabet": args.alphabet,
        "num_classes": len(char_dirs),
        "samples_per_class": args.samples,
        "frames_per_sample": args.frames,
        "grid": args.grid,
        "encoding": "events binned by time into F frames; crop y[4:254] "
                    "x[54:304]; per-frame ON(+1)-OFF(-1) counts max-normalized",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump({"meta": meta, "samples": samples}, fp,
                  ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"导出 {len(samples)} 个样本 -> {args.out} ({size_mb:.2f}MB)")


if __name__ == "__main__":
    main()
