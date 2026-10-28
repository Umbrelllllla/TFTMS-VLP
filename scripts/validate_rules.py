"""专项验证：GT 注释的真实语义（问题 3/4 的实证）。

对 dev 集每条表达式，用自由文本解析出约束，再拿 GT 目标与约束对比，回答：
1. distance：GT 目标的欧氏距离与描述值的关系 —— 是"d ≤ v"（within 语义）
   还是"|d-v| ≤ tol·v"（对称区间）？tol 该取多少？
2. place：GT 目标是否满足 nx=x/z、ny=y/z 的符号规则？各 place 值准确率？
3. facing：GT 目标的朝向夹角分布（完整 3D vs 水平投影），45°/135° 阈值是否合理？
4. size：GT 目标的尺寸相对描述区间的贴合度，size_pad 该取多少？

用法：uv run python -m scripts.validate_rules
"""

from __future__ import annotations

import math
import os
import statistics

from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import from_text
from methods.baseline.matcher import euclid, _heading, _toward_camera

_MISSING = object()


def _angle_vertical(o) -> float:
    h = _heading(o)
    t = _toward_camera(o)
    dot = h[0]*t[0] + h[1]*t[1] + h[2]*t[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _angle_horizontal(o) -> float:
    h = (math.cos(o.ry), -math.sin(o.ry))
    n = math.sqrt(o.x*o.x + o.z*o.z)
    t = (-o.x/n, -o.z/n) if n else (0.0, -1.0)
    dot = h[0]*t[0] + h[1]*t[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def main() -> None:
    ds_root = default_data_root()
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "train")
    img_dir = os.path.join(ds_root, "img", "train")
    with open("data/splits/train.txt", encoding="utf-8") as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    anns = find_annotation_files(ann_dir)
    pairs = [(s, anns[s]) for s in stems if s in anns]

    dist_records = []   # (v, d_gt)
    place_stats = {}    # place -> [gt_nx_ok, gt_total]
    facing_vert = []    # GT 目标夹角（完整3D）
    facing_horiz = []   # GT 目标夹角（水平投影）
    size_records = []   # (lo, hi, val)
    n_expr = 0

    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        for e in sample.expressions:
            if not e.matched_objects:
                continue
            c = from_text(e.description)
            gts = e.matched_objects
            if c.distance is not None:
                for o in gts:
                    dist_records.append((c.distance, euclid(o)))
            if c.place:
                nx, ny = (o.x/o.z, o.y/o.z) if o.z else (0.0, 0.0)
                p = c.place.strip().lower()
                ok = {
                    "left": nx < 0, "right": nx > 0,
                    "top": ny < 0, "bottom": ny > 0,
                    "center": abs(nx) < 0.2 and abs(ny) < 0.2,
                    "top left": nx < 0 and ny < 0,
                    "top right": nx > 0 and ny < 0,
                    "bottom left": nx < 0 and ny > 0,
                    "bottom right": nx > 0 and ny > 0,
                }.get(p)
                if ok is not None:
                    st = place_stats.setdefault(p, [0, 0])
                    st[0] += 1 if ok else 0
                    st[1] += 1
            if c.facing:
                for o in gts:
                    facing_vert.append(_angle_vertical(o))
                    facing_horiz.append(_angle_horizontal(o))
            for word, key in (("length", "length"), ("width", "width"), ("height", "height")):
                r = getattr(c, key)
                if r:
                    for o in gts:
                        val = {"length": o.l, "width": o.w, "height": o.h}[word]
                        size_records.append((r[0], r[1], val))
            n_expr += 1

    print(f"表达式总数: {n_expr}\n")

    # ---- distance ----
    if dist_records:
        le = sum(1 for v, d in dist_records if d <= v)
        ratios = sorted(d / v for v, d in dist_records if v > 0)
        print(f"[distance] 共 {len(dist_records)} 条 GT 记录")
        print(f"  d <= v        : {le/len(dist_records)*100:.1f}%   (within 语义 = 距离不超过描述值)")
        for t in (0.2, 0.3, 0.5, 0.7, 1.0):
            cnt = sum(1 for v, d in dist_records if v*(1-t) <= d <= v*(1+t))
            print(f"  |d-v|<=tol·v  : tol={t:4.1f} -> {cnt/len(dist_records)*100:.1f}%")
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
            idx = min(len(ratios)-1, int(q*(len(ratios)-1)))
            print(f"  d/v 分位 {q*100:5.1f}% = {ratios[idx]:.3f}")

    # ---- place ----
    print("\n[place] GT 目标满足 nx/ny 符号规则的比例")
    for p, (ok, tot) in sorted(place_stats.items()):
        print(f"  {p:12s} {ok}/{tot} = {ok/tot*100:.1f}%")

    # ---- facing ----
    if facing_vert:
        def qs(xs):
            xs = sorted(xs)
            return ", ".join(f"{q*100:.0f}%={xs[min(len(xs)-1,int(q*(len(xs)-1)))]:.1f}" for q in (0.05, 0.5, 0.95))
        print(f"\n[facing] GT 目标夹角分布 (n={len(facing_vert)})")
        print(f"  完整3D向量   : {qs(facing_vert)}")
        print(f"  水平投影     : {qs(facing_horiz)}")
        print(f"  夹角<=45°(facing me)      : 完整3D {sum(1 for a in facing_vert if a<=45)/len(facing_vert)*100:.1f}% | 水平 {sum(1 for a in facing_horiz if a<=45)/len(facing_horiz)*100:.1f}%")
        print(f"  夹角>=135°(facing away)   : 完整3D {sum(1 for a in facing_vert if a>=135)/len(facing_vert)*100:.1f}% | 水平 {sum(1 for a in facing_horiz if a>=135)/len(facing_horiz)*100:.1f}%")

    # ---- size ----
    if size_records:
        in_ = sum(1 for lo, hi, v in size_records if lo <= v <= hi)
        need_pad = {}
        for pad in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2):
            cnt = sum(1 for lo, hi, v in size_records if lo-pad <= v <= hi+pad)
            need_pad[pad] = cnt / len(size_records) * 100
        print(f"\n[size] GT 目标尺寸在描述区间内的比例 (n={len(size_records)})")
        print(f"  严格区间内 : {in_/len(size_records)*100:.1f}%")
        for pad, pct in need_pad.items():
            print(f"  +pad {pad:4.2f}m : {pct:.1f}%")


if __name__ == "__main__":
    main()
