"""GT 典型性验证：命中同一属性的目标里，GT 的连续量是否比非 GT 更"贴近描述"。

如果 GT 在距离/位置/尺寸/朝向上显著更典型 → 连续化打分能把 GT 排前、破同分（值得做）；
如果分布重叠 → GT 选择是主观细节，连续化无区分度（不值得做）。
口径：dev 自由文本，命中该属性的目标内比较 GT vs 非 GT。
"""

from __future__ import annotations

import os
import numpy as np

from methods.baseline.attributes import from_text
from methods.baseline.geom_features import _heading_theta
from methods.baseline.matcher import MatchConfig, attribute_matches, euclid
from scripts.eval_baseline import load_split, load_train_sample, default_data_root

cfg = MatchConfig()
proj = r"D:\TFTMS-VLP"
ds_root = default_data_root()

# 每属性收集 (GT 连续量, 非GT连续量)
stats = {k: {"gt": [], "nongt": []} for k in ("distance", "place", "size", "facing")}

split_txt = os.path.join(proj, "data", "splits", "train.txt")
pairs, img_dir = load_split(split_txt, ds_root)
for stem, ann_path in pairs:
    sample = load_train_sample(ann_path, img_dir)
    if sample is None:
        continue
    pool = sample.candidate_pool
    for expr in sample.expressions:
        gt = {o.to_tuple() for o in expr.matched_objects}
        c = from_text(expr.description)
        for o in pool:
            is_gt = o.to_tuple() in gt
            m = attribute_matches(c, o, cfg)
            # distance：命中距离属性的目标，|d-v|（相对 v）
            if m.get("distance") and c.distance:
                d = euclid(o)
                v = c.distance
                stats["distance"]["gt" if is_gt else "nongt"].append(abs(d - v) / v)
            # place：命中位置属性的目标，描述方向的"强度"（越靠方向越强）
            if m.get("place") and c.place:
                nx, ny = (o.x / o.z, o.y / o.z) if o.z else (0, 0)
                p = c.place.strip().lower()
                if p == "left":
                    s = -nx
                elif p == "right":
                    s = nx
                elif p == "top":
                    s = -ny
                elif p == "bottom":
                    s = ny
                elif p == "center":
                    s = -(abs(nx) + abs(ny))
                elif p in ("top left", "top right", "bottom left", "bottom right"):
                    want_nx_neg = p.endswith("left"); want_ny_neg = p.startswith("top")
                    s = (-nx if want_nx_neg else nx) + (-ny if want_ny_neg else ny)
                else:
                    s = 0.0
                stats["place"]["gt" if is_gt else "nongt"].append(s)
            # size：命中尺寸属性的目标，相对区间中点的偏差
            if m.get("size"):
                devs = []
                for k, attr in (("length", "l"), ("width", "w"), ("height", "h")):
                    r = getattr(c, k)
                    if r is None:
                        continue
                    mid = (r[0] + r[1]) / 2
                    if mid <= 0:
                        continue
                    devs.append(abs(getattr(o, attr) - mid) / mid)
                if devs:
                    stats["size"]["gt" if is_gt else "nongt"].append(np.mean(devs))
            # facing：命中朝向属性的目标，与描述朝向的偏差角
            if m.get("facing") and c.facing:
                theta = _heading_theta(o, cfg)
                f = c.facing.strip().lower()
                if f.startswith("facing away"):
                    want = 180.0
                elif f.startswith("side facing"):
                    want = 90.0
                else:
                    want = 0.0
                stats["facing"]["gt" if is_gt else "nongt"].append(abs(theta - want))

print("属性         |  组别  |  n   |  均值  |  中位数 |  25% |  75%")
print("-" * 70)
for k, v in stats.items():
    for label in ("gt", "nongt"):
        arr = np.array(v[label])
        if len(arr) == 0:
            print(f"{k:<12} | {label:<6} | {0:>4} |  —    |   —    |  —   |  —  ")
            continue
        print(f"{k:<12} | {label:<6} | {len(arr):>4} | {arr.mean():.3f} | {np.median(arr):.3f} | "
              f"{np.percentile(arr, 25):.3f} | {np.percentile(arr, 75):.3f}")
    # 判别度：GT 均值相对非 GT 的偏移（距离/尺寸：越小越典型；place 越大越典型；facing 越小越典型）
    g, ng = np.array(v["gt"]), np.array(v["nongt"])
    if len(g) and len(ng):
        if k in ("place",):
            d = g.mean() - ng.mean()
            note = "GT 更大" if d > 0 else "GT 更小"
        else:
            d = ng.mean() - g.mean()
            note = "GT 更典型(偏差更小)" if d > 0 else "GT 更不典型(偏差更大)"
        print(f"            | 判别   |    | Δ={d:+.4f} {note}")
