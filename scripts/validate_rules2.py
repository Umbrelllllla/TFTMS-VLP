"""补充验证：place 的 2D 框替代方案 + facing 按标签分布。

用法：uv run python -m scripts.validate_rules2
"""

from __future__ import annotations

import math
import os

from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import from_text
from methods.baseline.matcher import _heading, _toward_camera

IMG_W, IMG_H = 1920.0, 1080.0


def _angle_vertical(o) -> float:
    h = _heading(o)
    t = _toward_camera(o)
    dot = h[0]*t[0] + h[1]*t[1] + h[2]*t[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def main() -> None:
    ds_root = default_data_root()
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "train")
    img_dir = os.path.join(ds_root, "img", "train")
    with open("data/splits/train.txt", encoding="utf-8") as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    anns = find_annotation_files(ann_dir)
    pairs = [(s, anns[s]) for s in stems if s in anns]

    place_2d = {}    # place -> [ok, tot]  用 2D 框中心 vs 图像中心
    place_3d = {}    # 对照：nx/ny 符号
    facing_by_label = {}   # facing 值 -> 夹角列表

    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        for e in sample.expressions:
            if not e.matched_objects:
                continue
            c = from_text(e.description)
            for o in e.matched_objects:
                nx, ny = (o.x/o.z, o.y/o.z) if o.z else (0.0, 0.0)
                u = (o.x1 + o.x2) / 2.0 - IMG_W / 2.0
                v = (o.y1 + o.y2) / 2.0 - IMG_H / 2.0
                if c.place:
                    p = c.place.strip().lower()
                    ok3 = {
                        "left": nx < 0, "right": nx > 0,
                        "top": ny < 0, "bottom": ny > 0,
                        "center": abs(nx) < 0.2 and abs(ny) < 0.2,
                        "top left": nx < 0 and ny < 0, "top right": nx > 0 and ny < 0,
                        "bottom left": nx < 0 and ny > 0, "bottom right": nx > 0 and ny > 0,
                    }.get(p)
                    ok2 = {
                        "left": u < 0, "right": u > 0,
                        "top": v < 0, "bottom": v > 0,
                        "center": abs(u) < IMG_W*0.05 and abs(v) < IMG_H*0.05,
                        "top left": u < 0 and v < 0, "top right": u > 0 and v < 0,
                        "bottom left": u < 0 and v > 0, "bottom right": u > 0 and v > 0,
                    }.get(p)
                    if ok3 is not None:
                        st = place_3d.setdefault(p, [0, 0]); st[0] += ok3; st[1] += 1
                    if ok2 is not None:
                        st = place_2d.setdefault(p, [0, 0]); st[0] += ok2; st[1] += 1
                if c.facing:
                    facing_by_label.setdefault(c.facing, []).append(_angle_vertical(o))

    print("[place] 3D(nx/ny 符号) vs 2D(框中心 vs 图像中心) 对 GT 的命中率")
    all_p = set(place_3d) | set(place_2d)
    for p in sorted(all_p):
        a = place_3d.get(p, [0, 0]); b = place_2d.get(p, [0, 0])
        pa = a[0]/a[1]*100 if a[1] else 0
        pb = b[0]/b[1]*100 if b[1] else 0
        print(f"  {p:12s} 3D:{pa:5.1f}% ({a[0]}/{a[1]})   2D:{pb:5.1f}% ({b[0]}/{b[1]})")

    print("\n[facing] 按标签看 GT 夹角分布（完整3D）")
    for label, xs in facing_by_label.items():
        xs = sorted(xs)
        def q(qq):
            return xs[min(len(xs)-1, int(qq*(len(xs)-1)))]
        le45 = sum(1 for x in xs if x <= 45) / len(xs) * 100
        ge135 = sum(1 for x in xs if x >= 135) / len(xs) * 100
        print(f"  {label:22s} n={len(xs):5d}  5%={q(.05):.0f} 50%={q(.5):.0f} 95%={q(.95):.0f}  <=45°:{le45:.1f}%  >=135°:{ge135:.1f}%")


if __name__ == "__main__":
    main()
