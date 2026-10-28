"""深挖 S=0：描述连 GT 自身都不满足时，appearance(颜色) 为什么没对上。

这是'我们规则没界定清楚'最直接的证据来源。找出 public_properties 里 appearance 的
取值与 GT 实际 color 的配对，看是系统性的（可修：同义词/复合色/标签不一致）还是偶发噪声。
"""

from __future__ import annotations

import os
from collections import Counter

from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import from_properties, _split_kv


def main():
    ds_root = default_data_root()
    img_dir = os.path.join(ds_root, "img", "train")
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "train")
    with open("data/splits/train.txt", encoding="utf-8") as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    anns = find_annotation_files(ann_dir)
    pairs = [(s, anns[s]) for s in stems if s in anns]

    prop_color_vs_gt = Counter()   # (prop_appearance_value, gt_color) -> n
    prop_colors_seen = Counter()
    examples = []

    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        for e in sample.expressions:
            if not e.matched_objects:
                continue
            gt = e.matched_objects[0]
            # 找到 appearance 属性值
            app = None
            for s in e.public_properties:
                k, v = _split_kv(s)
                if k == "appearance":
                    app = v
            if app is None:
                continue
            prop_colors_seen[app.lower()] += 1
            gt_col = gt.color.lower()
            # 简单等值判断（同我们匹配器）
            if app.lower() != gt_col:
                prop_color_vs_gt[(app.lower(), gt_col)] += 1
                if len(examples) < 40:
                    examples.append((stem, e.ann_id, app, gt_col, gt.color))

    print("=== 描述中 appearance 取值频次（全量 dev）===")
    for k, v in prop_colors_seen.most_common():
        print(f"  {k:16s} {v}")

    print("\n=== appearance(描述) != GT.color 的配对频次（规则判定不匹配）===")
    total_mismatch = sum(prop_color_vs_gt.values())
    print(f"  不匹配总次数: {total_mismatch}")
    for (app, gt_col), n in prop_color_vs_gt.most_common(40):
        print(f"  prop={app:16s} gt_color={gt_col:16s} n={n}")

    print("\n--- 示例 ---")
    for stem, ann_id, app, gt_col, gt_raw in examples[:25]:
        print(f"  [{stem}/{ann_id}] prop={app!r}  gt.color={gt_raw!r}  (lower: {gt_col})")


if __name__ == "__main__":
    main()
