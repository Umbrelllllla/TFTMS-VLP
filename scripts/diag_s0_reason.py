"""debug：找出 dev 里 S=0（GT 不满足全部提到属性）的真实原因分布。"""
from __future__ import annotations
import os
from collections import Counter
from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import from_properties
from methods.baseline.matcher import MatchConfig, attribute_matches

_ATTR_KEYS = ["objectname","place","facing","distance","length","width","height","appearance","occlusion"]

def main():
    ds_root = default_data_root()
    img_dir = os.path.join(ds_root, "img", "train")
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "train")
    with open("data/splits/train.txt", encoding="utf-8") as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    anns = find_annotation_files(ann_dir)
    pairs = [(s, anns[s]) for s in stems if s in anns]
    cfg = MatchConfig()
    miss = Counter()
    examples = []
    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        for e in sample.expressions:
            if not e.matched_objects:
                continue
            gt = e.matched_objects[0]
            c = from_properties(e.public_properties)
            am = attribute_matches(c, gt, cfg)
            mentioned = [k for k in _ATTR_KEYS if getattr(c, k) is not None]
            fails = []
            for k in mentioned:
                key = "size" if k in ("length","width","height") else k
                if not am.get(key, False):
                    fails.append(key)
            if fails:
                miss.update(fails)
                if len(examples) < 25:
                    examples.append((stem, e.ann_id, mentioned, fails, e.public_properties, str(gt)))
    print("=== GT 不满足规则的属性分布（S=0 真实原因）===")
    for k, v in miss.most_common():
        print(f"  {k:14s} {v}")
    print("\n--- 示例 ---")
    for stem, ann_id, mentioned, fails, props, gt in examples:
        print(f"[{stem}/{ann_id}] mentioned={mentioned} fails={fails}")
        print(f"   props={props}")
        print(f"   gt={gt}")

if __name__ == "__main__":
    main()
