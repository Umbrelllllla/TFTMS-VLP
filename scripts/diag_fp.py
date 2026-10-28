"""全量 FP 归因：遍历 dev 全部表达式，统计 2508 个 FP 目标的属性命中结构。

对每条表达式：
1. 用与 eval_baseline 相同的规则（from_text + attribute_matches + score_object + predict_set）
2. 找出预测集中非 GT 的目标（FP）
3. 统计该目标每个描述属性是否命中（distance/place/size/facing/appearance/occlusion/objectname）
4. 汇总：GT 歧义 FP（全部属性命中）vs 规则误判 FP（有属性未命中）及未命中属性分布
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

from code.datasets.vlmod_dataset import iter_train
from methods.baseline.attributes import from_text
from methods.baseline.matcher import MatchConfig, attribute_matches, predict_scores, predict_set

proj = r"D:\TFTMS-VLP"
with open(os.path.join(proj, "data", "splits", "train.txt"), encoding="utf-8") as fh:
    dev_stems = set(line.strip() for line in fh if line.strip())

cfg = MatchConfig()  # rel_drop=0.2, pred_mode=rel 与 eval_baseline 一致

_ATTRS = ["objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]

total_fp = 0
gt_ambiguous = 0        # FP 目标所有描述属性都命中（GT 歧义）
rule_mismatch = 0       # FP 目标至少一个属性未命中（规则误判）
miss_counter = Counter()  # 未命中的属性（可多个）
desc_attr_counter = Counter()  # 描述本身提到的属性数分布
examples = []

for sample in iter_train():
    stem = os.path.splitext(os.path.basename(sample.image_path))[0]
    if stem not in dev_stems:
        continue
    pool = sample.candidate_pool
    for e in sample.expressions:
        c = from_text(e.description)
        gt_set = {o.to_tuple() for o in e.matched_objects}
        # 与 eval_baseline 完全同路径：predict_scores + predict_set
        scored = predict_scores(c, sample.candidate_pool, cfg)
        pred = predict_set(scored, cfg)
        pred_objs = list(pred)
        # FP = 预测中不在 GT 的目标
        for o in pred_objs:
            if o.to_tuple() in gt_set:
                continue
            total_fp += 1
            c_fresh = from_text(e.description)  # 干净约束，避免跨目标污染
            attrs = attribute_matches(c_fresh, o, cfg)
            mentioned = [k for k, v in c.__dict__.items() if v is not None and k != "objectname"]
            mentioned_attrs = set()
            if c.objectname:
                mentioned_attrs.add("objectname")
            for k in ("appearance", "place", "facing", "occlusion"):
                if getattr(c, k) is not None:
                    mentioned_attrs.add(k)
            if c.distance is not None:
                mentioned_attrs.add("distance")
            for k in ("length", "width", "height"):
                if getattr(c, k) is not None:
                    mentioned_attrs.add("size")
            hit = set()
            for k in _ATTRS:
                v = attrs.get(k)
                if isinstance(v, (bool, np.bool_)) and v:
                    hit.add(k)
            desc_attr_counter[len(mentioned_attrs)] += 1
            if hit >= mentioned_attrs:
                gt_ambiguous += 1
                if len(examples) < 8:
                    examples.append(("AMBIG", stem, e.description, o, mentioned_attrs, hit))
            else:
                rule_mismatch += 1
                for k in mentioned_attrs - hit:
                    miss_counter[k] += 1
                if len(examples) < 16:
                    examples.append(("MISMATCH", stem, e.description, o, mentioned_attrs, hit, sorted(mentioned_attrs - hit)))

print(f"dev 总 FP = {total_fp}")
print(f"  GT 歧义（全部属性命中，规则判对但 GT 不认）: {gt_ambiguous} ({gt_ambiguous / max(total_fp, 1):.1%})")
print(f"  规则误判（至少一个属性未命中仍被选）: {rule_mismatch} ({rule_mismatch / max(total_fp, 1):.1%})")
print(f"\n未命中属性分布（规则误判 FP 中）：")
for k, v in miss_counter.most_common():
    print(f"  {k:12s} {v} ({v / max(rule_mismatch, 1):.1%})")
print(f"\n描述提到的属性数分布：")
for k in sorted(desc_attr_counter):
    print(f"  {k} 个属性: {desc_attr_counter[k]}")

print("\n--- 示例 ---")
for ex in examples:
    if ex[0] == "AMBIG":
        _, stem, desc, o, mentioned, hit = ex
        print(f"[AMBIG] {stem}\n  desc={desc[:100]}\n  目标={o.obj_class} 距离={(o.x**2+o.y**2+o.z**2)**0.5:.1f} 位置=({o.x / o.z:.3f},{o.y / o.z:.3f}) hwl=({o.h:.2f},{o.w:.2f},{o.l:.2f}) ry={o.ry:.2f} 颜色={o.color}\n  提到={mentioned} 命中={hit}\n")
    else:
        _, stem, desc, o, mentioned, hit, miss = ex
        print(f"[MISMATCH] {stem}\n  desc={desc[:100]}\n  目标={o.obj_class} 距离={(o.x**2+o.y**2+o.z**2)**0.5:.1f} 位置=({o.x / o.z:.3f},{o.y / o.z:.3f}) hwl=({o.h:.2f},{o.w:.2f},{o.l:.2f}) ry={o.ry:.2f} 颜色={o.color}\n  提到={mentioned} 命中={hit} 未命中={miss}\n")
