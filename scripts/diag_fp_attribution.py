"""FP 归因：规则 rel0.2 窗口选出的 FP 目标，错在哪个属性（连续化改进的优先级依据）。"""

from __future__ import annotations

import os
from collections import Counter

from methods.baseline.attributes import from_text
from methods.baseline.matcher import MatchConfig, attribute_matches, predict_scores, euclid
from scripts.eval_baseline import load_split, load_train_sample, default_data_root

cfg = MatchConfig()
proj = r"D:\TFTMS-VLP"
ds_root = default_data_root()

reason_cnt = Counter()      # 该 FP 缺的属性（按未命中属性归因）
fp_by_expr = Counter()      # 每条表达式的 FP 数分布
total_fp = 0
samples = []                # 典型案例（FP 数 >= 3 的表达式）

for split in ("dev",):
    split_txt = os.path.join(proj, "data", "splits", "train.txt" if split == "dev" else "val.txt")
    pairs, img_dir = load_split(split_txt, ds_root)
    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        pool = sample.candidate_pool
        for expr in sample.expressions:
            gt = {o.to_tuple() for o in expr.matched_objects}
            c = from_text(expr.description)
            scored = predict_scores(c, pool, cfg)
            if not scored:
                continue
            top = scored[0][1]
            pred = {o.to_tuple() for o, s in scored if s >= top - cfg.rel_drop}
            fp = pred - gt
            if not fp:
                continue
            total_fp += len(fp)
            fp_by_expr[len(fp)] += 1
            if len(fp) >= 3 and len(samples) < 8:
                sample_d = []
                for tup in sorted(fp):
                    o = next(x for x in pool if x.to_tuple() == tup)
                    m = attribute_matches(c, o, cfg)
                    missing = [k for k, v in m.items() if not v]
                    sample_d.append((o.obj_class, o.color, round(o.x, 1), round(o.y, 1), round(o.z, 1),
                                     round(euclid(o), 1), round(o.ry, 2), missing))
                samples.append((expr.description, top, [round(s, 2) for o, s in scored[:6]], sample_d))
            for tup in fp:
                o = next(x for x in pool if x.to_tuple() == tup)
                m = attribute_matches(c, o, cfg)
                missing = [k for k, v in m.items() if not v]
                reason_cnt["+".join(sorted(missing)) if missing else "全部命中(纯多选)"] += 1

print(f"FP 总数 {total_fp}")
print(f"FP/表达式分布: {dict(sorted(fp_by_expr.items()))}")
print("\nFP 缺失属性归因（按缺失属性组合）:")
for k, v in reason_cnt.most_common(15):
    print(f"  {k}: {v} ({v / total_fp:.1%})")

print("\n典型案例（FP>=3 的表达式）:")
for desc, top, scores, fp_list in samples:
    print(f"\n  描述: {desc}")
    print(f"  最高分={top} 前6分数={scores}")
    for o in fp_list[:4]:
        print(f"    FP {o[0]}/{o[1]} xyz=({o[2]},{o[3]},{o[4]}) dist={o[5]} ry={o[6]} 缺:{o[7]}")
