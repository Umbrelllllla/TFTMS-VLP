"""对照：iter_train 路径 vs eval_baseline(load_split) 路径，找 FP 总数差异根源。"""

from __future__ import annotations

import os

import numpy as np

from code.datasets.vlmod_dataset import iter_train, load_train_sample, default_data_root
from methods.baseline.attributes import from_text
from methods.baseline.matcher import MatchConfig, predict_scores, predict_set

proj = r"D:\TFTMS-VLP"

# ---- 路径 A：iter_train ----
dev_stems = set()
with open(os.path.join(proj, "data", "splits", "train.txt"), encoding="utf-8") as fh:
    dev_stems = {l.strip() for l in fh if l.strip()}

cfg = MatchConfig()
fp_a = tp_a = fn_a = 0
n_a = 0
for sample in iter_train():
    stem = os.path.splitext(os.path.basename(sample.image_path))[0]
    if stem not in dev_stems:
        continue
    for e in sample.expressions:
        c = from_text(e.description)
        scored = predict_scores(c, sample.candidate_pool, cfg)
        if not scored:
            continue
        pred = predict_set(scored, cfg)
        gt = {o.to_tuple() for o in e.matched_objects}
        pt = {o.to_tuple() for o in pred}
        tp_a += len(pt & gt); fp_a += len(pt - gt); fn_a += len(gt - pt)
        n_a += 1
print(f"[A iter_train] 表达式={n_a} TP={tp_a} FP={fp_a} FN={fn_a}")

# ---- 路径 B：load_split + load_train_sample ----
from scripts.eval_baseline import load_split
split_txt = os.path.join(proj, "data", "splits", "train.txt")
pairs, img_dir = load_split(split_txt, default_data_root())
print(f"[B] pairs 数量={len(pairs)} img_dir={img_dir}")
fp_b = tp_b = fn_b = 0
n_b = 0
for stem, ann_path in pairs:
    sample = load_train_sample(ann_path, img_dir)
    if sample is None:
        continue
    for e in sample.expressions:
        c = from_text(e.description)
        scored = predict_scores(c, sample.candidate_pool, cfg)
        if not scored:
            continue
        pred = predict_set(scored, cfg)
        gt = {o.to_tuple() for o in e.matched_objects}
        pt = {o.to_tuple() for o in pred}
        tp_b += len(pt & gt); fp_b += len(pt - gt); fn_b += len(gt - pt)
        n_b += 1
print(f"[B load_split] 表达式={n_b} TP={tp_b} FP={fp_b} FN={fn_b}")
