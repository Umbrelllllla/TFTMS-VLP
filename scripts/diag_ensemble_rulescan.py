"""决策级 E 策略：rule_rd × fus_rd 联合扫描（v1 BCE 融合头最优 ckpt）。"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from methods.baseline.attributes import from_text
from methods.baseline.geom_features import continuous_features
from methods.baseline.matcher import MatchConfig, attribute_matches, predict_scores, score_object
from scripts.diag_ensemble import load_clip, build_features, collect
from scripts.train_fusion import FusionHead

proj = r"D:\TFTMS-VLP"
run_dir = os.path.join(proj, "runs")

ckpt = torch.load(os.path.join(run_dir, "fusion_head.pt"), map_location="cpu", weights_only=False)
model = FusionHead(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
feat_mean = ckpt.get("feat_mean")
feat_std = ckpt.get("feat_std")
if feat_mean is None:
    feat_mean = np.zeros(ckpt["in_dim"], dtype=np.float32)
    feat_std = np.ones(ckpt["in_dim"], dtype=np.float32)

img, txt, img_idx, txt_idx = load_clip("train", run_dir)
dev = collect("train.txt", ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx)
val = collect("val.txt", ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx)


def run(exprs, rule_rd, fus_rd):
    TP = FP = FN = 0
    for rule_vec, probs, gt in exprs:
        if len(rule_vec) == 0:
            continue
        top_r = max(rule_vec)
        a = {i for i, s in enumerate(rule_vec) if s >= top_r - rule_rd}
        if not a:
            pred = set()
        else:
            sub_top = max(probs[i] for i in a)
            pred = {i for i in a if probs[i] >= sub_top - fus_rd}
        TP += len(pred & gt); FP += len(pred - gt); FN += len(gt - pred)
    p = TP / (TP + FP) if TP + FP else 0
    r = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return p, r, f1, FP, FN


for split, exprs in (("dev", dev), ("val", val)):
    print(f"===== {split} =====")
    for rr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        best = None
        for fr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            p, r, f1, fp, fn = run(exprs, rr, fr)
            if best is None or f1 > best[2]:
                best = (fr, p, r, f1, fp, fn)
        print(f"rule_rd={rr:.2f}  最优 fus_rd={best[0]:.2f}  P={best[1]:.4f} R={best[2]:.4f} "
              f"F1={best[3]:.4f} FP={best[4]} FN={best[5]}")
