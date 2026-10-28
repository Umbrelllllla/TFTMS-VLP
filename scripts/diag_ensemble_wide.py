"""E 策略（规则门+融合头）宽窗口扫描：fus_rd 0.35~0.80 找 F1 峰值。"""

from __future__ import annotations

import os

import numpy as np
import torch

from scripts.diag_ensemble import load_clip, collect
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


def run(label, exprs, rule_rd, fus_rd):
    TP = FP = FN = acc = 0
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
        tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
        TP += tp; FP += fp; FN += fn
        if pred == gt:
            acc += 1
    n = len(exprs)
    p = TP / (TP + FP) if TP + FP else 0
    r = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    print(f"fus_rd={fus_rd:<5} P={p:.4f} R={r:.4f} F1={f1:.4f} FP={FP:6d} FN={FN:6d} Acc@1={acc / n:.4f}")


print("===== dev wide =====")
for fr in [0.35, 0.40, 0.50, 0.60, 0.80]:
    run("dev", dev, 0.2, fr)
print("===== val wide =====")
for fr in [0.35, 0.40, 0.50, 0.60, 0.80]:
    run("val", val, 0.2, fr)
