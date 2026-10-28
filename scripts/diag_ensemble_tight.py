"""收紧版 ensemble 网格扫描：rule_rd × fus_rd 全网格，找 FP 更少、R 不掉的配置。

线上教训（2026-10-24）：测试集每描述多选 0.45 目标（FP 407/900 描述），第 1 名仅 0.004。
目标：dev/val 上 FP/描述 ≤ 0.10（对齐线上 ~90 FP），R ≥ 0.99，F1 尽量高。
"""

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
    fp_per = FP / n
    print(f"{label:<26} rule={rule_rd} fus={fus_rd:<4} P={p:.4f} R={r:.4f} F1={f1:.4f} "
          f"FP={FP:5d} FN={FN:4d} FP/描述={fp_per:.3f} Acc@1={acc / n:.4f}")


print("===== dev（9597 表达式） =====")
for rr in [0.10, 0.15, 0.20]:
    for fr in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        run("dev", dev, rr, fr)
print("\n===== val（2400 表达式） =====")
for rr in [0.10, 0.15, 0.20]:
    for fr in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        run("val", val, rr, fr)
