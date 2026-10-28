"""logits 窗口版 E 策略扫描：用 sigmoid 前 logits 做门内 rel 窗口（概率被压平，logits 保留差异）。

线上教训：融合头概率在规则门 A 内饱和（0.91~0.92 差 0.008），概率窗口砍不动爆炸多选。
logits = -log(1/p - 1)，差异放大 ~10 倍，rel 窗口能真实区分 A 内目标。
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


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run(label, exprs, rule_rd, lg_rd):
    TP = FP = FN = acc = 0
    for rule_vec, probs, gt in exprs:
        if len(rule_vec) == 0:
            continue
        top_r = max(rule_vec)
        a = {i for i, s in enumerate(rule_vec) if s >= top_r - rule_rd}
        if not a:
            pred = set()
        else:
            lg = to_logit(probs)
            sub_top = float(lg[max(a)])
            pred = {i for i in a if lg[i] >= sub_top - lg_rd}
        tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
        TP += tp; FP += fp; FN += fn
        if pred == gt:
            acc += 1
    n = len(exprs)
    p = TP / (TP + FP) if TP + FP else 0
    r = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    print(f"{label:<26} rule={rule_rd} lg={lg_rd:<4} P={p:.4f} R={r:.4f} F1={f1:.4f} "
          f"FP={FP:5d} FN={FN:4d} FP/描述={FP / n:.3f} Acc@1={acc / n:.4f}")


print("===== dev（9597） logits 窗口扫描 =====")
for lg in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    run("dev", dev, 0.2, lg)
print("\n===== val（2400） logits 窗口扫描 =====")
for lg in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    run("val", val, 0.2, lg)
