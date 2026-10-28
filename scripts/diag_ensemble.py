"""决策级融合（ensemble）实验：规则裁判 + 融合头裁判，多种合并策略对比。

策略：
  A. 纯规则 rel 0.2            —— 复现规则 baseline
  B. 纯融合头 rel 0.06         —— 复现当前融合头
  C. 交集 A∩B                  —— 两个都选才选（严格）
  D. 并集 A∪B                  —— 任一选就选（宽松）
  E. 规则硬约束：只在规则集合 A 内按融合头概率 rel 窗口选
  F. rank 加权：两分数都转池内排名（0~1）再加权 α，rel 窗口选
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from methods.baseline.attributes import from_text
from methods.baseline.geom_features import continuous_features
from methods.baseline.matcher import MatchConfig, attribute_matches, predict_scores, predict_set, score_object
from scripts.train_fusion import FusionHead

proj = r"D:\TFTMS-VLP"
run_dir = os.path.join(proj, "runs")
_ATTR_ORDER = ["objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]


def load_clip(split, run_dir):
    img = np.load(os.path.join(run_dir, f"clip_img_{split}.npy"))
    txt = np.load(os.path.join(run_dir, f"clip_text_{split}.npy"))
    with open(os.path.join(run_dir, f"clip_img_{split}_idx.json"), encoding="utf-8") as fh:
        img_idx = json.load(fh)
    with open(os.path.join(run_dir, f"clip_text_{split}_idx.json"), encoding="utf-8") as fh:
        txt_idx = json.load(fh)
    return img, txt, img_idx, txt_idx


def build_features(tvec, ivec, c, o, cfg):
    clip_sim = float(np.dot(tvec, ivec))
    attr = attribute_matches(c, o, cfg)
    rule_score = score_object(c, o, cfg)
    flag = [1.0 if attr.get(k, False) else 0.0 for k in _ATTR_ORDER]
    cont = continuous_features(c, o, cfg)
    return np.asarray([clip_sim, rule_score, *flag, *cont], dtype=np.float32)


def collect(split_txt, ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx):
    """返回 [(rule_scored, probs, gt_set)]。"""
    with open(os.path.join(proj, "data", "splits", split_txt), encoding="utf-8") as fh:
        stems = set(l.strip() for l in fh if l.strip())
    img_by_stem, txt_by_stem = {}, {}
    for i, m in enumerate(img_idx):
        img_by_stem.setdefault(m["stem"], []).append(i)
    for i, m in enumerate(txt_idx):
        txt_by_stem.setdefault(m["stem"], []).append(i)
    from code.datasets.vlmod_dataset import iter_train
    cfg = MatchConfig()
    exprs = []
    with torch.inference_mode():
        for sample in iter_train():
            stem = os.path.splitext(os.path.basename(sample.image_path))[0]
            if stem not in stems:
                continue
            if stem not in img_by_stem or stem not in txt_by_stem:
                continue
            pool = sample.candidate_pool
            img_rows, txt_rows = img_by_stem[stem], txt_by_stem[stem]
            for trow in txt_rows:
                tmeta = txt_idx[trow]
                expr = sample.expressions[tmeta["expr_i"]]
                gt_set = set(tmeta["gt_pool_idx"])
                c = from_text(expr.description)
                tvec = txt[trow]
                rule_scored = predict_scores(c, pool, cfg)  # [(obj, score)] 降序
                probs = np.zeros(len(pool), dtype=np.float32)
                for p, o in enumerate(pool):
                    f = build_features(tvec, img[img_rows[p]], c, o, cfg)
                    fx = (f - feat_mean) / feat_std
                    probs[p] = float(model(torch.from_numpy(fx[None, :]))[0, 0])
                # 对齐：rule_scored 的对象顺序和 pool 不同，映射 obj→rule_score
                rule_map = {o.to_tuple(): s for o, s in rule_scored}
                rule_vec = np.array([rule_map.get(o.to_tuple(), 0.0) for o in pool], dtype=np.float32)
                exprs.append((rule_vec, probs, gt_set))
    return exprs


def run(label, exprs, select_fn):
    TP = FP = FN = 0
    acc_hit = 0
    for rule_vec, probs, gt in exprs:
        pred = select_fn(rule_vec, probs)
        tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
        TP += tp; FP += fp; FN += fn
        if pred == gt:
            acc_hit += 1
    n = len(exprs)
    p = TP / (TP + FP) if TP + FP else 0
    r = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    print(f"{label:<32} P={p:.4f} R={r:.4f} F1={f1:.4f} FP={FP:6d} FN={FN:6d} Acc@1={acc_hit / n:.4f}")


def sel_rule(rd=0.2):
    def f(rule_vec, probs):
        if len(rule_vec) == 0:
            return set()
        top = max(rule_vec)
        return {i for i, s in enumerate(rule_vec) if s >= top - rd}
    return f


def sel_fusion(rd=0.06):
    def f(rule_vec, probs):
        if len(probs) == 0:
            return set()
        top = max(probs)
        return {i for i, s in enumerate(probs) if s >= top - rd}
    return f


def sel_inter(rule_rd, fus_rd):
    def f(rule_vec, probs):
        a = sel_rule(rule_rd)(rule_vec, probs)
        b = sel_fusion(fus_rd)(rule_vec, probs)
        return a & b
    return f


def sel_union(rule_rd, fus_rd):
    def f(rule_vec, probs):
        a = sel_rule(rule_rd)(rule_vec, probs)
        b = sel_fusion(fus_rd)(rule_vec, probs)
        return a | b
    return f


def sel_rule_gate_fusion(rule_rd, fus_rd):
    """规则硬约束：只在规则集合 A 内按融合头概率 rel 窗口选。"""
    def f(rule_vec, probs):
        if len(rule_vec) == 0:
            return set()
        top_r = max(rule_vec)
        a = {i for i, s in enumerate(rule_vec) if s >= top_r - rule_rd}
        if not a:
            return set()
        sub_top = max(probs[i] for i in a)
        return {i for i in a if probs[i] >= sub_top - fus_rd}
    return f


def sel_rank_weighted(alpha, rd):
    """两分数都转池内排名（0~1），加权平均，rel 窗口选。"""
    def f(rule_vec, probs):
        n = len(rule_vec)
        if n == 0:
            return set()
        def ranks(v):
            order = np.argsort(-v)  # 降序
            r = np.empty(n, dtype=np.float32)
            r[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
            return r
        ens = alpha * ranks(rule_vec) + (1 - alpha) * ranks(probs)
        top = float(ens.max())
        return {int(i) for i in range(n) if ens[i] >= top - rd}
    return f


ckpt = torch.load(os.path.join(run_dir, "fusion_head.pt"), map_location="cpu", weights_only=False)
model = FusionHead(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
feat_mean = ckpt.get("feat_mean")
feat_std = ckpt.get("feat_std")
if feat_mean is None:
    feat_mean = np.zeros(ckpt["in_dim"], dtype=np.float32)
    feat_std = np.ones(ckpt["in_dim"], dtype=np.float32)
print(f"融合头 in_dim={ckpt['in_dim']}")

img, txt, img_idx, txt_idx = load_clip("train", run_dir)
dev = collect("train.txt", ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx)
val = collect("val.txt", ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx)

for name, exprs in (("dev", dev), ("val", val)):
    print(f"\n===== {name} =====")
    run("A 纯规则 rel0.2", exprs, sel_rule(0.2))
    run("B 纯融合头 rel0.06", exprs, sel_fusion(0.06))
    run("C 交集 规则0.2∩融合0.06", exprs, sel_inter(0.2, 0.06))
    run("C2 交集 规则0.2∩融合0.10", exprs, sel_inter(0.2, 0.10))
    run("D 并集 规则0.2∪融合0.06", exprs, sel_union(0.2, 0.06))
    run("E 规则门+融合 rel0.06", exprs, sel_rule_gate_fusion(0.2, 0.06))
    run("E2 规则门+融合 rel0.10", exprs, sel_rule_gate_fusion(0.2, 0.10))
    run("E3 规则门+融合 rel0.15", exprs, sel_rule_gate_fusion(0.2, 0.15))
    for a in (0.3, 0.5, 0.7):
        run(f"F rank加权 α={a} rel0.10", exprs, sel_rank_weighted(a, 0.10))
        run(f"F rank加权 α={a} rel0.15", exprs, sel_rank_weighted(a, 0.15))
