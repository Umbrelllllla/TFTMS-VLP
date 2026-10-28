"""路线B 13 维融合头：rel_drop 扫描（一次加载模型/特征，循环阈值算官方口径）。"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from methods.baseline.attributes import from_text
from methods.baseline.geom_features import continuous_features
from methods.baseline.matcher import MatchConfig, attribute_matches, score_object
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


def run(split_txt, label, ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx, rel_drops):
    with open(os.path.join(proj, "data", "splits", split_txt), encoding="utf-8") as fh:
        stems = set(l.strip() for l in fh if l.strip())
    img_by_stem, txt_by_stem = {}, {}
    for i, m in enumerate(img_idx):
        img_by_stem.setdefault(m["stem"], []).append(i)
    for i, m in enumerate(txt_idx):
        txt_by_stem.setdefault(m["stem"], []).append(i)

    from code.datasets.vlmod_dataset import iter_train
    cfg = MatchConfig()
    exprs = []  # (probs, gt_set)
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
                probs = []
                for p, o in enumerate(pool):
                    f = build_features(tvec, img[img_rows[p]], c, o, cfg)
                    if ckpt["in_dim"] == 12:
                        f = np.delete(f, 0)  # drop-clip 消融版
                    fx = (f - feat_mean) / feat_std
                    probs.append(float(model(torch.from_numpy(fx[None, :]))[0, 0]))
                exprs.append((probs, gt_set))

    print(f"[{label}] 表达式={len(exprs)}")
    print(f"{'rel_drop':<10}{'P':<9}{'R':<9}{'F1':<9}{'FP':<7}{'FN':<7}{'Acc@1':<9}")
    for rd in rel_drops:
        TP = FP = FN = 0
        acc_hit = 0
        for probs, gt in exprs:
            if not probs:
                continue
            top = max(probs)
            pred = {i for i, s in enumerate(probs) if s >= top - rd}
            tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
            TP += tp; FP += fp; FN += fn
            if pred == gt:
                acc_hit += 1
        p = TP / (TP + FP) if TP + FP else 0
        r = TP / (TP + FN) if TP + FN else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"{rd:<10.3f}{p:<9.4f}{r:<9.4f}{f1:<9.4f}{FP:<7d}{FN:<7d}{acc_hit / len(exprs):<9.4f}")


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
rd_list = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
run("train.txt", "dev", ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx, rd_list)
run("val.txt", "val", ckpt, feat_mean, feat_std, model, img, txt, img_idx, txt_idx, rd_list)
