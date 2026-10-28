"""路线 B · 第 5 步：集成评测（融合头分数 → 官方口径指标）。

对 dev/val 的每条表达式：
1. 特征 = [clip_sim, rule_score, 7 属性命中标志, 4 连续几何误差]（13 维，与训练构造完全一致）
2. 融合头输出 0~1 置信度
3. 排序后按预测策略选"判定匹配"的目标集合（默认 rel：>= 最高分 - rel_drop）
4. 官方口径二分类统计 TP/FP/FN → Precision/Recall/F1，另算 Acc@1

对比对象：规则 baseline（dev 文本 F1 96.65 / val 96.72，见笔记第 16 节）。

用法：
    uv run python -m scripts.eval_route_b --split dev [--rel-drop 0.1] [--errors 20]
    uv run python -m scripts.eval_route_b --split val
产物：控制台指标；runs/errors_routeB_<split>.jsonl（--errors N 时）
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from methods.baseline.attributes import from_text
from methods.baseline.geom_features import continuous_features
from methods.baseline.matcher import MatchConfig, attribute_matches, score_object

_ATTR_ORDER = ["objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]


def parse_args():
    ap = argparse.ArgumentParser(description="路线B融合头评测（官方口径）")
    ap.add_argument("--split", choices=["dev", "val"], required=True)
    ap.add_argument("--pred-mode", choices=["rel", "abs", "rel_abs"], default="rel")
    ap.add_argument("--rel-drop", type=float, default=0.1)
    ap.add_argument("--abs-thr", type=float, default=0.5)
    ap.add_argument("--errors", type=int, default=0, help="输出 N 条错误案例")
    return ap.parse_args()


def load_clip(split: str, run_dir: str):
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


def predict_set(scores, pred_mode, rel_drop, abs_thr):
    """按分数选判定匹配集合，返回候选索引列表（与规则版 predict_set 同构）。"""
    if not scores:
        return []
    top = max(scores)
    if pred_mode == "rel":
        return [i for i, s in enumerate(scores) if s >= top - rel_drop]
    if pred_mode == "rel_abs":
        return [i for i, s in enumerate(scores) if s >= top - rel_drop and s >= abs_thr]
    return [i for i, s in enumerate(scores) if s >= abs_thr]


def main() -> None:
    args = parse_args()
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(proj, "runs")

    ckpt = torch.load(os.path.join(run_dir, "fusion_head.pt"),
                      map_location="cpu", weights_only=False)  # 自己训练的可信 ckpt（含 numpy 标准化参数）
    from scripts.train_fusion import FusionHead
    model = FusionHead(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    feat_mean = ckpt.get("feat_mean")
    feat_std = ckpt.get("feat_std")
    if feat_mean is None:
        feat_mean = np.zeros(ckpt["in_dim"], dtype=np.float32)
        feat_std = np.ones(ckpt["in_dim"], dtype=np.float32)
    print(f"融合头已加载（in_dim={ckpt['in_dim']}, hidden={ckpt['hidden']}）", flush=True)

    img, txt, img_idx, txt_idx = load_clip("train", run_dir)  # 缓存按 train 存；dev/val 由划分文件过滤
    with open(os.path.join(proj, "data", "splits", "train.txt"), encoding="utf-8") as fh:
        dev_stems = set(line.strip() for line in fh if line.strip())
    with open(os.path.join(proj, "data", "splits", "val.txt"), encoding="utf-8") as fh:
        val_stems = set(line.strip() for line in fh if line.strip())
    target = dev_stems if args.split == "dev" else val_stems

    img_by_stem, txt_by_stem = {}, {}
    for i, m in enumerate(img_idx):
        img_by_stem.setdefault(m["stem"], []).append(i)
    for i, m in enumerate(txt_idx):
        txt_by_stem.setdefault(m["stem"], []).append(i)

    from code.datasets.vlmod_dataset import iter_train
    cfg = MatchConfig()

    tp = fp = fn = 0
    n_expr = 0
    empty_gt = 0
    acc1_hit = 0
    errors = []

    with torch.inference_mode():
        for sample in iter_train():
            stem = os.path.splitext(os.path.basename(sample.image_path))[0]
            if stem not in target:
                continue
            if stem not in img_by_stem or stem not in txt_by_stem:
                continue
            pool = sample.candidate_pool
            img_rows = img_by_stem[stem]
            txt_rows = txt_by_stem[stem]
            if len(img_rows) != len(pool):
                raise ValueError(f"{stem}: 缓存目标数 {len(img_rows)} != 候选池 {len(pool)}")

            for trow in txt_rows:
                tmeta = txt_idx[trow]
                expr = sample.expressions[tmeta["expr_i"]]
                desc = expr.description
                gt_set = set(tmeta["gt_pool_idx"])
                n_expr += 1
                if not gt_set:
                    empty_gt += 1
                c = from_text(desc)
                tvec = txt[trow]
                # 逐候选：特征 → 融合头概率
                feats, probs = [], []
                for p, o in enumerate(pool):
                    f = build_features(tvec, img[img_rows[p]], c, o, cfg)
                    if ckpt["in_dim"] == 8:
                        f = np.delete(f, 0)  # 消融版模型：特征去掉 clip_sim 列
                    feats.append(f)
                    fx = (f - feat_mean) / feat_std
                    probs.append(float(model(torch.from_numpy(fx[None, :]))[0, 0]))
                pred = set(predict_set(probs, args.pred_mode, args.rel_drop, args.abs_thr))
                # 官方口径二分类
                tp += len(pred & gt_set)
                fp += len(pred - gt_set)
                fn += len(gt_set - pred)
                if pred == gt_set:
                    acc1_hit += 1
                if args.errors and len(errors) < args.errors and (fp or fn):
                    errors.append({
                        "stem": stem, "desc": desc, "gt_idx": sorted(gt_set), "pred_idx": sorted(pred),
                        "probs": [round(p, 4) for p in probs],
                        "rule_scores": [round(float(score_object(c, o, cfg)), 3) for o in pool],
                        "fp": sorted(pred - gt_set), "fn": sorted(gt_set - pred),
                    })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\n===== 路线B（规则几何 + CLIP语义 + 自训融合头）· {args.split} =====", flush=True)
    print(f"表达式 {n_expr}（空GT {empty_gt}）  TP {tp}  FP {fp}  FN {fn}", flush=True)
    print(f"Precision {precision:.4f}  Recall {recall:.4f}  F1 {f1:.4f}  Acc@1 {acc1_hit / n_expr:.4f}", flush=True)
    print(f"[对比] 规则 baseline（笔记第13节）: dev F1 94.09 / val 93.75", flush=True)

    if errors:
        with open(os.path.join(run_dir, f"errors_routeB_{args.split}.jsonl"), "w", encoding="utf-8") as fh:
            for e in errors:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"错误案例已写 runs/errors_routeB_{args.split}.jsonl（{len(errors)} 条）", flush=True)


if __name__ == "__main__":
    main()
