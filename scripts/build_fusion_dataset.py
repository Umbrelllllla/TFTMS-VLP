"""路线 B · 第 3 步：构造融合头训练样本（正负样本对）。

思路（大白话）：训练集自带正确答案（label_3）。对每张图里的一条描述：
- 描述真正指向的目标（GT）= 正样本（标签 1）
- 同图里其他候选目标 = 负样本（标签 0）
每个"描述 × 候选目标"变成一个特征行，融合头（小 MLP）就从这些样本里学"两个分数怎么合"。

特征行（13 维）：
    [0] clip_sim     描述文本与候选裁剪图的 CLIP 余弦相似度（语义）
    [1] rule_score   现有规则 baseline 的加权命中分（几何）
    [2..8] 7 个属性命中标志：objectname/appearance/place/size/distance/facing/occlusion（0/1）
    [9..12] 4 个连续几何误差特征 v2：dist_err/size_dev/facing_delta/place_match（精确偏差，A 内可区分）
说明：属性标志用 from_text（自由文本）解析 + matcher 规则判定，和真实测试条件一致；
      连续特征由 methods/baseline/geom_features.py 构造，build 与 eval 共用，保证对齐。

采样（v2，2026-10-24 难负样本）：
    负样本只取"规则门 A 内非 GT 目标"（推理时融合头只在这批目标里选，训练分布必须一致）
    + 每描述 2 个随机负样本保底。全池负样本会稀释难样本，导致 A 内概率饱和无法排序。

对齐方式（关键）：
    CLIP 特征缓存的行号 = 该图 candidate_pool 的池内顺序（提取脚本按 pool 顺序写缓存），
    因此：表达式 expr_i 的文本特征行 = txt_by_stem[stem][expr_i]；
          池内第 p 个对象的图像特征行 = img_by_stem[stem][p]。
    完整 Rope3DObject 直接从数据集加载（规则打分需要 x/y/z/ry/h/w/l 等 3D 字段）。

抽样（用户拍板：不全量训练）：
    --max-images 默认 1000：从 dev 训练图（3199 张）里按固定 seed 随机抽 1000 张图。
    约 1000 图 × 3 表达式 × A 内约 5 候选 ≈ 1.5 万行样本，正样本约 3 千，足够训一个小 MLP。

用法：
    uv run python -m scripts.build_fusion_dataset --max-images 1000 --seed 42
产物：
    runs/fusion_train.npz    X: (N,13) float32, y: (N,) int8
    runs/fusion_train_meta.json  [{stem, expr_i, obj_i, desc, is_gt, clip_sim, rule_score}]
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from methods.baseline.attributes import from_text
from methods.baseline.geom_features import continuous_features
from methods.baseline.matcher import MatchConfig, attribute_matches, predict_scores, score_object

FEAT_DIM = 13
_ATTR_ORDER = ["objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]


def parse_args():
    ap = argparse.ArgumentParser(description="构造融合头训练样本")
    ap.add_argument("--max-images", type=int, default=1000,
                    help="从 dev 训练图里随机抽多少张（0=全部）")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_clip(split: str, run_dir: str):
    img = np.load(os.path.join(run_dir, f"clip_img_{split}.npy"))
    txt = np.load(os.path.join(run_dir, f"clip_text_{split}.npy"))
    with open(os.path.join(run_dir, f"clip_img_{split}_idx.json"), encoding="utf-8") as fh:
        img_idx = json.load(fh)
    with open(os.path.join(run_dir, f"clip_text_{split}_idx.json"), encoding="utf-8") as fh:
        txt_idx = json.load(fh)
    return img, txt, img_idx, txt_idx


def main() -> None:
    args = parse_args()
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(proj, "runs")
    img, txt, img_idx, txt_idx = load_clip("train", run_dir)

    with open(os.path.join(proj, "data", "splits", "train.txt"), encoding="utf-8") as fh:
        train_stems = set(line.strip() for line in fh if line.strip())

    img_by_stem, txt_by_stem = {}, {}
    for i, m in enumerate(img_idx):
        img_by_stem.setdefault(m["stem"], []).append(i)
    for i, m in enumerate(txt_idx):
        txt_by_stem.setdefault(m["stem"], []).append(i)

    # 抽样：固定 seed 从 dev 训练图里随机抽
    usable = sorted(train_stems & set(img_by_stem.keys()) & set(txt_by_stem.keys()))
    if args.max_images and args.max_images < len(usable):
        rng = random.Random(args.seed)
        chosen = set(rng.sample(usable, args.max_images))
        print(f"抽样：dev 训练图 {len(usable)} -> 抽 {args.max_images}（seed={args.seed}）", flush=True)
    else:
        chosen = set(usable)
        print(f"用全部可用 dev 训练图：{len(usable)}", flush=True)

    from code.datasets.vlmod_dataset import iter_train
    cfg = MatchConfig()
    X_rows, y_rows, meta, group_rows = [], [], [], []

    for sample in iter_train():
        stem = os.path.splitext(os.path.basename(sample.image_path))[0]
        if stem not in chosen:
            continue  # 非抽样图/val 图不参与训练
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
            c = from_text(desc)
            tvec = txt[trow]
            group_id = len(X_rows)  # 该描述的第一个样本行号（rank loss 分组用）

            # 全池负样本（v1 恢复，2026-10-24）：
            for p, o in enumerate(pool):
                ivec = img[img_rows[p]]
                clip_sim = float(np.dot(tvec, ivec))
                attr = attribute_matches(c, o, cfg)
                rule_score = score_object(c, o, cfg)
                flag = [1.0 if attr.get(k, False) else 0.0 for k in _ATTR_ORDER]
                cont = continuous_features(c, o, cfg)
                X_rows.append([clip_sim, rule_score, *flag, *cont])
                is_gt = 1 if p in gt_set else 0
                y_rows.append(is_gt)
                group_rows.append(group_id)  # 每个样本行都记所属描述组（rank loss 分组用）
                meta.append({
                    "stem": stem, "expr_i": tmeta["expr_i"], "obj_i": p,
                    "desc": desc, "is_gt": bool(is_gt),
                    "clip_sim": round(clip_sim, 4), "rule_score": round(rule_score, 4),
                })

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int8)
    group = np.asarray(group_rows, dtype=np.int32)
    np.savez(os.path.join(run_dir, "fusion_train.npz"), X=X, y=y, group=group)
    with open(os.path.join(run_dir, "fusion_train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    print(f"样本行 {X.shape}，正样本 {int(y.sum())}，负样本 {int(len(y) - y.sum())}，"
          f"正样本占比 {y.mean():.1%}，保存到 runs/fusion_train.npz", flush=True)


if __name__ == "__main__":
    main()
