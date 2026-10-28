"""决策级融合（ensemble）提交生成：规则门 + 融合头（E 策略，最终版）。

流程（与 diag_ensemble 的 E 策略严格一致，dev/val 上 F1 0.9712/0.9715）：
1. 每条测试描述 → 自由文本解析 → 规则打分 → 规则集合 A（rel_drop=0.2）
2. 全池目标算融合头概率（13 维特征：clip_sim, rule_score, 7 flags, 4 连续几何误差）
3. 只在 A 内按融合头概率 rel 窗口选：>= max(probs[A]) - 0.30

依赖：runs/clip_img_test.npy + clip_text_test.npy + 对应 _idx.json
      （scripts/clip_feature_extract --split test 产出）
输出：result_ensemble/*.txt → result_ensemble.zip

用法：
    uv run python -m scripts.write_submission_ensemble
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import torch

from code.datasets.vlmod_dataset import default_data_root, load_test_sample, find_annotation_files
from methods.baseline.attributes import from_text
from methods.baseline.geom_features import continuous_features
from methods.baseline.matcher import MatchConfig, attribute_matches, predict_scores, score_object
from scripts.train_fusion import FusionHead

proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_dir = os.path.join(proj, "runs")
_ATTR_ORDER = ["objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]

RULE_REL = 0.2      # 规则裁判窗口（冻结配置）
FUS_REL = 0.30      # 融合头在规则集合内窗口（dev/val 峰值；线上偏宽，见 --fus-rel）
OUT_DIR = "result_ensemble"


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fus-rel", type=float, default=FUS_REL,
                    help="融合头门内窗口（0.30=dev/val 最优；0.10/0.15=概率收紧版，压 FP）")
    ap.add_argument("--logits", action="store_true",
                    help="用 sigmoid 前 logits 做窗口（概率在 A 内饱和，logits 保留差异）")
    ap.add_argument("--lg-rel", type=float, default=0.20, help="logits 窗口值（配合 --logits）")
    ap.add_argument("--top-k", type=int, default=0,
                    help="每描述最多选 K 个目标（0=不限；8=只砍异常爆炸多选）")
    ap.add_argument("--out", type=str, default=None, help="输出目录")
    return ap.parse_args()


def load_model():
    ckpt = torch.load(os.path.join(run_dir, "fusion_head.pt"), map_location="cpu", weights_only=False)
    model = FusionHead(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    feat_mean = ckpt.get("feat_mean")
    feat_std = ckpt.get("feat_std")
    if feat_mean is None:
        feat_mean = np.zeros(ckpt["in_dim"], dtype=np.float32)
        feat_std = np.ones(ckpt["in_dim"], dtype=np.float32)
    return model, feat_mean, feat_std


def build_features(tvec, ivec, c, o, cfg):
    clip_sim = float(np.dot(tvec, ivec))
    attr = attribute_matches(c, o, cfg)
    rule_score = score_object(c, o, cfg)
    flag = [1.0 if attr.get(k, False) else 0.0 for k in _ATTR_ORDER]
    cont = continuous_features(c, o, cfg)
    return np.asarray([clip_sim, rule_score, *flag, *cont], dtype=np.float32)


def main() -> None:
    args = parse_args()
    fus_rel = args.fus_rel
    use_logits = args.logits
    lg_rel = args.lg_rel
    top_k = args.top_k
    tag = []
    if use_logits:
        tag.append(f"lg{lg_rel:.2f}")
    else:
        tag.append(f"p{fus_rel:.2f}")
    if top_k:
        tag.append(f"k{top_k}")
    out_dir = args.out or f"result_ensemble_{'_'.join(tag)}"
    model, feat_mean, feat_std = load_model()
    img = np.load(os.path.join(run_dir, "clip_img_test.npy"))
    txt = np.load(os.path.join(run_dir, "clip_text_test.npy"))
    with open(os.path.join(run_dir, "clip_img_test_idx.json"), encoding="utf-8") as fh:
        img_idx = json.load(fh)
    with open(os.path.join(run_dir, "clip_text_test_idx.json"), encoding="utf-8") as fh:
        txt_idx = json.load(fh)
    img_by_stem, txt_by_stem = {}, {}
    for i, m in enumerate(img_idx):
        img_by_stem.setdefault(m["stem"], []).append(i)
    for i, m in enumerate(txt_idx):
        txt_by_stem.setdefault(m["stem"], []).append(i)
    print(f"测试集特征缓存：图 {img.shape[0]} 目标 / {len(img_by_stem)} 图，文本 {txt.shape[0]} 条")

    ds_root = default_data_root()
    img_dir = os.path.join(ds_root, "img", "test")
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "test")
    anns = find_annotation_files(ann_dir)

    cfg = MatchConfig()
    os.makedirs(out_dir, exist_ok=True)

    n_written = n_missing = 0
    with torch.inference_mode():
        for stem, ann_path in sorted(anns.items()):
            sample = load_test_sample(ann_path, img_dir)
            if sample is None:
                print(f"跳过（缺图）：{stem}")
                continue
            if stem not in img_by_stem or stem not in txt_by_stem:
                print(f"跳过（缺 CLIP 特征）：{stem}")
                n_missing += 1
                continue
            pool = sample.candidates
            img_rows, txt_rows = img_by_stem[stem], txt_by_stem[stem]
            assert len(img_rows) == len(pool), f"{stem} 图行数 {len(img_rows)} != 候选数 {len(pool)}"
            assert len(txt_rows) == len(sample.descriptions), f"{stem} 文本行数不匹配"

            matched_sets = []
            for trow, desc in zip(txt_rows, sample.descriptions):
                tvec = txt[trow]
                c = from_text(desc)
                scored = predict_scores(c, pool, cfg)
                top_r = scored[0][1] if scored else 0.0
                rule_map = {o.to_tuple(): s for o, s in scored}
                rule_vec = np.array([rule_map.get(o.to_tuple(), 0.0) for o in pool], dtype=np.float32)
                a = {i for i, s in enumerate(rule_vec) if s >= top_r - RULE_REL}
                probs = np.zeros(len(pool), dtype=np.float32)
                for p, o in enumerate(pool):
                    f = build_features(tvec, img[img_rows[p]], c, o, cfg)
                    fx = (f - feat_mean) / feat_std
                    probs[p] = float(model(torch.from_numpy(fx[None, :]))[0, 0])
                if a:
                    if use_logits:
                        lg = -np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6)))
                        sub_top = float(lg[max(a)])
                        pred = {i for i in a if lg[i] >= sub_top - lg_rel}
                    else:
                        sub_top = float(probs[max(a)])
                        pred = {i for i in a if probs[i] >= sub_top - fus_rel}
                    if top_k and len(pred) > top_k:
                        ranked = sorted(pred, key=lambda i: probs[i], reverse=True)
                        pred = set(ranked[:top_k])
                else:
                    pred = set()
                matched_sets.append({pool[i].to_tuple() for i in pred})

            lines = []
            for o in pool:
                t = o.to_tuple()
                lines.append(" ".join("1" if t in ms else "0" for ms in matched_sets))
            out_path = os.path.join(out_dir, stem + ".txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            n_written += 1

    print(f"已生成 {n_written} 个提交文件到 {os.path.abspath(out_dir)}（跳过 {n_missing}）")

    zip_path = out_dir + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(out_dir)):
            if name.endswith(".txt"):
                zf.write(os.path.join(out_dir, name), arcname="result/" + name)
    print(f"已打包(result/结构): {os.path.abspath(zip_path)}")


if __name__ == "__main__":
    main()
