"""规则 baseline 评测（官方口径，严谨版）。

官方打分（vlp.chd.edu.cn/tasks/2）：对每个测试样本，输出每个目标是否匹配 3 条描述
（每行 3 个 0/1），按"逐 目标×描述"二分类统计 TP/FP/FN，得 Precision/Recall/F1。

本脚本：
- 整体指标：F1 / Precision / Recall / TP / FP / FN / Acc@1
- 空 GT 表达式：单独计数；其预测一律算 FP（不再跳过）
- 预测数量分布：每条表达式预测了多少目标
- 逐属性 leave-one-out 消融：移除某一属性类型后整体 F1 变化（替代不严格的"属性维度统计"）
- 错误案例 dump：描述/约束/候选得分/各属性命中/GT/预测/错误类型 → runs/errors_*.jsonl

用法：
    uv run python -m scripts.eval_baseline --split dev
    uv run python -m scripts.eval_baseline --split val --text-only --errors 50 --ablate
"""

from __future__ import annotations

import argparse
import json
import os

from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import Constraints, extract, from_text
from methods.baseline.matcher import (MatchConfig, attribute_matches, predict_scores,
                                      predict_set, top1)

_ATTR_KEYS = ["objectname", "place", "facing", "distance", "length", "width",
              "height", "appearance", "occlusion"]
_ABLATE_GROUPS = {
    "objectname": ["objectname"],
    "place": ["place"],
    "facing": ["facing"],
    "distance": ["distance"],
    "size": ["length", "width", "height"],
    "appearance": ["appearance"],
    "occlusion": ["occlusion"],
}


def _drop_attrs(c: Constraints, keys):
    """返回移除了指定属性后的约束副本。"""
    d = {k: getattr(c, k) for k in _ATTR_KEYS}
    for k in keys:
        d[k] = None
    return Constraints(**d)


def evaluate(pairs, img_dir, cfg: MatchConfig, text_only: bool,
             error_limit: int = 0, error_path: str | None = None):
    """在 (stem, ann_path) 列表上跑评估，返回指标 dict。"""
    n_expr = 0
    empty_gt = 0
    acc1_num = 0
    acc1_den = 0
    TP = FP = FN = 0
    pred_sizes = {}
    errors = []

    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        for e in sample.expressions:
            gt = {o.to_tuple() for o in e.matched_objects}
            if not gt:
                empty_gt += 1
            c = from_text(e.description) if text_only else extract(e.public_properties, e.description)
            scored = predict_scores(c, sample.candidate_pool, cfg)
            if not scored:
                continue
            pred = predict_set(scored, cfg)
            n_expr += 1
            pred_tuples = {o.to_tuple() for o in pred}
            tp = len(pred_tuples & gt)
            fp = len(pred_tuples - gt)
            fn = len(gt - pred_tuples)
            TP += tp; FP += fp; FN += fn
            pred_sizes[len(pred_tuples)] = pred_sizes.get(len(pred_tuples), 0) + 1
            if gt:
                acc1_den += 1
                t1 = top1(scored)
                if t1 is not None and t1.to_tuple() in gt:
                    acc1_num += 1
            # 错误案例收集
            if (fp > 0 or fn > 0) and len(errors) < error_limit:
                err = {
                    "image": stem,
                    "ann_id": e.ann_id,
                    "description": e.description,
                    "properties": e.public_properties,
                    "constraints": {k: getattr(c, k) for k in _ATTR_KEYS},
                    "gt": [o.to_tuple() for o in e.matched_objects],
                    "pred": [o.to_tuple() for o in pred],
                    "fp": sorted(o.to_tuple() for o in pred if o.to_tuple() not in gt),
                    "fn": sorted(o.to_tuple() for o in e.matched_objects if o.to_tuple() not in pred_tuples),
                    "scored": [
                        {"obj": o.to_tuple(), "score": round(s, 3),
                         "attrs": {k: v for k, v in attribute_matches(c, o, cfg).items()}}
                        for o, s in scored[:8]
                    ],
                }
                errors.append(err)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f = prf(TP, FP, FN)
    result = {
        "n_expr": n_expr, "empty_gt": empty_gt,
        "TP": TP, "FP": FP, "FN": FN,
        "precision": p, "recall": r, "f1": f,
        "acc1": acc1_num / acc1_den if acc1_den else None,
        "pred_size_dist": dict(sorted(pred_sizes.items())),
    }
    if error_path and errors:
        os.makedirs(os.path.dirname(error_path) or ".", exist_ok=True)
        with open(error_path, "w", encoding="utf-8") as fh:
            for e in errors:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        result["errors_written"] = len(errors)
    return result


def print_result(tag: str, r: dict):
    acc = f"  Acc@1={r['acc1']*100:.1f}%" if r["acc1"] is not None else ""
    print(f"{tag}: F1={r['f1']*100:.2f}  P={r['precision']*100:.2f}  R={r['recall']*100:.2f}  "
          f"TP={r['TP']} FP={r['FP']} FN={r['FN']}  空GT={r['empty_gt']} 表达式={r['n_expr']}{acc}")


def load_split(split_txt: str, ds_root: str):
    with open(split_txt, encoding="utf-8") as fh:
        stems = [ln.strip() for ln in fh if ln.strip()]
    img_dir = os.path.join(ds_root, "img", "train")
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "train")
    anns = find_annotation_files(ann_dir)
    return [(s, anns[s]) for s in stems if s in anns], img_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "val"], default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--pred-mode", choices=["rel", "abs", "rel_abs"], default="rel")
    ap.add_argument("--rel-drop", type=float, default=None)
    ap.add_argument("--abs-thr", type=float, default=None)
    ap.add_argument("--dist-tol", type=float, default=None)
    ap.add_argument("--dist-slack", type=float, default=None)
    ap.add_argument("--dist-mode", choices=["le", "bracket"], default=None)
    ap.add_argument("--facing-horizontal", action="store_true",
                    help="朝向夹角用水平投影（忽略 y 垂直分量）")
    ap.add_argument("--errors", type=int, default=0, help="保存多少条错误案例")
    ap.add_argument("--ablate", action="store_true", help="逐属性 leave-one-out 消融")
    ap.add_argument("--place-cont", type=float, default=None,
                    help="place 命中时的连续加分权重（GT 典型性实证，笔记第 21 节；0=关闭）")
    args = ap.parse_args()

    ds_root = args.root or default_data_root()
    split_txt = os.path.join("data", "splits", "train.txt" if args.split == "dev" else "val.txt")
    pairs, img_dir = load_split(split_txt, ds_root)
    if args.max:
        pairs = pairs[: args.max]

    cfg = MatchConfig()
    if args.rel_drop is not None:
        cfg.rel_drop = args.rel_drop
    if args.abs_thr is not None:
        cfg.abs_thr = args.abs_thr
    if args.dist_tol is not None:
        cfg.dist_tol = args.dist_tol
    if args.dist_slack is not None:
        cfg.dist_slack = args.dist_slack
    if args.dist_mode is not None:
        cfg.dist_mode = args.dist_mode
    cfg.pred_mode = args.pred_mode
    cfg.facing_use_vertical = not args.facing_horizontal
    if args.place_cont is not None:
        cfg.place_cont = args.place_cont

    err_path = os.path.join("runs", f"errors_{args.split}.jsonl") if args.errors else None
    r = evaluate(pairs, img_dir, cfg, args.text_only, error_limit=args.errors, error_path=err_path)
    print_result(f"[{args.split}]", r)
    if "errors_written" in r:
        print(f"  错误案例已写入: {err_path} ({r['errors_written']} 条)")
    dist = r["pred_size_dist"]
    if dist:
        top = sorted(dist.items())[:6]
        print("  预测数量分布(前6): " + ", ".join(f"|pred|={k}:{v}" for k, v in top))

    if args.ablate:
        print("--- 逐属性 leave-one-out 消融（移除后整体F1） ---")
        base = r["f1"]
        print(f"  base        F1={base*100:.2f}")

        def run_ablated(keys):
            out = []
            for stem, ann_path in pairs:
                sample = load_train_sample(ann_path, img_dir)
                if sample is None:
                    continue
                for e in sample.expressions:
                    gt = {o.to_tuple() for o in e.matched_objects}
                    c = from_text(e.description) if args.text_only else extract(e.public_properties, e.description)
                    c = _drop_attrs(c, keys)
                    scored = predict_scores(c, sample.candidate_pool, cfg)
                    if not scored:
                        continue
                    pred = predict_set(scored, cfg)
                    pt = {o.to_tuple() for o in pred}
                    out.append((len(pt & gt), len(pt - gt), len(gt - pt)))
            tp = sum(x[0] for x in out); fp = sum(x[1] for x in out); fn = sum(x[2] for x in out)
            p = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            return 2 * p * r_ / (p + r_) if p + r_ else 0.0

        for name, keys in _ABLATE_GROUPS.items():
            f_abl = run_ablated(keys)
            print(f"  -{name:10s} F1={f_abl*100:.2f}  Δ={ (f_abl-base)*100:+.2f}")


if __name__ == "__main__":
    main()
