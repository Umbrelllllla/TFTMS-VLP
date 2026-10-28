"""重扫 rel_drop 阈值（修复复数+颜色后，错误结构已变：97% FP 为 GT 歧义）。

数据只加载一次，循环不同 rel_drop 计算官方口径 P/R/F1/Acc@1。
"""

from __future__ import annotations

import os

from code.datasets.vlmod_dataset import iter_train
from methods.baseline.attributes import from_text
from methods.baseline.matcher import MatchConfig, predict_scores, predict_set

proj = r"D:\TFTMS-VLP"


def run_split(split_txt: str, rel_drops, label: str):
    with open(os.path.join(proj, "data", "splits", split_txt), encoding="utf-8") as fh:
        stems = set(l.strip() for l in fh if l.strip())
    # 收集表达式（一次加载）
    exprs = []
    for sample in iter_train():
        stem = os.path.splitext(os.path.basename(sample.image_path))[0]
        if stem not in stems:
            continue
        for e in sample.expressions:
            c = from_text(e.description)
            scored = predict_scores(c, sample.candidate_pool, MatchConfig())
            gt = {o.to_tuple() for o in e.matched_objects}
            exprs.append((scored, gt))
    print(f"[{label}] 表达式数={len(exprs)}")
    print(f"{'rel_drop':<10}{'P':<8}{'R':<8}{'F1':<8}{'FP':<7}{'FN':<6}{'Acc@1':<8}")
    for rd in rel_drops:
        cfg = MatchConfig()
        cfg.rel_drop = rd
        TP = FP = FN = 0
        acc_num = acc_den = 0
        for scored, gt in exprs:
            if not scored:
                continue
            pred = predict_set(scored, cfg)
            pt = {o.to_tuple() for o in pred}
            tp = len(pt & gt); fp = len(pt - gt); fn = len(gt - pt)
            TP += tp; FP += fp; FN += fn
            if gt:
                acc_den += 1
                t1 = scored[0][0].to_tuple()
                if t1 in gt:
                    acc_num += 1
        p = TP / (TP + FP) if TP + FP else 0
        r = TP / (TP + FN) if TP + FN else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        acc = acc_num / acc_den if acc_den else 0
        print(f"{rd:<10.3f}{p:<8.4f}{r:<8.4f}{f1:<8.4f}{FP:<7d}{FN:<6d}{acc:<8.4f}")


run_split("train.txt", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "dev")
run_split("val.txt", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "val")
