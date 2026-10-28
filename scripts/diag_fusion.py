"""临时诊断：检查 fusion_train.npz 里各特征对正/负样本的区分度，定位 AUC<0.5 的原因。"""

import json
import os

import numpy as np

proj = r"D:\TFTMS-VLP"
data = np.load(os.path.join(proj, "runs", "fusion_train.npz"))
X, y = data["X"], data["y"].astype(bool)
names = ["clip_sim", "rule_score", "objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]

print(f"样本 {X.shape}，正样本 {int(y.sum())}（{y.mean():.1%}）\n")
for i, n in enumerate(names):
    pos = X[y, i]
    neg = X[~y, i]
    print(f"{n:10s} 正: mean {pos.mean():.4f}  neg: mean {neg.mean():.4f}  "
          f"正>负占比 {(pos > neg.mean()).mean():.2%}")

# rule_score 与 clip_sim 的正负直方分布
print("\n--- rule_score 分桶（正/负）---")
for thr in [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0]:
    mask = X[:, 1] >= thr
    if mask.sum() == 0:
        continue
    print(f"rule_score>={thr}: 样本 {mask.sum():5d}，其中正样本 {y[mask].mean():.1%}")

print("\n--- clip_sim 分桶（正/负）---")
for lo, hi in [(-1, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1)]:
    mask = (X[:, 0] >= lo) & (X[:, 0] < hi)
    if mask.sum() == 0:
        continue
    print(f"clip_sim[{lo},{hi}): 样本 {mask.sum():5d}，其中正样本 {y[mask].mean():.1%}")

# 属性标志分布
print("\n--- 属性命中标志：正样本命中率 vs 负样本命中率 ---")
for i, n in enumerate(names[2:], start=2):
    print(f"{n:10s} 正命中率 {X[y, i].mean():.2%}  负命中率 {X[~y, i].mean():.2%}")
