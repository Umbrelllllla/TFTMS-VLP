"""诊断 2：单特征 AUC + 模型概率 AUC，定位 AUC<0.5 是特征问题还是训练/对齐问题。"""

import os

import numpy as np
import torch

proj = r"D:\TFTMS-VLP"
data = np.load(os.path.join(proj, "runs", "fusion_train.npz"))
X, y = data["X"], data["y"].astype(bool)
names = ["clip_sim", "rule_score", "objectname", "appearance", "place", "size", "distance", "facing", "occlusion"]


def auc(score, yv):
    order = np.argsort(-score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    n_pos, n_neg = int(yv.sum()), int((~yv).sum())
    return (ranks[yv].mean() - (n_pos - 1) / 2) / n_neg


print("单特征 AUC（>0.5 说明该特征按分数排序有正向区分度）：")
for i, n in enumerate(names):
    print(f"  {n:10s} AUC = {auc(X[:, i], y):.4f}")

# 模型输出概率的 AUC
ckpt = torch.load(os.path.join(proj, "runs", "fusion_head.pt"), map_location="cpu")
from scripts.train_fusion import FusionHead
model = FusionHead(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
with torch.inference_mode():
    prob = model(torch.from_numpy(X)).numpy().ravel()
print(f"\n训练后模型概率 AUC = {auc(prob, y):.4f}")
print(f"正样本平均概率 {prob[y].mean():.4f}，负样本平均概率 {prob[~y].mean():.4f}")
