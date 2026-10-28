"""诊断 3：手工验证模型概率与标签的关系，并重写 AUC 对照。"""

import os

import numpy as np
import torch

proj = r"D:\TFTMS-VLP"
data = np.load(os.path.join(proj, "runs", "fusion_train.npz"))
X, y = data["X"], data["y"].astype(bool)

ckpt = torch.load(os.path.join(proj, "runs", "fusion_head.pt"), map_location="cpu")
from scripts.train_fusion import FusionHead
model = FusionHead(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
with torch.inference_mode():
    prob = model(torch.from_numpy(X)).numpy().ravel()

# 1) 高分样本 vs 低分样本的正样本占比
order = np.argsort(-prob)
top = y[order[:2000]]
bot = y[order[-2000:]]
print(f"全局正样本占比 {y.mean():.1%}")
print(f"概率最高的前 2000 个样本中正样本占比 {top.mean():.1%}")
print(f"概率最低的后 2000 个样本中正样本占比 {bot.mean():.1%}")

# 2) 正/负样本平均概率与秩
ranks = np.empty_like(order, dtype=np.float64)
ranks[order] = np.arange(len(order), dtype=np.float64)
print(f"正样本平均概率 {prob[y].mean():.4f}，负样本平均概率 {prob[~y].mean():.4f}")
print(f"正样本平均秩(0=最高分) {ranks[y].mean():.1f}，负样本平均秩 {ranks[~y].mean():.1f}")

# 3) 重写 AUC（用配对比较，绝不写错）
n = len(y)
pairs_pos_neg = 0
cnt = 0
# 用排序统计：AUC = P(正样本分数 > 负样本分数)
sorted_idx = np.argsort(prob)  # 升序
n_pos, n_neg = int(y.sum()), int((~y).sum())
# 对每个负样本，统计有多少正样本分数严格更低
neg_mask = ~y
pos_mask = y
# 正样本中分数低于该负样本的比例之和 / n_pos
# 用升序秩：AUC = (sum_rank_pos_1based - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
r1 = np.empty(n, dtype=np.float64)
r1[sorted_idx] = np.arange(1, n + 1, dtype=np.float64)  # 1-based 升序秩
auc1 = (r1[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
print(f"\n1-based 升序秩 AUC（标准 Mann-Whitney）= {auc1:.4f}")

# 4) 逐样本抽查：打印 20 个高概率样本的标签
print("\n抽查（按概率降序前 20 个：标签 1/0 与概率）：")
for i in range(20):
    print(f"  idx={order[i]:5d}  y={int(y[order[i]])}  prob={prob[order[i]]:.3f}")
