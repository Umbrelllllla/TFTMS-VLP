"""路线 B · 第 4 步：训练融合头（小 MLP）。

大白话：融合头就是一个"裁判"，输入两个评委的分（规则几何分 + CLIP 语义分）
和 7 个属性命中标志，输出 0~1 的"像不像描述目标"的置信度。它用训练样本
（GT=1，其他候选=0）学出"几何和语义怎么合最准"。CLIP 和规则本身都不动，
只训这一个小网络 —— 这就是路线 B 里"自己训练出来的模型"。

结构：Linear(9->32) + ReLU + Dropout(0.3) + Linear(32->1) + Sigmoid
损失：BCE；优化器：Adam(lr=1e-3)；epochs=50；batch=256（1.8 万行样本 CPU 秒级训完）

用法：
    uv run python -m scripts.train_fusion [--epochs 50]
产物：
    runs/fusion_head.pt    state_dict + 维度/配置（torch 保存）
    控制台：训练 loss、训练集 AUC
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, in_dim: int = 9, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def parse_args():
    ap = argparse.ArgumentParser(description="训练融合头 MLP")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-standardize", action="store_true",
                    help="不做特征标准化（默认标准化；实验对比用）")
    ap.add_argument("--drop-clip", action="store_true",
                    help="消融实验：去掉第 0 列 clip_sim，只用 rule_score+7 属性标志（8 维）")
    ap.add_argument("--loss", choices=["bce", "rank"], default="bce",
                    help="bce=交叉熵分类损失（v1 默认）；rank=margin ranking 排序损失（直接学 GT 排非 GT 前）")
    ap.add_argument("--rank-margin", type=float, default=0.2,
                    help="rank loss 的 margin（pos 分数须高出 neg 至少 margin）")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(proj, "runs")
    data = np.load(os.path.join(run_dir, "fusion_train.npz"))
    X = data["X"].astype(np.float32)
    if args.drop_clip:
        X = np.delete(X, 0, axis=1)  # 去掉 clip_sim 列（消融）
    y_np = data["y"].astype(np.float32)
    group_np = data["group"] if "group" in data else None
    y = torch.from_numpy(y_np).float().view(-1, 1)
    print(f"训练数据: {X.shape}，正样本 {int(y.sum())} / {len(y)}"
          f"（loss={args.loss}，{'已去掉 clip_sim 列，消融实验' if args.drop_clip else ''}）", flush=True)

    # 特征标准化：用训练集均值/方差，scaler 存入 ckpt（eval 时复用，防止泄露）
    if args.no_standardize:
        feat_mean = np.zeros(X.shape[1], dtype=np.float32)
        feat_std = np.ones(X.shape[1], dtype=np.float32)
        print("特征标准化: 关闭（原值输入）", flush=True)
    else:
        feat_mean = X.mean(axis=0)
        feat_std = X.std(axis=0) + 1e-8
        X = (X - feat_mean) / feat_std
        print(f"特征标准化: mean={np.round(feat_mean, 3)} std={np.round(X.std(axis=0), 3)}", flush=True)
    X = torch.from_numpy(X).float()

    model = FusionHead(in_dim=X.shape[1], hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(X)

    # rank loss 的 (pos, neg) 组内配对：同一描述下 GT 与非 GT 两两成对
    pos_pairs, neg_pairs = None, None
    if args.loss == "rank":
        assert group_np is not None, "rank loss 需要 build_fusion_dataset 输出 group 列"
        pairs_p, pairs_n = [], []
        for g in np.unique(group_np):
            idx = np.where(group_np == g)[0]
            pos_idx = idx[y_np[idx] == 1]
            neg_idx = idx[y_np[idx] == 0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            # 组内全配对（pos × neg）
            pp, qq = np.meshgrid(pos_idx, neg_idx)
            pairs_p.append(pp.ravel())
            pairs_n.append(qq.ravel())
        if pairs_p:
            pos_pairs = torch.from_numpy(np.concatenate(pairs_p).astype(np.int64))
            neg_pairs = torch.from_numpy(np.concatenate(pairs_n).astype(np.int64))
            print(f"rank loss 配对：{len(pos_pairs)} 对（组数 {len(np.unique(group_np))}）", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        if args.loss == "bce":
            loss_fn = nn.BCELoss()
            for i in range(0, n, args.batch):
                idx = perm[i:i + args.batch]
                xb, yb = X[idx], y[idx]
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
        else:
            # rank loss：随机抽 batch 对，pos 分数 - neg 分数 >= margin
            npairs = len(pos_pairs)
            margin = torch.tensor(args.rank_margin)
            for _ in range(0, npairs, args.batch):
                idx = torch.randint(0, npairs, (args.batch,))
                xp = X[pos_pairs[idx]]
                xn = X[neg_pairs[idx]]
                sp = model(xp).squeeze(-1)
                sn = model(xn).squeeze(-1)
                opt.zero_grad()
                loss = torch.clamp(margin - (sp - sn), min=0.0).mean()
                loss.backward()
                opt.step()
                total += loss.item() * args.batch
        if epoch in (1, 5, 10, 20, 30, 40, 50) or epoch == args.epochs:
            print(f"epoch {epoch:3d}  loss {total / n:.4f}", flush=True)

    # 训练集内 AUC（标准 Mann-Whitney，1-based 升序秩；仅作"模型有在学习"的粗查）
    model.eval()
    with torch.inference_mode():
        prob = model(X).numpy().ravel()
        yv = y.numpy().ravel()
        n_pos, n_neg = int(yv.sum()), len(yv) - int(yv.sum())
        order = np.argsort(prob)  # 升序
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
        auc = (ranks[yv == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    print(f"训练集内 AUC: {auc:.4f}", flush=True)

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X.shape[1],
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss": args.loss,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }, os.path.join(run_dir, "fusion_head.pt"))
    print(f"已保存 runs/fusion_head.pt（含特征标准化参数）", flush=True)


if __name__ == "__main__":
    main()
