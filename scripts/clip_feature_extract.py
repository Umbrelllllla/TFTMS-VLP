"""路线 B · 第 2 步：CLIP（中文）特征提取与缓存。

任务：对每个候选目标的 2D 框裁剪图做 CLIP 图像编码；对每条描述文本做 CLIP 文本编码。
产物全部缓存到 runs/，之后训练/评测直接读缓存，不再重算（本机无 GPU，全 CPU 推理，特征只算一次）。

模型：Chinese-CLIP ViT-B/16（OFA-Sys 官方，中文图文匹配，~2 亿中文图文对预训练）。
      cn_clip 库加载；use_modelscope=True 时自动从魔搭(ModelScope)下载权重到 download_root。
      （HF 直连与 OSS 直链在本机均超时/403，魔搭是可用通道——2026-10-22 实测）

运行示例：
    uv run python -m scripts.clip_feature_extract --split train --limit 20   # 冒烟测试
    uv run python -m scripts.clip_feature_extract --split train              # 全量训练集
    uv run python -m scripts.clip_feature_extract --split test               # 全量测试集

产物：
    runs/clip_img_<split>.npy         图像特征矩阵 (N_objs, 512)，行序与索引 json 对应，L2 归一化
    runs/clip_img_<split>_idx.json    [{stem, obj_i, box:[x1,y1,x2,y2], cls}]
    runs/clip_text_<split>.npy        文本特征矩阵 (N_expr, 512)，行序与索引 json 对应，L2 归一化
    runs/clip_text_<split>_idx.json   [{stem, expr_i, desc, gt_pool_idx:[...]}]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

MODEL_NAME = "ViT-B-16"
FEAT_DIM = 512
BATCH = 32  # 图像批大小（CPU 上平衡吞吐与内存）


def parse_args():
    ap = argparse.ArgumentParser(description="Chinese-CLIP 特征提取与缓存")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本（冒烟测试用），0=全量")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略已有缓存从头跑（默认：已有缓存则自动续跑）")
    return ap.parse_args()


def save_checkpoint(img_feats, text_feats, img_idx, text_idx, out_dir, split, done_stems, t0):
    """把累积特征写入正式缓存文件（checkpoint，可被 resume 复用）。"""
    img_arr = np.concatenate(img_feats, axis=0) if img_feats else np.zeros((0, FEAT_DIM), dtype=np.float32)
    txt_arr = np.concatenate(text_feats, axis=0) if text_feats else np.zeros((0, FEAT_DIM), dtype=np.float32)
    np.save(os.path.join(out_dir, f"clip_img_{split}.npy"), img_arr)
    with open(os.path.join(out_dir, f"clip_img_{split}_idx.json"), "w", encoding="utf-8") as fh:
        json.dump(img_idx, fh, ensure_ascii=False, indent=1)
    np.save(os.path.join(out_dir, f"clip_text_{split}.npy"), txt_arr)
    with open(os.path.join(out_dir, f"clip_text_{split}_idx.json"), "w", encoding="utf-8") as fh:
        json.dump(text_idx, fh, ensure_ascii=False, indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] checkpoint：图 {len(done_stems)}，"
          f"目标 {img_arr.shape[0]}，表达式 {txt_arr.shape[0]}，累计 {time.time()-t0:.0f}s", flush=True)


def load_done(out_dir, split):
    """已有缓存则返回已完成的 stem 集合，否则空集。"""
    p = os.path.join(out_dir, f"clip_img_{split}_idx.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return {m["stem"] for m in json.load(fh)}
    return set()


def crop_box(img: Image.Image, obj) -> Image.Image:
    """按 2D 框裁剪，越界 clamp；空框退化为整图。"""
    w, h = img.size
    x1 = max(0, int(round(obj.x1)))
    y1 = max(0, int(round(obj.y1)))
    x2 = min(w, int(round(obj.x2)))
    y2 = min(h, int(round(obj.y2)))
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img.crop((x1, y1, x2, y2))


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main() -> None:
    args = parse_args()
    import cn_clip.clip as clip  # 延迟导入，便于 --help 快速返回
    from cn_clip.clip import load_from_name, tokenize

    weight_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "cn_clip")
    os.makedirs(weight_dir, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] 加载中文 CLIP {MODEL_NAME}（首次自动从魔搭下载权重到 {weight_dir}）...", flush=True)
    t0 = time.time()
    model, preprocess = load_from_name(
        MODEL_NAME,
        device="cpu",
        download_root=weight_dir,
        use_modelscope=True,
    )
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] 模型加载完成，耗时 {time.time()-t0:.0f}s（CPU 推理，无 GPU）", flush=True)

    if args.split == "train":
        from code.datasets.vlmod_dataset import iter_train
        samples = list(iter_train())
    else:
        from code.datasets.vlmod_dataset import iter_test
        samples = list(iter_test())
    if args.limit:
        samples = samples[:args.limit]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(out_dir, exist_ok=True)
    done = set() if args.no_resume else load_done(out_dir, args.split)
    if done:
        print(f"[{time.strftime('%H:%M:%S')}] 续跑模式：已有缓存 {len(done)} 张图，跳过它们", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] 样本数（图）: {len(samples)}（待处理 {len(samples)-len(done)}）", flush=True)

    img_feats, img_idx = [], []
    text_feats, text_idx = [], []
    if done:
        # 续跑：把已缓存特征作为累积起点，最终保存=旧+新，避免丢数据
        img_feats = [np.load(os.path.join(out_dir, f"clip_img_{args.split}.npy"))]
        text_feats = [np.load(os.path.join(out_dir, f"clip_text_{args.split}.npy"))]
        with open(os.path.join(out_dir, f"clip_img_{args.split}_idx.json"), encoding="utf-8") as fh:
            img_idx = json.load(fh)
        with open(os.path.join(out_dir, f"clip_text_{args.split}_idx.json"), encoding="utf-8") as fh:
            text_idx = json.load(fh)
    new_count = 0  # 本次会话新处理的图数

    with torch.inference_mode():
        for n, sample in enumerate(samples, 1):
            stem = os.path.splitext(os.path.basename(sample.image_path))[0]
            if stem in done:
                continue
            # ---------- 图像支路：候选池每个目标裁剪 → 图像特征 ----------
            pool = sample.candidate_pool if args.split == "train" else sample.candidates
            img = Image.open(sample.image_path).convert("RGB")
            crops = [crop_box(img, o) for o in pool]
            for batch in batched(crops, BATCH):
                imgs = torch.stack([preprocess(c) for c in batch])
                feats = model.encode_image(imgs)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                img_feats.append(feats.cpu().numpy())
            for i, o in enumerate(pool):
                img_idx.append({
                    "stem": stem,
                    "obj_i": i,
                    "box": [round(o.x1, 1), round(o.y1, 1), round(o.x2, 1), round(o.y2, 1)],
                    "cls": o.obj_class,
                })

            # ---------- 文本支路：描述 → 文本特征 ----------
            if args.split == "train":
                descs = [e.description for e in sample.expressions]
                # GT 目标在候选池中的位置（label_3 -> candidate_pool 匹配）
                pool_pos = {obj.to_tuple(): i for i, obj in enumerate(pool)}
                gt_idx = [
                    [pool_pos[o.to_tuple()] for o in e.matched_objects if o.to_tuple() in pool_pos]
                    for e in sample.expressions
                ]
                text_meta = [{"stem": stem, "expr_i": i, "desc": d, "gt_pool_idx": g}
                             for i, (d, g) in enumerate(zip(descs, gt_idx))]
            else:
                descs = sample.descriptions
                text_meta = [{"stem": stem, "expr_i": i, "desc": d, "gt_pool_idx": []}
                             for i, d in enumerate(descs)]

            tok = tokenize(descs)
            tfeats = model.encode_text(tok)
            tfeats = torch.nn.functional.normalize(tfeats, p=2, dim=-1)
            text_feats.append(tfeats.cpu().numpy())
            text_idx.extend(text_meta)

            new_count += 1
            if new_count % 200 == 0:
                save_checkpoint(img_feats, text_feats, img_idx, text_idx,
                                out_dir, args.split, done | {os.path.splitext(os.path.basename(s.image_path))[0]
                                                              for s in samples[:n]}, t0)
            if n % 100 == 0 or n == len(samples):
                print(f"[{time.strftime('%H:%M:%S')}] 进度 {n}/{len(samples)}（本次新处理 {new_count}），"
                      f"累计目标 {sum(len(x) for x in img_feats)}，表达式 {sum(len(x) for x in text_feats)}",
                      flush=True)

    # ---------- 最终保存（checkpoint 已每 200 图落盘一次，这里收尾） ----------
    save_checkpoint(img_feats, text_feats, img_idx, text_idx,
                    out_dir, args.split,
                    done | {os.path.splitext(os.path.basename(s.image_path))[0] for s in samples}, t0)
    print(f"[{time.strftime('%H:%M:%S')}] 完成：图像特征与文本特征已写入 runs/，"
          f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
