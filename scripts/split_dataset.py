"""图片级 80/20 数据划分脚本（防数据泄露）。

规则：
- 按图片（scene）级划分：同一张图的 3 条表达式必须同组，不跨开发/验证。
- seed 固定可复现；输出 data/splits/train.txt 与 val.txt（每行一个 json 基名）。
- 跳过缺标注/缺图的样本（与加载器一致）。

用法：
    uv run python -m scripts.split_dataset --ratio 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random

from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample


def main() -> None:
    ap = argparse.ArgumentParser(description="图片级 80/20 划分")
    ap.add_argument("--ratio", type=float, default=0.8, help="开发集比例，默认 0.8")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    ap.add_argument("--root", type=str, default=None, help="数据根目录，默认 VLMOD_DATA_ROOT 或 D:/VLMOD")
    args = ap.parse_args()

    ds_root = args.root or default_data_root()
    img_dir = os.path.join(ds_root, "img", "train")
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "train")
    anns = find_annotation_files(ann_dir)

    # 只保留可正常加载（有标注且有图）的样本
    valid = []
    for stem, ann_path in anns.items():
        sample = load_train_sample(ann_path, img_dir)
        if sample is not None:
            valid.append(stem)
    valid.sort()

    rng = random.Random(args.seed)
    rng.shuffle(valid)
    n_train = int(round(len(valid) * args.ratio))
    train_stems, val_stems = valid[:n_train], valid[n_train:]

    out_dir = os.path.join("data", "splits")
    os.makedirs(out_dir, exist_ok=True)
    train_txt = os.path.join(out_dir, "train.txt")
    val_txt = os.path.join(out_dir, "val.txt")
    with open(train_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(train_stems) + ("\n" if train_stems else ""))
    with open(val_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(val_stems) + ("\n" if val_stems else ""))

    print(f"数据根目录: {ds_root}")
    print(f"可加载样本总数: {len(valid)}")
    print(f"开发集(ratio={args.ratio:.2f}, seed={args.seed}): {len(train_stems)}")
    print(f"内部验证集: {len(val_stems)}")
    print(f"写入: {os.path.abspath(train_txt)}")
    print(f"写入: {os.path.abspath(val_txt)}")


if __name__ == "__main__":
    main()
