"""按官方格式为测试集生成提交文件（vlp.chd.edu.cn/tasks/2）。

对每个测试 JSON：
- 用自由文本解析 3 条 public_description → 约束
- 对 test_data 每个目标打分，判定它匹配哪些描述
- 输出同名 txt：每行 3 个 0/1（dA dB dC），行序 = test_data 顺序

输出到 result/ 目录，最后打包成 zip 提交。
用法：
    uv run python -m scripts.write_submission --rel-drop 0.4
"""

from __future__ import annotations

import argparse
import os
import zipfile

from code.datasets.vlmod_dataset import default_data_root, load_test_sample, find_annotation_files
from methods.baseline.attributes import from_text
from methods.baseline.matcher import MatchConfig, predict_scores, predict_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--out", type=str, default="result")
    ap.add_argument("--zip", type=str, default=None, help="生成的 zip 路径（默认 out + '.zip'）")
    ap.add_argument("--pred-mode", choices=["rel", "abs"], default="rel")
    ap.add_argument("--rel-drop", type=float, default=0.4)
    ap.add_argument("--abs-thr", type=float, default=1.0)
    args = ap.parse_args()

    ds_root = args.root or default_data_root()
    img_dir = os.path.join(ds_root, "img", "test")
    ann_dir = os.path.join(ds_root, "MonoMulti3D", "test")
    anns = find_annotation_files(ann_dir)

    cfg = MatchConfig(pred_mode=args.pred_mode, rel_drop=args.rel_drop, abs_thr=args.abs_thr)
    os.makedirs(args.out, exist_ok=True)

    n_written = 0
    for stem, ann_path in sorted(anns.items()):
        sample = load_test_sample(ann_path, img_dir)
        if sample is None:
            print(f"跳过（缺图）：{stem}")
            continue
        # 每条描述选出的匹配目标集合（按 to_tuple 判等）
        matched_sets = []
        for desc in sample.descriptions:
            c = from_text(desc)
            scored = predict_scores(c, sample.candidates, cfg)
            matched_sets.append({o.to_tuple() for o in predict_set(scored, cfg)})
        # 逐目标输出 3 列 0/1
        lines = []
        for o in sample.candidates:
            t = o.to_tuple()
            lines.append(" ".join("1" if t in ms else "0" for ms in matched_sets))
        out_path = os.path.join(args.out, stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        n_written += 1

    print(f"已生成 {n_written} 个提交文件到 {os.path.abspath(args.out)}")

    if args.zip is None:
        args.zip = args.out + ".zip"
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(args.out)):
            if name.endswith(".txt"):
                zf.write(os.path.join(args.out, name), arcname=name)
    print(f"已打包: {os.path.abspath(args.zip)}")


if __name__ == "__main__":
    main()
