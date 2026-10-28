"""决定性核查：'GT 主观歧义' 到底是数据处理/规则没界定清楚，还是描述信息不足的真多解。

背景：此前把 dev 约 11% 的 FP 归因为'描述只给 2~3 个属性，多个目标同时满足，标注者挑了
其中一个'。用户质疑这可能是我们数据处理有问题、规则没界定清楚。

本脚本用官方结构化属性表 public_properties（from_properties 解析，不含自由文本解析缺陷）
作为每条描述的全部约束，对每张图的候选池（3 条表达式的 GT 去重）匹配，量化"满足集合大小 S"：

  S==0  : 连 GT 自身都不满足我们的规则 → 属性→字段映射 / 容差 / 解析没界定对（可修）
  S==1  : 规则界定清楚即唯一锁定 GT → 该样本不存在歧义
  S>=2  : 多个候选同时满足全部属性 → 潜在真多解；再做 leave-one-attribute-out，
          判断是"描述信息量确实不足（去掉任一属性都无法唯一）"还是"某属性规则可缩小"。

距离用两种口径对照（le slack=0.1 / bracket ±50%），因为 distance 是最大不确定性来源。

用法：uv run python -m scripts.diag_ambiguity [--split dev|val] [--max N]
"""

from __future__ import annotations

import argparse
import os

from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import from_properties
from methods.baseline.matcher import MatchConfig, attribute_matches

# 所有属性键（顺序用于 leave-one-out）
_ATTR_KEYS = ["objectname", "place", "facing", "distance",
              "length", "width", "height", "appearance", "occlusion"]
# 消融时把 size 视为一组
_SIZE_KEYS = ["length", "width", "height"]
_ABLATE_UNITS = {
    "objectname": ["objectname"], "place": ["place"], "facing": ["facing"],
    "distance": ["distance"], "size": ["length", "width", "height"],
    "appearance": ["appearance"], "occlusion": ["occlusion"],
}


def all_mentioned_match(c, o, cfg: MatchConfig) -> bool:
    """该目标是否满足描述 c 里'所有被提到的属性'（未提到的属性不惩罚）。"""
    am = attribute_matches(c, o, cfg)
    for k in _ATTR_KEYS:
        v = getattr(c, k)
        if v is None:
            continue
        if k in ("length", "width", "height"):
            if not am.get("size", False):
                return False
        else:
            if not am.get(k, False):
                return False
    return True


def mentioned_keys(c):
    keys = []
    for k in _ATTR_KEYS:
        if getattr(c, k) is not None:
            keys.append(k)
    return keys


def run(ds_root, pairs, img_dir, dist_mode: str):
    stats = {"expr": 0, "s0": 0, "s1": 0, "s2plus": 0}
    s0_gt_miss = {}       # S==0 时 GT 自身未命中属性 -> 计数
    s2_extra = {"extra_is_gt": 0, "extra_not_gt": 0}
    s2_no_single_attr_helps = 0   # 去掉任一单属性集合都不缩小
    s2_which_attr_helps = {}      # 去掉该属性后集合缩小 -> 计数
    examples = []

    for stem, ann_path in pairs:
        sample = load_train_sample(ann_path, img_dir)
        if sample is None:
            continue
        pool = sample.candidate_pool
        # 每个目标的"所属表达式 GT"映射（用于判断额外命中目标是否也是别的 GT）
        obj_owner = {}
        for i, e in enumerate(sample.expressions):
            for o in e.matched_objects:
                obj_owner.setdefault(o.to_tuple(), i)

        for e in sample.expressions:
            if not e.matched_objects:
                continue
            gt = e.matched_objects[0]
            c = from_properties(e.public_properties)
            if not mentioned_keys(c):
                continue
            cfg = MatchConfig()
            cfg.dist_mode = dist_mode

            hits = [o for o in pool if all_mentioned_match(c, o, cfg)]
            S = len(hits)
            stats["expr"] += 1
            gt_in = any(o.to_tuple() == gt.to_tuple() for o in hits)

            if S == 0:
                stats["s0"] += 1
                am = attribute_matches(c, gt, cfg)
                for k in mentioned_keys(c):
                    if k in ("length", "width", "height"):
                        key = "size"
                    else:
                        key = k
                    if not am.get(key, False):
                        s0_gt_miss[key] = s0_gt_miss.get(key, 0) + 1
            elif S == 1:
                stats["s1"] += 1
            else:
                stats["s2plus"] += 1
                extra = [o for o in hits if o.to_tuple() != gt.to_tuple()]
                extra_is_gt = all(obj_owner.get(o.to_tuple()) is not None for o in extra)
                if extra_is_gt:
                    s2_extra["extra_is_gt"] += 1
                else:
                    s2_extra["extra_not_gt"] += 1

                # leave-one-attribute-out：去掉某属性组后集合是否缩小
                base_tuples = {o.to_tuple() for o in hits}
                any_shrink = False
                for name, keys in _ABLATE_UNITS.items():
                    c2 = Constraints_remove(c, keys)
                    if not mentioned_keys(c2):
                        continue
                    hits2 = [o for o in pool if all_mentioned_match(c2, o, cfg)]
                    if len({o.to_tuple() for o in hits2}) < len(base_tuples):
                        any_shrink = True
                        s2_which_attr_helps[name] = s2_which_attr_helps.get(name, 0) + 1
                if not any_shrink:
                    s2_no_single_attr_helps += 1

                if len(examples) < 12:
                    examples.append({
                        "stem": stem, "ann_id": e.ann_id, "S": S,
                        "desc": e.description[:110], "props": e.public_properties,
                        "mentioned": mentioned_keys(c),
                        "gt_in": gt_in,
                        "gt": str(gt),
                        "extra": [str(o) for o in extra[:3]],
                    })

    print(f"\n===== 距离口径: {dist_mode} =====")
    ex = stats["expr"]
    print(f"表达式 {ex}   满足集合大小: S=0: {stats['s0']} ({stats['s0']/ex*100:.2f}%)   "
          f"S=1(唯一): {stats['s1']} ({stats['s1']/ex*100:.2f}%)   "
          f"S>=2(多解): {stats['s2plus']} ({stats['s2plus']/ex*100:.2f}%)")
    if stats["s0"]:
        miss = ", ".join(f"{k}:{v}" for k, v in sorted(s0_gt_miss.items(), key=lambda x: -x[1]))
        print(f"  [S=0 可修] GT 自身未命中属性分布: {miss}")
    print(f"  [S>=2] 额外命中目标中：属于其他GT: {s2_extra['extra_is_gt']}  非GT(候选池仅此3个目标,理论不应): {s2_extra['extra_not_gt']}")
    print(f"  [S>=2 信息量] 去掉任一单属性组都无法缩小的样本(真·信息不足): {s2_no_single_attr_helps}")
    if s2_which_attr_helps:
        w = ", ".join(f"{k}:{v}" for k, v in sorted(s2_which_attr_helps.items(), key=lambda x: -x[1]))
        print(f"  [S>=2 可缩] 去掉该属性组后满足集合缩小: {w}")

    if examples:
        print("\n--- 多解示例 ---")
        for x in examples:
            print(f"[{x['stem']}/{x['ann_id']}] S={x['S']} GT在集合={x['gt_in']} 提到={x['mentioned']}")
            print(f"   desc={x['desc']}")
            print(f"   gt ={x['gt']}")
            for e2 in x["extra"]:
                print(f"   +  ={e2}")


def Constraints_remove(c, keys):
    """移除指定属性后的约束副本（简单构造）。"""
    from methods.baseline.attributes import Constraints
    d = {k: getattr(c, k) for k in _ATTR_KEYS}
    for k in keys:
        d[k] = None
    return Constraints(**d)


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
    ap.add_argument("--dist", choices=["le", "bracket"], default="le")
    args = ap.parse_args()

    ds_root = args.root or default_data_root()
    split_txt = os.path.join("data", "splits", "train.txt" if args.split == "dev" else "val.txt")
    pairs, img_dir = load_split(split_txt, ds_root)
    if args.max:
        pairs = pairs[: args.max]
    run(ds_root, pairs, img_dir, args.dist)


if __name__ == "__main__":
    main()
