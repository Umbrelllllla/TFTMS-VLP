"""GT 标注者偏好分析：在"歧义案例"（多条目标同时满足描述约束）中，
比较 GT 选中的目标 vs 未选中的目标（歧义 FP）在距离/位置/遮挡/尺寸区间位置/朝向夹角上的分布差异。
发现系统性偏好 → 可设计"标注者偏好排序"收窄预测集（推理侧优化，不碰数据）。

只读分析，不修改任何数据/代码。
"""

from __future__ import annotations

import os
import statistics

from code.datasets.vlmod_dataset import iter_train
from methods.baseline.attributes import from_text
from methods.baseline.matcher import MatchConfig, attribute_matches

proj = r"D:\TFTMS-VLP"
with open(os.path.join(proj, "data", "splits", "train.txt"), encoding="utf-8") as fh:
    dev_stems = set(line.strip() for line in fh if line.strip())

cfg = MatchConfig()

# 统计字段
n_ambiguous_expr = 0       # 有歧义（S-G 非空）的表达式数
n_gt = 0                   # GT 目标总数（歧义案例内）
n_fp = 0                   # 歧义 FP 总数（S-G）
fields = {
    "dist": ([], []),       # 欧氏距离
    "nx": ([], []),         # 归一化像面横坐标
    "ny": ([], []),         # 归一化像面纵坐标
    "abs_nx": ([], []),     # |nx|
    "abs_ny": ([], []),     # |ny|
    "occ0": ([], []),       # 遮挡=未遮挡 比例
    "size_pos": ([], []),   # 尺寸区间内的相对位置（0=下界,1=上界；多尺寸取平均）
    "heading_deg": ([], []),# |车头夹角|（0=正对相机）
}
_occ_key = {"no occluded": 0, "partially occluded": 1, "occluded": 2}


def _size_relpos(c, o):
    """描述里提到的每个尺寸属性，目标值在区间内的相对位置 (v-lo)/(hi-lo)，多尺寸取平均。"""
    _obj_attr = {"length": "l", "width": "w", "height": "h"}
    pos = []
    for k in ("length", "width", "height"):
        r = getattr(c, k)
        if r is None:
            continue
        v = getattr(o, _obj_attr[k])
        lo, hi = r
        if hi > lo:
            pos.append((v - lo) / (hi - lo))
        else:
            pos.append(0.5)
    return sum(pos) / len(pos) if pos else None


def _heading_deg(o):
    import math
    # 车头向量 (cos ry, 0, -sin ry) 与指向相机向量 (0,0,1) 的夹角
    hx = math.cos(o.ry)
    hz = -math.sin(o.ry)
    dot = hz  # (hx,0,hz)·(0,0,1) = hz
    ang = math.acos(max(-1.0, min(1.0, dot))) * 180 / math.pi
    return ang


for sample in iter_train():
    stem = os.path.splitext(os.path.basename(sample.image_path))[0]
    if stem not in dev_stems:
        continue
    for e in sample.expressions:
        c = from_text(e.description)
        gt_set = {o.to_tuple() for o in e.matched_objects}
        # 约束满足集 S：所有描述属性都命中的候选
        S = []
        for o in sample.candidate_pool:
            attrs = attribute_matches(c, o, cfg)
            mentioned = set()
            if c.objectname:
                mentioned.add("objectname")
            for k in ("appearance", "place", "facing", "occlusion"):
                if getattr(c, k) is not None:
                    mentioned.add(k)
            if c.distance is not None:
                mentioned.add("distance")
            for k in ("length", "width", "height"):
                if getattr(c, k) is not None:
                    mentioned.add("size")
            hit = {k for k, v in attrs.items() if isinstance(v, (bool,)) and v}
            if hit >= mentioned:
                S.append(o)
        G = [o for o in S if o.to_tuple() in gt_set]
        F = [o for o in S if o.to_tuple() not in gt_set]
        if not F:
            continue
        n_ambiguous_expr += 1
        n_gt += len(G)
        n_fp += len(F)
        for o in G:
            d = (o.x**2 + o.y**2 + o.z**2) ** 0.5
            nx, ny = o.x / o.z, o.y / o.z
            fields["dist"][0].append(d); fields["nx"][0].append(nx); fields["ny"][0].append(ny)
            fields["abs_nx"][0].append(abs(nx)); fields["abs_ny"][0].append(abs(ny))
            fields["occ0"][0].append(_occ_key.get(o.occlusion, 1) == 0)
            sp = _size_relpos(c, o)
            if sp is not None:
                fields["size_pos"][0].append(sp)
            fields["heading_deg"][0].append(_heading_deg(o))
        for o in F:
            d = (o.x**2 + o.y**2 + o.z**2) ** 0.5
            nx, ny = o.x / o.z, o.y / o.z
            fields["dist"][1].append(d); fields["nx"][1].append(nx); fields["ny"][1].append(ny)
            fields["abs_nx"][1].append(abs(nx)); fields["abs_ny"][1].append(abs(ny))
            fields["occ0"][1].append(_occ_key.get(o.occlusion, 1) == 0)
            sp = _size_relpos(c, o)
            if sp is not None:
                fields["size_pos"][1].append(sp)
            fields["heading_deg"][1].append(_heading_deg(o))

print(f"歧义表达式数 = {n_ambiguous_expr}")
print(f"歧义案例中 GT 目标 = {n_gt}，歧义 FP = {n_fp}")
print(f"\n{'字段':<10} {'GT选中(均值)':<14} {'未选中(均值)':<14} {'差异'}")
for k, (a, b) in fields.items():
    if not a or not b:
        continue
    ma, mb = statistics.mean(a), statistics.mean(b)
    print(f"{k:<10} {ma:<14.4f} {mb:<14.4f} {ma - mb:+.4f}")
