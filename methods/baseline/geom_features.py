"""连续几何误差特征 v1（恢复版，2026-10-24）：打破规则离散同分（GT 歧义 FP 的根源）。

规则分数 = 属性命中/未命中 × 权重之和，分辨率 ≈0.2。当描述只给 2~6 个属性、
多个目标全部命中时，它们离散分完全相同、并列最高，rel 窗口收窄也砍不掉
（实测 rel_drop 0→0.3 只差 5 个 FP，2002 个歧义 FP 全是同分并列目标）。

连续特征（4 维）让融合头能在"同分候选"之间继续区分：
    [0] dist_ratio : 目标欧氏距离 / 描述距离（描述未提及 → 1.0，配合 distance flag=0 让 MLP 学忽略）
    [1] size_pos   : 目标尺寸在描述区间内的相对位置均值（未提及 → 0.5）
    [2] facing_err : 车头夹角相对描述朝向的偏差（未提及 → 0.5）
    [3] place_err  : 目标归一化像面位置相对描述位置理想区的偏差（未提及 → 0.0）

【2026-10-24 实验结论（重要）】
- v2 精确偏差版（dist_err 米差/size_dev 比率/facing_delta 度数/place_match 原始坐标）已验证失败：
  混合负样本下纯融合头 B=0.9328/0.9347、决策级 E=0.9689/0.9674，均低于 v1 的 0.9479/0.9506
  与 0.9712/0.9715。主因：dist_err 米差跨描述尺度不一致（20m 描述差 1m vs 2m 描述差 0.1m，
  全局标准化后被稀释）；place_match 尺度与符号混乱。
- 恢复 v1 特征作为最优基线，后续改进走"训练目标"方向（rank loss）而非特征方向。
- 注意：2D 框特征（bbox_cx/bbox_cy/bbox_area）也已实验过（2026-10-24）——16 维版本训练 AUC
  涨到 0.9374 但 dev/val F1 反而降到 0.9389/0.9392（过拟合训练集、泛化差），是负结果，已回退。

build_fusion_dataset 与 eval_route_b 都从这里取特征，保证训练/推理严格对齐。
"""

from __future__ import annotations

import math

from methods.baseline.attributes import Constraints
from methods.baseline.matcher import MatchConfig


def _euclid(o) -> float:
    return math.sqrt(o.x * o.x + o.y * o.y + o.z * o.z)


def _normalized_pos(o):
    if o.z == 0:
        return 0.0, 0.0
    return o.x / o.z, o.y / o.z


def _heading_theta(o, cfg: MatchConfig) -> float:
    """车头方向与"目标→相机"方向的夹角（0~180°），与 matcher.match_facing 完全同公式。"""
    if cfg.facing_use_vertical:
        hx, hy, hz = math.cos(o.ry), 0.0, -math.sin(o.ry)
        n = _euclid(o)
        tx, ty, tz = (-o.x / n, -o.y / n, -o.z / n) if n else (0.0, 0.0, -1.0)
        dot = hx * tx + hy * ty + hz * tz
    else:
        hx, hz = math.cos(o.ry), -math.sin(o.ry)
        n = math.sqrt(o.x * o.x + o.z * o.z)
        tx, tz = (-o.x / n, -o.z / n) if n else (0.0, -1.0)
        dot = hx * tx + hz * tz
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def continuous_features(c: Constraints, o, cfg: MatchConfig) -> list[float]:
    """返回 4 维连续几何误差特征 v1（顺序固定）。"""
    # [0] 距离比率：目标欧氏距离 / 描述距离（尺度不变：描述 20m 差 1m 与 2m 差 0.1m 等比例）
    if c.distance is not None and c.distance > 0:
        dist_ratio = _euclid(o) / c.distance
    else:
        dist_ratio = 1.0

    # [1] 尺寸区间内相对位置（均值；多尺寸提及取平均）
    _obj_attr = {"length": "l", "width": "w", "height": "h"}
    pos = []
    for k in ("length", "width", "height"):
        r = getattr(c, k)
        if r is None:
            continue
        lo, hi = r
        if hi <= lo:
            pos.append(0.5)
            continue
        v = getattr(o, _obj_attr[k])
        pos.append(max(0.0, min(1.0, (v - lo) / (hi - lo))))
    size_pos = sum(pos) / len(pos) if pos else 0.5

    # [2] 朝向偏差
    theta = _heading_theta(o, cfg)
    f = c.facing.strip().lower() if c.facing else ""
    if f.startswith("facing away"):
        facing_err = abs(theta - 180.0) / 180.0
    elif f.startswith("side facing"):
        facing_err = abs(theta - 90.0) / 90.0
    elif f.startswith("facing me"):
        facing_err = theta / 180.0
    else:
        facing_err = 0.5  # 未提及

    # [3] 位置偏差：目标 (nx,ny) 相对描述位置理想区的偏移
    nx, ny = _normalized_pos(o)
    p = c.place.strip().lower() if c.place else ""
    if p == "left":
        place_err = max(nx, 0.0)
    elif p == "right":
        place_err = max(-nx, 0.0)
    elif p == "top":
        place_err = max(ny, 0.0)
    elif p == "bottom":
        place_err = max(-ny, 0.0)
    elif p == "center":
        dx = max(abs(nx) - cfg.center_nx, 0.0)
        dy = max(abs(ny) - cfg.center_ny, 0.0)
        place_err = math.sqrt(dx * dx + dy * dy)
    elif p in ("top left", "top right", "bottom left", "bottom right"):
        want_nx_neg = p.endswith("left")
        want_ny_neg = p.startswith("top")
        dx = max((nx - 0.0) if want_nx_neg else (-nx), 0.0)
        dy = max((ny - 0.0) if want_ny_neg else (-ny), 0.0)
        place_err = math.sqrt(dx * dx + dy * dy)
    else:
        place_err = 0.0  # 未提及/未识别

    return [dist_ratio, size_pos, facing_err, place_err]
