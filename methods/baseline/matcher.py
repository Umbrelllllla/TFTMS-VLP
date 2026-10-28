"""规则 baseline：打分与预测。

对场景中的每个候选目标，依据《docs/dataset.md》确认的口径逐属性打分：
- place: 用归一化像面坐标 nx=x/z、ny=y/z（左/右/上/下/中心）
- facing: 用车头向量 heading=(cos ry,0,-sin ry) 与"指向相机方向"的夹角阈值
- distance: 欧氏 sqrt(x²+y²+z²) 落在 [d·(1-tol), d·(1+tol)]
- length/width/height: 落在 [lo, hi]（可加 pad）
- objectname/appearance/occlusion: 等值

得分 = 命中的属性权重之和。阈值统一放 config，只在开发集上调、调完冻结。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from code.datasets.rope3d import Rope3DObject
from methods.baseline.attributes import Constraints, _normalize_color_word


@dataclass
class MatchConfig:
    """打分/阈值配置（冻结用）。"""
    dist_mode: str = "le"            # 'le': d <= v*(1+dist_slack)（GT 实证语义）；'bracket': [v(1-tol), v(1+tol)]
    dist_slack: float = 0.1          # le 模式下上限松弛（GT 99% 分位 d/v=1.010）
    dist_tol: float = 0.5            # bracket 模式的相对容差（旧口径，保留）
    size_pad: float = 0.0            # 尺寸区间外扩容差（米；GT 实证 100% 严格区间内）
    facing_me_thr: float = 45.0      # 朝向我：车头夹角 <= 45°
    facing_away_thr: float = 135.0   # 背离我：车头夹角 >= 135°
    center_nx: float = 0.2           # 中心判定：|nx| < 0.2
    center_ny: float = 0.2
    place_cont: float = 0.4          # place 命中时的连续加分权重（GT 实证：命中位置的目标里，
                                     # GT 归一化坐标显著更靠描述方向（dev Δ=+0.062）；扫描 0.2~0.8 取 0.4
                                     # ——dev F1 96.65→96.80、FP -103、R 掉 0.02；0.6 起 R 崩。见笔记第 21 节）
    weights: dict = field(default_factory=lambda: {
        "objectname": 1.0, "appearance": 0.8, "place": 1.0,
        "size": 1.0, "distance": 0.6, "facing": 0.6, "occlusion": 0.8,
    })
    # 预测集口径（官方是"逐 目标×描述 二分类"）
    pred_mode: str = "rel"       # 'rel': >= max_score-rel_drop；'abs': >= abs_thr；'rel_abs': 两者都满足
    rel_drop: float = 0.2        # rel 模式下离最高分的容差（扫描 0.2/0.4/0.6/0.8 后取 0.2）
    abs_thr: float = 1.0         # abs 模式的绝对阈值
    facing_use_vertical: bool = True   # False 时朝向夹角用水平投影（忽略 y）


def euclid(o: Rope3DObject) -> float:
    return math.sqrt(o.x * o.x + o.y * o.y + o.z * o.z)


def _normalized_pos(o: Rope3DObject):
    """相机坐标下归一化像面位置 nx=x/z, ny=y/z（≈(u-cu)/fu, (v-cv)/fv）。"""
    if o.z == 0:
        return 0.0, 0.0
    return o.x / o.z, o.y / o.z


def _heading(o: Rope3DObject):
    """车头单位向量 (cos ry, 0, -sin ry)。"""
    return (math.cos(o.ry), 0.0, -math.sin(o.ry))


def _toward_camera(o: Rope3DObject):
    """指向相机的方向（由目标指向原点）。"""
    n = math.sqrt(o.x * o.x + o.y * o.y + o.z * o.z)
    if n == 0:
        return (0.0, 0.0, -1.0)
    return (-o.x / n, -o.y / n, -o.z / n)


def match_place(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> bool:
    if not c.place:
        return True
    nx, ny = _normalized_pos(o)
    p = c.place.strip().lower()
    if p == "left":
        return nx < 0
    if p == "right":
        return nx > 0
    if p == "center":
        return abs(nx) < cfg.center_nx and abs(ny) < cfg.center_ny
    if p == "top":
        return ny < 0
    if p == "bottom":
        return ny > 0
    if p == "top left":
        return nx < 0 and ny < 0
    if p == "top right":
        return nx > 0 and ny < 0
    if p == "bottom left":
        return nx < 0 and ny > 0
    if p == "bottom right":
        return nx > 0 and ny > 0
    return True  # 未识别的 place 不惩罚


def match_facing(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> bool:
    if not c.facing:
        return True
    if cfg.facing_use_vertical:
        h = _heading(o)
        t = _toward_camera(o)
        dot = h[0] * t[0] + h[1] * t[1] + h[2] * t[2]
    else:
        # 水平投影：只取 x/z 分量（目标在路面高度，垂直分量会污染"朝向"语义）
        h = (math.cos(o.ry), -math.sin(o.ry))
        n = math.sqrt(o.x * o.x + o.z * o.z)
        t = (-o.x / n, -o.z / n) if n else (0.0, -1.0)
        dot = h[0] * t[0] + h[1] * t[1]
    dot = max(-1.0, min(1.0, dot))
    theta = math.degrees(math.acos(dot))
    f = c.facing.strip().lower()
    if f.startswith("facing away"):
        return theta >= cfg.facing_away_thr
    if f.startswith("side facing"):
        return cfg.facing_me_thr < theta < cfg.facing_away_thr
    # facing me
    return theta <= cfg.facing_me_thr


def match_distance(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> bool:
    if c.distance is None:
        return True
    d = euclid(o)
    if cfg.dist_mode == "le":
        # GT 实证：97.2% 的 GT 目标满足 d<=v，99% 分位 d/v=1.010（"within ≈X"= 距离不超过 X）
        return d <= c.distance * (1 + cfg.dist_slack)
    lo = c.distance * (1 - cfg.dist_tol)
    hi = c.distance * (1 + cfg.dist_tol)
    return lo <= d <= hi


def _in_interval(v: float, r, pad: float) -> bool:
    lo, hi = r
    return (lo - pad) <= v <= (hi + pad)


def match_size(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> bool:
    ok = True
    if c.length and not _in_interval(o.l, c.length, cfg.size_pad):
        ok = False
    if c.width and not _in_interval(o.w, c.width, cfg.size_pad):
        ok = False
    if c.height and not _in_interval(o.h, c.height, cfg.size_pad):
        ok = False
    return ok


_OCC = {"no occluded": 0, "not occluded": 0, "partially occluded": 1, "occluded": 1}


def match_occlusion(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> bool:
    if not c.occlusion:
        return True
    target = _OCC.get(c.occlusion.strip().lower())
    if target is None:
        return True
    return int(o.occlusion) == target


_VEHICLE_CLASSES = {"car", "van", "bus", "truck", "big vehicle", "motorcycle"}


def _class_matches(constraint: str, obj_class: str) -> bool:
    """类别匹配：'vehicle' 表示任意车类；'a|b' 表示任一；否则等值。"""
    cc = constraint.strip().lower()
    oc = obj_class.lower()
    if cc == "vehicle":
        return oc in _VEHICLE_CLASSES
    if "|" in cc:
        return any(part.strip().lower() == oc for part in cc.split("|"))
    return oc == cc


def _color_matches(constraint: str, color: str) -> bool:
    """颜色匹配：约束端与目标端都规范化成颜色集合，两端交集非空即命中。

    规范化规则：把颜色串按 `-/| 空格` 拆分，逐词映射到规范化色系
    （silver-grey→{gray}、yellow-orange→{yellow,orange}、dark-brown→{black,brown}）。

    修复（2026-10-24）：上一版只把目标端规范化、约束端仍用原始串 `in` 规范化集合，
    导致『约束=复合色 ↔ 目标=复合色』判不中（silver-grey vs silver-grey → False），
    dev 有 309 条表达式因此 GT 匹配不上（S=0）。两端都规范化成集合后，
    约束单色↔目标复合色、约束复合色↔目标复合色、约束复合色↔目标单色 全部正确。"""
    def _norm(s: str) -> set:
        out = set()
        for part in re.split(r"[-/| ]+", s.strip().lower()):
            n = _normalize_color_word(part)
            if n:
                out.add(n)
        return out

    cc = _norm(constraint)
    col = _norm(color)
    if not cc or not col:
        # 任一端完全无法规范化（含非颜色词），退回严格等值
        return color.strip().lower() == constraint.strip().lower()
    return bool(cc & col)


def attribute_matches(c: Constraints, o: Rope3DObject, cfg: MatchConfig):
    """返回 {属性名: 是否命中}，仅包含描述里提到过的属性。"""
    out = {}
    if c.objectname:
        out["objectname"] = _class_matches(c.objectname, o.obj_class)
    if c.appearance:
        out["appearance"] = _color_matches(c.appearance, o.color)
    if c.place:
        out["place"] = match_place(c, o, cfg)
    if c.occlusion:
        out["occlusion"] = match_occlusion(c, o, cfg)
    if c.length or c.width or c.height:
        out["size"] = match_size(c, o, cfg)
    if c.distance is not None:
        out["distance"] = match_distance(c, o, cfg)
    if c.facing:
        out["facing"] = match_facing(c, o, cfg)
    return out


def _place_strength(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> float:
    """位置命中时目标相对描述方向的"强度"（0~1，越大越靠描述方向）。

    口径：与 match_place 同方向。left→-nx、right→nx、top→-ny、bottom→ny、
    center→-(|nx|+|ny|)（越靠中心越大）、角落→两方向坐标和。
    归一化：clamp 到 [0, 0.5] 再 /0.5（GT 实证 75% 分位 < 0.5，见笔记第 21 节）。
    """
    if not c.place:
        return 0.0
    nx, ny = _normalized_pos(o)
    p = c.place.strip().lower()
    if p == "left":
        s = -nx
    elif p == "right":
        s = nx
    elif p == "top":
        s = -ny
    elif p == "bottom":
        s = ny
    elif p == "center":
        s = -(abs(nx) + abs(ny))
    elif p in ("top left", "top right", "bottom left", "bottom right"):
        want_nx_neg = p.endswith("left")
        want_ny_neg = p.startswith("top")
        s = (-nx if want_nx_neg else nx) + (-ny if want_ny_neg else ny)
    else:
        return 0.0
    return min(max(s, 0.0), 0.5) / 0.5


def score_object(c: Constraints, o: Rope3DObject, cfg: MatchConfig) -> float:
    """加权命中分（place 命中时叠加连续加分，破离散同分）。"""
    w = cfg.weights
    total = 0.0
    for name, hit in attribute_matches(c, o, cfg).items():
        if not hit:
            continue
        total += w.get(name, 1.0)
        if name == "place" and cfg.place_cont > 0:
            total += cfg.place_cont * _place_strength(c, o, cfg)
    return total


def predict_scores(c: Constraints, candidates, cfg: MatchConfig):
    """为候选池打分，返回 [(obj, score)] 按分降序。"""
    scored = [(o, score_object(c, o, cfg)) for o in candidates]
    scored.sort(key=lambda x: (-x[1], x[0].to_tuple()))
    return scored


def top1(scored):
    return scored[0][0] if scored else None


def predict_set(scored, cfg: MatchConfig):
    """按配置选出"判定匹配"的目标集合（官方二分类用）。"""
    if not scored:
        return []
    top_score = scored[0][1]
    if cfg.pred_mode == "rel":
        return [o for o, s in scored if s >= top_score - cfg.rel_drop]
    if cfg.pred_mode == "rel_abs":
        return [o for o, s in scored if s >= top_score - cfg.rel_drop and s >= cfg.abs_thr]
    return [o for o, s in scored if s >= cfg.abs_thr]


def above_threshold(scored, thr: float):
    return [o for o, s in scored if s >= thr]
