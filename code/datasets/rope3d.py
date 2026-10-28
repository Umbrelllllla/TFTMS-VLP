"""Rope3D 目标标注解析。

比赛数据继承自 Rope3D 的标注体系，一条目标包含 16 个字段，按空格分隔：
    class  trunc  occ  alpha  x1 y1 x2 y2  h w l  x y z  ry  color

两种来源格式：
1. 测试集 test_data：16 个字段用空格分隔的字符串。
2. 训练集 label_3    ：16 个字段的 Python repr 字符串，形如
   "['car', 2, 0, -1.46..., 671.3..., ..., 'white']"，需用 ast.literal_eval 解析。

注意：字段顺序/坐标系在正式建模前需用官方数据加载器交叉确认，勿想当然。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields


@dataclass
class Rope3DObject:
    """单个候选 3D 目标。

    字段顺序与 Rope3D 标注一致：
    class 类别 / trunc 截断 / occ 遮挡 / alpha 朝向角
    x1 y1 x2 y2 2D 框（像素）
    h w l 3D 尺寸（高 宽 长，米）
    x y z 3D 中心位置（米）
    ry 绕 Y 轴旋转角 / color 颜色
    """
    obj_class: str
    truncation: int
    occlusion: int
    alpha: float
    x1: float
    y1: float
    x2: float
    y2: float
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float
    color: str

    def to_tuple(self) -> tuple:
        """返回 16 字段的规范化元组，用于去重/比较。"""
        return (
            self.obj_class, self.truncation, self.occlusion, round(self.alpha, 4),
            round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2),
            round(self.h, 4), round(self.w, 4), round(self.l, 4),
            round(self.x, 4), round(self.y, 4), round(self.z, 4),
            round(self.ry, 4), self.color,
        )

    def __repr__(self) -> str:  # 可读打印
        return (f"<{self.obj_class} color={self.color} "
                f"bbox=({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}) "
                f"3D=({self.x:.1f},{self.y:.1f},{self.z:.1f}) hwl=({self.h:.2f},{self.w:.2f},{self.l:.2f}) "
                f"ry={self.ry:.2f}>")


_FIELD_NAMES = [f.name for f in fields(Rope3DObject)]


def parse_line(line: str) -> Rope3DObject:
    """解析测试集 test_data 的空格分隔行（16 个字段）。

    Args:
        line: 形如 "car 0 0 -1.5034 823.94 ... black" 的字符串。

    Returns:
        Rope3DObject；字段数不对时抛 ValueError。
    """
    parts = line.split()
    if len(parts) != 16:
        raise ValueError(
            f"期望 16 个字段，实际 {len(parts)} 个: {line[:120]!r}"
        )
    return _build(parts)


def parse_repr(s: str) -> Rope3DObject:
    """解析训练集 label_3 的 Python repr 字符串（16 个元素）。

    Args:
        s: 形如 "['car', 2, 0, -1.46, 671.3, ..., 'white']" 的字符串。

    Returns:
        Rope3DObject；解析失败抛异常。
    """
    lst = ast.literal_eval(s)
    if not isinstance(lst, (list, tuple)) or len(lst) != 16:
        raise ValueError(f"label_3 期望 16 个元素，实际 {len(lst) if hasattr(lst,'__len__') else 'N/A'}: {s[:120]!r}")
    return _build(lst)


def _build(raw) -> Rope3DObject:
    """把 16 个原始值按字段顺序构造成 Rope3DObject。"""
    obj_class = str(raw[0])
    truncation = int(raw[1])
    occlusion = int(raw[2])
    alpha = float(raw[3])
    x1, y1, x2, y2 = (float(v) for v in raw[4:8])
    h, w, l = (float(v) for v in raw[8:11])
    x, y, z = (float(v) for v in raw[11:14])
    ry = float(raw[14])
    color = str(raw[15]).strip()
    return Rope3DObject(
        obj_class=obj_class, truncation=truncation, occlusion=occlusion,
        alpha=alpha, x1=x1, y1=y1, x2=x2, y2=y2,
        h=h, w=w, l=l, x=x, y=y, z=z, ry=ry, color=color,
    )
