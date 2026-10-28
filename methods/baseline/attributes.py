"""规则 baseline：属性约束抽取。

从描述中提取可用于打分的属性约束。
- 训练集：描述带结构化 `public_properties`（键值列表），直接解析，可靠。
- 测试集：只有自由文本 `public_description`，用关键词/正则解析（真实测试条件）。

自由文本解析覆盖（大小写不敏感）：
- 类别：car/sedan/person/truck/...，含泛化 "vehicle"（任意车类）
- 位置：left/right/top/bottom/center/组合
- 朝向：facing me / facing away from me / side facing me
- 颜色：常见颜色词、复合色（yellow-orange → "yellow|orange"）、同义词（silver→gray 等）
- 距离：about/approximately/roughly/nearly + X m(eters) [+ away] / within ...
- 尺寸：width/length/height 的区间（from X to Y / between X and Y）和单值（of/about X m）
- 遮挡：not occluded（否定）、partially occluded、heavily occluded

约束对象 `Constraints` 统一作为打分器的输入；objectname/appearance 用 "|" 表示"任一命中"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Constraints:
    """一条描述解析出的属性约束（None 表示该属性未提及）。"""
    objectname: str | None = None      # 类别；"vehicle"=任意车类；"a|b"=任一
    place: str | None = None           # top/left/right/... 
    facing: str | None = None          # 'facing me' | 'facing away from me' | 'side facing me'
    distance: float | None = None      # 目标距离值（米）
    length: tuple | None = None        # (lo, hi) 米
    width: tuple | None = None
    height: tuple | None = None
    appearance: str | None = None      # 颜色；"a|b"=任一
    occlusion: str | None = None       # 'no occluded' | 'partially occluded' | 'occluded'

    def non_null(self):
        return [(k, v) for k, v in self.__dict__.items() if v is not None]


_PROP_KEYS = {
    "objectname": "objectname",
    "place": "place",
    "facing_direction": "facing",
    "distance": "distance",
    "length": "length",
    "width": "width",
    "height": "height",
    "appearance": "appearance",
    "occlusion": "occlusion",
}


def _split_kv(s: str):
    """'length : 4.1 to 4.5 m' -> ('length', '4.1 to 4.5 m')"""
    if ":" in s:
        k, v = s.split(":", 1)
        return k.strip().lower(), v.strip()
    return s.strip().lower(), ""


def _parse_range(v: str):
    """'4.1 to 4.5 m' -> (4.1, 4.5)；'4.1 m' -> (4.1, 4.1)。失败返回 None。"""
    m = re.search(r"([\d.]+)\s*to\s*([\d.]+)", v)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"([\d.]+)", v)
    if m:
        x = float(m.group(1))
        return (x, x)
    return None


def _parse_scalar(v: str):
    m = re.search(r"([\d.]+)", v)
    return float(m.group(1)) if m else None


def from_properties(properties) -> Constraints:
    """从结构化 public_properties（键值字符串列表）解析。"""
    c = Constraints()
    for s in properties:
        key, val = _split_kv(s)
        if not key or not val:
            continue
        if key == "objectname":
            c.objectname = val
        elif key == "place":
            c.place = val
        elif key == "facing_direction":
            c.facing = val
        elif key == "distance":
            c.distance = _parse_scalar(val)
        elif key in ("length", "width", "height"):
            r = _parse_range(val)
            if r:
                setattr(c, key, r)
        elif key == "appearance":
            c.appearance = val
        elif key == "occlusion":
            c.occlusion = val
    return c


# ============ 自由文本解析（测试集用） ============

# --- 朝向（顺序：side 先于 me，避免被 substring 吞掉） ---
_FACING_PATTERNS = [
    (re.compile(r"side facing me|facing sideways|side view|facing to the side", re.I), "side facing me"),
    (re.compile(r"facing away from me|facing away from the camera|facing away", re.I), "facing away from me"),
    (re.compile(r"facing me|facing the camera|toward\s+me|facing toward", re.I), "facing me"),
]

# --- 位置（先组合后单值） ---
_PLACE_PATTERNS = [
    (re.compile(r"top[- ]left", re.I), "top left"),
    (re.compile(r"top[- ]right", re.I), "top right"),
    (re.compile(r"bottom[- ]left", re.I), "bottom left"),
    (re.compile(r"bottom[- ]right", re.I), "bottom right"),
    (re.compile(r"\btop\b", re.I), "top"),
    (re.compile(r"\bbottom\b", re.I), "bottom"),
    (re.compile(r"\bleft\b", re.I), "left"),
    (re.compile(r"\bright\b", re.I), "right"),
    (re.compile(r"in the (center|centre)|middle", re.I), "center"),
]

# --- 颜色（含同义词） ---
_COLOR_SYNONYMS = {
    "white": ["white", "ivory", "cream", "pearl", "milky"],
    "black": ["black", "charcoal", "graphite", "dark", "onyx"],
    "gray": ["gray", "grey", "silver", "ash", "slate"],
    "red": ["red", "maroon", "crimson", "scarlet"],
    "blue": ["blue", "navy", "azure", "sky blue"],
    "green": ["green", "olive", "emerald"],
    "yellow": ["yellow", "gold", "amber"],
    "orange": ["orange", "copper", "ginger"],
    "brown": ["brown", "tan", "beige", "bronze"],
    "purple": ["purple", "violet", "magenta"],
    "pink": ["pink", "rose", "salmon"],
}
_ALL_COLOR_WORDS = sorted({w for ws in _COLOR_SYNONYMS.values() for w in ws}, key=len, reverse=True)


def _normalize_color_word(w: str) -> str | None:
    for canon, syns in _COLOR_SYNONYMS.items():
        if w.lower() in syns or w.lower() == canon:
            return canon
    return None


def _parse_color(text: str) -> str | None:
    # 复合色优先："yellow-orange" → "yellow|orange"（避免 \borange\b 先命中）
    m = re.search(r"([a-z]+)\s*-\s*([a-z]+)", text, re.I)
    if m:
        a = _normalize_color_word(m.group(1))
        b = _normalize_color_word(m.group(2))
        if a and b and a != b:
            return f"{a}|{b}"
    for raw in _ALL_COLOR_WORDS:
        if re.search(rf"\b{re.escape(raw)}\b", text, re.I):
            return next(c for c, ws in _COLOR_SYNONYMS.items() if raw in ws)
    return None


# --- 类别（含泛化与同义词，含复数） ---
_OBJ_SYNONYMS = {
    "car": ["car", "cars", "sedan", "automobile", "auto", "saloon", "hatchback", "suv"],
    "truck": ["truck", "trucks", "lorry", "lorries", "pickup"],
    "bus": ["bus", "buses", "coach", "coaches"],
    "van": ["van", "vans"],
    "pedestrian": ["pedestrian", "pedestrians", "person", "people", "walker", "human"],
    "cyclist": ["cyclist", "cyclists", "bicycle", "bicycles", "bike rider", "rider"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "motor bike"],
}
_VEHICLE_WORDS = {"vehicle", "vehicles", "motor vehicle", "motorized vehicle"}
_ALL_OBJ_WORDS = sorted({w for ws in _OBJ_SYNONYMS.values() for w in ws}, key=len, reverse=True)


def _parse_objectname(text: str) -> str | None:
    for raw in _ALL_OBJ_WORDS:
        if re.search(rf"\b{re.escape(raw)}\b", text, re.I):
            return next(c for c, ws in _OBJ_SYNONYMS.items() if raw in ws)
    for raw in sorted(_VEHICLE_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(raw)}\b", text, re.I):
            return "vehicle"
    return None


# --- 遮挡（含否定） ---
def _parse_occlusion(text: str) -> str | None:
    t = text.lower()
    if re.search(r"not\s+(?:occluded|covered|blocked)|without\s+(?:any\s+)?occlusion|no\s+occlusion|"
                 r"without\s+being\s+occluded|isn'?t\s+occluded|unoccluded|fully\s+visible|"
                 r"completely\s+visible|not\s+covered|not\s+partially\s+occluded", t):
        return "no occluded"
    if re.search(r"partially?\s+occluded|partly\s+occluded|half\s+occluded|some\s+occlusion|"
                 r"partially?\s+covered", t):
        return "partially occluded"
    if re.search(r"heavily\s+occluded|seriously\s+occluded|severely\s+occluded|mostly\s+occluded|occluded", t):
        return "occluded"
    return None


# --- 距离（近似表达） ---
_DIST_PATTERNS = [
    re.compile(r"(?:within\s+)?(?:approximately|about|around|roughly|nearly|approx\.?)?\s*([\d.]+)\s*m(?:eters)?\s+away", re.I),
    re.compile(r"within\s+(?:approximately|about|around|roughly)?\s*([\d.]+)\s*m(?:eters)?", re.I),
    re.compile(r"(?:approximately|about|around|roughly|nearly)\s+([\d.]+)\s*m(?:eters)?", re.I),
]


def _parse_distance(text: str) -> float | None:
    for pat in _DIST_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


# --- 尺寸（区间与单值） ---
_APPROX = r"(?:approximately|about|around|roughly|nearly|approx\.?)?\s*"


def _parse_size(text: str, word: str):
    """解析 'width ranging from approximately 1.2 to 1.6 meters' /
    'a height of approximately 0.9 to 1.3 meters' / 'approximately 1.5 meters wide' 等。
    取 word 前后窗口找数字对；找不到对就取单个数字。
    注意：word 支持复数（lengths/widths/heights），用 {word}s? 匹配。"""
    # 先找 word 前/后的数字对（更稳，容忍 'approximately' 夹在中间）
    # 注意：rf 字符串里大括号量词必须写成 {{0,30}}，否则被 f-string 插值成元组
    pats = [
        re.compile(rf"\b{word}s?\b.{{0,30}}?{_APPROX}([\d.]+)\s*(?:to|-|and)\s*{_APPROX}([\d.]+)\s*m(?:eters)?", re.I),
        re.compile(rf"{_APPROX}([\d.]+)\s*(?:to|-|and)\s*{_APPROX}([\d.]+)\s*m(?:eters)?.{{0,20}}?\b{word}s?\b", re.I),
        re.compile(rf"\b{word}s?\b.{{0,30}}?{_APPROX}([\d.]+)\s*m(?:eters)?", re.I),
        re.compile(rf"{_APPROX}([\d.]+)\s*m(?:eters)?.{{0,20}}?\b{word}s?\b", re.I),
    ]
    for pat in pats:
        m = pat.search(text)
        if m:
            g = m.groups()
            try:
                if len(g) >= 2 and g[1] is not None:
                    return (float(g[0]), float(g[1]))
                return (float(g[0]), float(g[0]))
            except (TypeError, ValueError):
                continue
    return None


def from_text(text: str) -> Constraints:
    """从自由文本 public_description 做轻量关键词解析（测试集用）。"""
    c = Constraints()
    for pat, val in _FACING_PATTERNS:
        if pat.search(text):
            c.facing = val
            break
    for pat, val in _PLACE_PATTERNS:
        if pat.search(text):
            c.place = val
            break
    c.appearance = _parse_color(text)
    c.objectname = _parse_objectname(text)
    c.occlusion = _parse_occlusion(text)
    c.distance = _parse_distance(text)
    for word in ("length", "width", "height"):
        r = _parse_size(text, word)
        if r:
            setattr(c, word, r)
    return c


def extract(properties, description: str | None = None) -> Constraints:
    """优先用结构化属性；缺省时回退到自由文本解析。"""
    c = from_properties(properties)
    if description:
        ft = from_text(description)
        # 结构化缺失的字段用文本补
        for k in ("objectname", "place", "facing", "distance", "length", "width",
                  "height", "appearance", "occlusion"):
            if getattr(c, k) is None:
                setattr(c, k, getattr(ft, k))
    return c
