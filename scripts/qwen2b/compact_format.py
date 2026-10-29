"""Compact, deterministic condition format for faster autoregressive decoding."""
from decimal import Decimal, InvalidOperation
import hashlib
import re

import run_parse_once as base


SYSTEM_PROMPT = """Extract only conditions explicitly stated in the description.
Output one compact line, no explanation. Use fields in this exact order when present:
C=car|K=white|P=tl|F=me|O=none|D=30|L=3.2:3.8|W=1.5:1.9|H=1.1:1.5
C category: car,van,truck,bus. Generic vehicle gives no C.
K color. P position: tl,t,tr,l,c,r,bl,b,br.
F facing: me,away,side. O occlusion: none,part.
D distance upper bound. L/W/H are length/width/height min:max.
Omit unstated fields. Copy numbers exactly.
Description: White cars at the top right are not occluded and are 3.2 to 3.8 meters long.
Output: C=car|K=white|P=tr|O=none|L=3.2:3.8"""

CODE_ORDER = ("C", "K", "P", "F", "O", "D", "L", "W", "H")
KEY_TO_CODE = {
    "category": "C", "color": "K", "position": "P", "facing": "F", "occlusion": "O",
    "distance_max_m": "D", "length": "L", "width": "W", "height": "H",
}
POSITION_ENCODE = {
    "top left": "tl", "top": "t", "top right": "tr", "left": "l", "center": "c",
    "right": "r", "bottom left": "bl", "bottom": "b", "bottom right": "br",
}
POSITION_DECODE = {value: key for key, value in POSITION_ENCODE.items()}
FACING_ENCODE = {"facing me": "me", "facing away from me": "away", "side facing me": "side"}
FACING_DECODE = {value: key for key, value in FACING_ENCODE.items()}
OCCLUSION_ENCODE = {"no occluded": "none", "partially occluded": "part"}
OCCLUSION_DECODE = {value: key for key, value in OCCLUSION_ENCODE.items()}


def number(value):
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError(f"invalid number: {value}")
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def encode(conditions):
    unknown = set(conditions) - base.ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown keys: {sorted(unknown)}")
    values = {}
    if "category" in conditions: values["C"] = conditions["category"]
    if "color" in conditions: values["K"] = conditions["color"]
    if "position" in conditions: values["P"] = POSITION_ENCODE[conditions["position"]]
    if "facing" in conditions: values["F"] = FACING_ENCODE[conditions["facing"]]
    if "occlusion" in conditions: values["O"] = OCCLUSION_ENCODE[conditions["occlusion"]]
    if "distance_max_m" in conditions: values["D"] = number(conditions["distance_max_m"])
    for dimension, code in (("length", "L"), ("width", "W"), ("height", "H")):
        lo, hi = f"{dimension}_min_m", f"{dimension}_max_m"
        if (lo in conditions) != (hi in conditions):
            raise ValueError(f"{dimension} needs min and max")
        if lo in conditions:
            values[code] = f"{number(conditions[lo])}:{number(conditions[hi])}"
    if not values:
        raise ValueError("empty conditions")
    return "|".join(f"{code}={values[code]}" for code in CODE_ORDER if code in values)


def parse(raw):
    try:
        if not raw or raw.strip() != raw or "\n" in raw:
            raise ValueError("output must be one non-empty stripped line")
        pairs = raw.split("|")
        values = {}
        observed = []
        for pair in pairs:
            if pair.count("=") != 1:
                raise ValueError(f"invalid segment: {pair}")
            code, value = pair.split("=", 1)
            if code not in CODE_ORDER or not value or code in values:
                raise ValueError(f"invalid or duplicate code: {code}")
            values[code] = value
            observed.append(code)
        # Field order is requested from the model to stabilize generation, but
        # decoding is order-independent because the compact language is keyed.
        result = {}
        if "C" in values:
            if values["C"] not in base.VOCAB["category"]: raise ValueError("invalid category")
            result["category"] = values["C"]
        if "K" in values:
            if not re.fullmatch(r"[a-z][a-z -]*", values["K"]): raise ValueError("invalid color")
            result["color"] = values["K"]
        if "P" in values:
            result["position"] = POSITION_DECODE[values["P"]]
        if "F" in values:
            result["facing"] = FACING_DECODE[values["F"]]
        if "O" in values:
            result["occlusion"] = OCCLUSION_DECODE[values["O"]]
        if "D" in values:
            result["distance_max_m"] = float(Decimal(values["D"]))
        for code, dimension in (("L", "length"), ("W", "width"), ("H", "height")):
            if code not in values: continue
            if values[code].count(":") != 1: raise ValueError(f"invalid range: {code}")
            lo, hi = (Decimal(x) for x in values[code].split(":"))
            if not lo.is_finite() or not hi.is_finite() or lo < 0 or hi < lo:
                raise ValueError(f"invalid range: {code}")
            result[f"{dimension}_min_m"] = float(lo)
            result[f"{dimension}_max_m"] = float(hi)
        return result, None
    except (ValueError, KeyError, InvalidOperation) as error:
        return None, str(error)


def prompt_hash():
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

