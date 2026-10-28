"""最小复现：为什么 appearance 等值匹配 GT 却判不匹配。"""
from code.datasets.rope3d import parse_repr
from methods.baseline.matcher import _color_matches
from methods.baseline.attributes import from_properties, _split_kv

props = ['length : 4.0 to 4.4 m', 'facing_direction : facing away from me', 'objectName : car', 'appearance : silver-grey']
gt_repr = "['car', 0, 0, 0.0, 831.0, 295.0, 906.0, 363.0, 1.24, 1.72, 4.22, -3.1, -5.4, 70.5, -1.57, 'silver-grey']"

c = from_properties(props)
gt = parse_repr(gt_repr)

print("c.appearance =", repr(c.appearance))
print("gt.color     =", repr(gt.color))
print("_color_matches(c.appearance, gt.color) =", _color_matches(c.appearance, gt.color))

# 直接查原始 label 里 raw[15] 到底是什么
import ast
raw = ast.literal_eval(gt_repr)
print("raw[15]      =", repr(raw[15]), " type:", type(raw[15]))
