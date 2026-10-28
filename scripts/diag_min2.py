"""检查 silver-grey 里是否有不可见/非ASCII字符。"""
from code.datasets.rope3d import parse_repr
from methods.baseline.attributes import from_properties

props = ['appearance : silver-grey']
gt_repr = "['car', 0, 0, 0.0, 1, 2, 3, 4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 'silver-grey']"
c = from_properties(props)
gt = parse_repr(gt_repr)

def show(tag, s):
    print(f"{tag}: {s!r}  len={len(s)}  codes={[hex(ord(ch)) for ch in s]}")

show("c.appearance", c.appearance)
show("gt.color    ", gt.color)
print("equal:", c.appearance == gt.color)

# 也看看 '|' 判定
import inspect, methods.baseline.matcher as M
src = inspect.getsource(M._color_matches)
print("\n实际运行的 _color_matches 源码:\n", src)
