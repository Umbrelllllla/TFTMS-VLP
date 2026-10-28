"""查剩余 S=0：facing 未命中的 28 条是什么情况（朝向规则问题 vs 数据噪声）。"""
from __future__ import annotations
import math, os
from code.datasets.vlmod_dataset import default_data_root, find_annotation_files, load_train_sample
from methods.baseline.attributes import from_properties
from methods.baseline.matcher import MatchConfig, attribute_matches, _heading, _toward_camera

def angle_deg(o):
    h = _heading(o); t = _toward_camera(o)
    dot = h[0]*t[0]+h[1]*t[1]+h[2]*t[2]
    return math.degrees(math.acos(max(-1,min(1,dot))))

def main():
    ds_root = default_data_root()
    img_dir = os.path.join(ds_root, "img","train"); ann_dir = os.path.join(ds_root,"MonoMulti3D","train")
    with open("data/splits/train.txt", encoding="utf-8") as fh:
        stems=[ln.strip() for ln in fh if ln.strip()]
    anns=find_annotation_files(ann_dir); pairs=[(s,anns[s]) for s in stems if s in anns]
    cfg=MatchConfig()
    n=0
    for stem,ann_path in pairs:
        sample=load_train_sample(ann_path,img_dir)
        if sample is None: continue
        for e in sample.expressions:
            if not e.matched_objects: continue
            gt=e.matched_objects[0]
            c=from_properties(e.public_properties)
            if c.facing is None: continue
            am=attribute_matches(c,gt,cfg)
            if not am.get("facing", True):
                n+=1
                if n<=28:
                    print(f"[{stem}/{e.ann_id}] facing={c.facing!r}  θ={angle_deg(gt):.1f}°  ry={gt.ry:.2f}")
                    print(f"   gt={gt}")
                    print(f"   props={e.public_properties}")
    print(f"\n共 {n} 条 facing 未命中")

if __name__=="__main__":
    main()
