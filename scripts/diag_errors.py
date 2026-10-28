"""分析当前版规则 baseline 的错误案例结构（errors_dev.jsonl）。"""

import json
import os

proj = r"D:\TFTMS-VLP"
with open(os.path.join(proj, "runs", "errors_dev.jsonl"), encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh]

print(f"错误案例总数 {len(rows)}\n")

fp_total = fn_total = 0
fp_all_hit = 0   # FP 目标所有属性命中（规则认为完全匹配但 GT 不认）
fp_part_hit = 0  # FP 目标部分属性命中
for r in rows:
    fp = r.get("fp") or []
    fn = r.get("fn") or []
    fp_total += len(fp)
    fn_total += len(fn)
    scored = r.get("scored") or []
    for fp_obj in fp:
        # 找到该对象的得分条目
        hit_attrs = []
        for s in scored:
            if s["obj"][0] == fp_obj[0] and abs(s["obj"][11] - fp_obj[11]) < 1e-6 and s["obj"][10] - fp_obj[10] < 1e-6:
                hit_attrs = [k for k, v in s["attrs"].items() if v]
                break
        # 描述中提到的属性（非 None 约束）
        c = r.get("constraints") or {}
        mentioned = [k for k, v in c.items() if v is not None and k not in ("objectname",) and v not in ("", [None, None])]
        hit_mentioned = [k for k in hit_attrs if k in mentioned or k == "objectname"]
        if len(hit_attrs) >= 3 and all(k in hit_attrs for k in mentioned if k != "objectname"):
            fp_all_hit += 1
        else:
            fp_part_hit += 1

print(f"FP 合计 {fp_total}（30 条错误表达式中）")
print(f"  其中：规则判定『全部描述属性命中』仍被 GT 排除（GT 主观歧义）: {fp_all_hit}")
print(f"  其中：规则判定『部分属性未命中』仍被选中（阈值/解析问题）: {fp_part_hit}")

# 逐条看描述提到哪些属性（约束非空）
print("\n--- 每条错误：描述提到的属性 ---")
for r in rows[:30]:
    c = r.get("constraints") or {}
    mentioned = [k for k, v in c.items() if v is not None and v not in ("", [None, None])]
    fp = len(r.get("fp") or [])
    fn = len(r.get("fn") or [])
    print(f"FP={fp} FN={fn}  属性={mentioned}  desc={r['description'][:80]}")
