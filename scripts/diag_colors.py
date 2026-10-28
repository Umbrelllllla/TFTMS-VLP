"""统计训练集目标颜色字段的取值分布（确认颜色规范化拆分方案覆盖）。"""

from __future__ import annotations

import os
from collections import Counter

from code.datasets.vlmod_dataset import iter_train

cnt = Counter()
sample_cnt = 0
for sample in iter_train():
    sample_cnt += 1
    for o in sample.candidate_pool:
        cnt[o.color] += 1

print(f"样本数={sample_cnt}  颜色取值数={len(cnt)}")
for k, v in cnt.most_common(40):
    print(f"  {k!r}: {v}")
