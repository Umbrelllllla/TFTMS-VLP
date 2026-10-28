"""inspect_json.py —— 数据加载验证脚本。

把训练集/测试集各一个样本的"目标 + 文本"清晰打印出来，
用于确认：字段解析正确、图片已对上、候选池构建正常。

用法（在 D:/TFTMS-VLP 根目录）：
    uv run python -m scripts.inspect_json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.datasets.vlmod_dataset import (
    default_data_root, iter_train, iter_test,
)


def main() -> None:
    root = default_data_root()
    print(f"数据根目录: {root}\n" + "=" * 60)

    # 训练集样例
    train = list(iter_train())
    print(f"[训练集] 有效样本数: {len(train)}")
    if train:
        s = train[0]
        print(f"  图片: {s.image_path}")
        print(f"  候选目标池: {len(s.candidate_pool)} 个 | 标定: {bool(s.calib)}")
        for e in s.expressions:
            print(f"\n  表达式 ann_id={e.ann_id}")
            print(f"    文本: {e.description}")
            print(f"    属性: {e.public_properties}")
            print(f"    匹配目标(GT): {len(e.matched_objects)} 个")
            for o in e.matched_objects[:3]:
                print(f"      - {o}")

    print("\n" + "=" * 60)

    # 测试集样例
    test = list(iter_test())
    print(f"[测试集] 有效样本数: {len(test)}")
    if test:
        s = test[0]
        print(f"  图片: {s.image_path}")
        print(f"  文本({len(s.descriptions)} 条):")
        for t in s.descriptions:
            print(f"    - {t}")
        print(f"  候选目标: {len(s.candidates)} 个")
        for o in s.candidates[:3]:
            print(f"    - {o}")


if __name__ == "__main__":
    main()
