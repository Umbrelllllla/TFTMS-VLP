# 实验规范

实验输出统一放在 `runs/`。

目录命名规则：

```text
YYYYMMDD_method_description
```

示例：

```text
runs/20260928_MonoMulti_baseline/
```

每次实验使用独立目录，避免覆盖已有结果。实验记录应包含方法名称、数据集版本与划分、配置、代码版本和结果，便于复现与对比。

实验配置保存在 `configs/experiments/`。`runs/` 内的实验结果不提交到 Git。
