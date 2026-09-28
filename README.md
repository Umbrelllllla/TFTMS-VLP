# TFTMS-VLP

面向 VLP Challenge、VLMOD 和 3D Visual Grounding 的长期维护深度学习科研仓库，用于复现论文方法、保存多方法实验、开发改进方法和团队协作。

技术选择为 Python、PyTorch 生态、uv 环境管理及 GitHub 团队协作。当前仅初始化工程骨架，不包含训练、模型或数据处理实现，也不安装深度学习依赖。

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `configs/datasets/` | 数据集配置 |
| `configs/methods/` | 方法配置 |
| `configs/experiments/` | 实验配置 |
| `data/` | 本地数据，不提交数据文件 |
| `weights/` | 本地权重与 checkpoint，不提交权重文件 |
| `runs/` | 实验输出，不提交实验结果 |
| `external/` | 外部方法或第三方代码的存放位置 |
| `docs/` | 环境、数据集、实验和开发规范 |
| `scripts/` | 训练、测试、推理脚本的空占位 |
| `code/` | 公共入口及共享的数据集、执行流程、工具目录 |
| `methods/` | 各方法的独立目录；`template/` 为新增方法的空模板 |

## uv 环境

Python 版本要求为 `>=3.10`，使用 uv 管理环境和依赖。

已使用 uv 初始化，`pyproject.toml` 和 `uv.lock` 已生成，依赖列表为空。团队成员按 [环境说明](docs/environment.md) 同步本地环境。暂不添加或安装 torch、transformers 等深度学习依赖。

## Git 协作

- `main`：保存稳定公共代码，不直接开发。
- `member/*`：个人开发分支，每位成员维护 `member/name`。
- 实验分支从个人分支创建；实验整理后回到个人分支。
- 稳定代码经检查与团队审查后，通过 Pull Request 合并至 `main`。
- 不提交本地环境、数据、权重、checkpoint 和实验输出。

详细约定见 [开发规范](docs/development.md) 和 [实验规范](docs/experiment.md)。
