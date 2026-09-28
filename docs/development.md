# 团队开发规范

- `main` 保存稳定公共代码，不直接在 `main` 开发。
- 每人维护自己的 `member/name` 分支，例如 `member/alice`。
- 实验分支从个人分支创建，例如从 `member/alice` 创建 `experiment/alice/MonoMulti-baseline`。
- 实验完成后整理变更；稳定代码经过检查和团队审查，再通过 Pull Request 合并到 `main`。
- 公共逻辑放在 `code/`，具体方法放在 `methods/` 的独立目录中，配置放在 `configs/`。
- 不提交 `.venv/`、数据、权重、checkpoint 或 `runs/` 实验结果。

当前阶段只维护工程骨架，不实现训练、模型或数据处理逻辑，不引入复杂框架。
