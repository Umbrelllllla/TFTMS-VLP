# 环境说明

- Python 要求：`>=3.10`。
- 环境与依赖管理：uv。
- 已生成 `pyproject.toml` 和 `uv.lock`，项目名称由 uv 生成为 `tftms-vlp`。
- 当前依赖列表为空，不安装 torch 或 transformers。

## 同步与检查

已有项目无需再次执行 `uv init`。安装 uv 后，在仓库根目录执行：

```sh
uv sync --locked
uv run python --version
uv run python -c "import sys; print(sys.executable)"
```

`uv sync --locked` 创建或同步项目的 `.venv/`。检查 Python 版本满足 `>=3.10`，解释器路径位于当前仓库的 `.venv/`。

本次本地初始化使用 uv 0.12.19，以 `D:\Anaconda\python.exe`（CPython 3.11.7）为基础创建 `.venv/`，未启用系统 site-packages。该路径只记录本机环境，其他成员不需要使用相同路径。

后续在仓库内通过 `uv run python ...` 使用项目环境，无需手动激活 `.venv/`。

`pyproject.toml` 和 `uv.lock` 应纳入版本控制，`.venv/` 不提交。后续添加依赖由团队另行确定。
