# 本机 Qwen3.5-2B 环境准备

2026-10-15 实测：RTX 3050 Laptop，6144 MiB 显存，驱动 580.97，项目盘剩余约 303GB。原 `.venv` 保持原状，新建 `.venv-qwen2b` 专供推理。

## 当前停在哪

最新状态：2026-10-16 已完成官方权重下载及校验，首次 12 题模型试跑也已完成。格式 12/12 合法、答案 6/12 一致，显存峰值约 1.71 GiB（PyTorch 分配量）。详见[试跑记录](../runs/qwen2b_smoke_20261016_224244/README.md)。以下保留安装排错经过，末尾“安装完成之后”描述初始试跑设计，其尚未运行的描述现已被此结果取代。

2026-10-16 修复进展：依赖已安装，直接 `import torch` 触发 c10.dll / WinError 1114。Anaconda 根目录的 C++ 运行库版本为 14.27，系统目录为 14.50；清理 PATH 无效，进程内预加载系统运行库后 PyTorch 2.10.0+cu128 导入成功且 CUDA 可用。`windows_runtime.py` 已加入环境检查和模型试跑入口，不覆盖系统文件、不改 Anaconda、不更换 PyTorch。裸 `python -c "import torch"` 不经过此入口，因此仍可能报原错。

实际检查已通过 `CUDA_NF4_CHECK_PASS`，包括 Qwen 模型类导入以及 GPU 上的 NF4 四位量化计算。第一次完整检查耗时数分钟，已增加分阶段日志，长时间无进展时每 60 秒输出调用栈供诊断。

可单独检查已安装环境（不下载模型）：

```powershell
.\.venv\Scripts\python.exe scripts/qwen2b/setup.py --check-only
```

检查结果保存为 `runs/qwen2b_setup/runtime_check.json`；完整安装命令仍为下文原命令。以下为 10 月 15 日的下载交接记录，安装状态以上述更新为准。

已建立新环境，下载并校验官方配置、分词器等小文件。PyTorch 和模型权重的大文件下载因持续低速手动停止；不是已确认的显存不足或软件不兼容。模型 `.part` 保留，可断点续传。PyTorch 安装未完成，不能声称环境可用；其重试是否复用大包下载取决于 uv 缓存。

官方权重约 4.55GB，PyTorch 下载约 2.7GiB。模型四位量化在加载时进行，因此需要先下载原始权重。当前速度每秒几百 KB，下载可能需要数小时。

## 你需要执行的命令

在项目根目录 `E:\Projects\TFTMS-VLP` 的终端执行：

```powershell
.\.venv\Scripts\python.exe scripts/qwen2b/setup.py
```

它按顺序完成：

1. 安装带 CUDA 的 PyTorch 到独立环境。CUDA 是让模型使用 NVIDIA 显卡运算的软件接口。
2. 安装固定版本的模型加载和量化工具，并检查依赖关系。
3. 实际执行一次 GPU 四位量化计算，成功打印 `CUDA_NF4_CHECK_PASS`。
4. 下载固定版本官方模型，逐个核对发布方的文件指纹，模型断点文件续传。
5. 全部成功后打印 `SETUP_COMPLETE`。本命令不训练，也不自动执行模型答题。

日志保存在 `runs/qwen2b_setup/setup.log`。成功后告知助手；失败时提供日志末尾错误，不必盲目重复安装。网络断开后可以重跑此命令，已校验的完整模型文件会跳过。

## 固定的软件和模型

- PyTorch 2.10.0，官方 cu128 包；不安装全局 CUDA Toolkit、不改驱动。
- Transformers 5.19.0、Accelerate 1.15.0、bitsandbytes 0.50.2。它们分别负责加载模型、设备管理、四位量化。
- 模型：Qwen/Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
- 模型位置：`weights/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/`。
- 实际安装的完整依赖版本在成功安装后写入 `runs/qwen2b_setup/installed.freeze.txt`。根项目的 `uv sync` 不会重建这套独立环境，需用上面的专用命令。

参考：[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)、[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[bitsandbytes 平台支持](https://huggingface.co/docs/bitsandbytes/main/en/installation)。以上依赖组合尚待本机验证。

## 安装完成之后

助手将先检查安装日志，再执行 `scripts/qwen2b/smoke.py`。这个程序已经写好，但尚未运行或验证模型加载兼容性。它只用前 12 道训练侧题目，纯文本、NF4 四位权重、每次一题、关闭思考输出、最多生成 8 个 token（模型处理文字的小单位）。

模型输出后才读取答案。程序记录原始输出、非法格式、耗时和显存。没有将模型输出限制为只能选择 0/1，因而能发现它不遵守格式的情况。它不是正式开发验证，也不能报告比赛成绩。
