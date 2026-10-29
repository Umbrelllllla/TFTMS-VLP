# A 模型 test 推理与提交文件

用户已将test推理方案从B改为A。B只保存了网页快照，没有启动test模型推理。

2026-10-18重新读取任务页 http://vlp.chd.edu.cn/tasks/2 ，确认要求每个JSON对应同名TXT；N行对应test_data顺序，3列对应public_description顺序，空格分隔0/1。最终ZIP内部为result/目录和全部TXT。网页快照位于runs/qwen2b_test_A/task_page.html。

本地test共300个JSON、4220个候选、12660次匹配判断。使用已完成240对扩大验证的A模型：原始基础模型＋A的LoRA参数，固定原提示词、特征计算、NF4量化、非思考贪心生成、最多8个词元。没有重新训练、放宽数值边界或用规则替模型回答。输入只包括当前描述和候选参数；图像只读取JPEG尺寸头以计算位置特征。

运行前验证基础权重、适配参数、特征与提示词来源及300份test JSON指纹，并检查旧验证样本的输出复现。每道原始回答即时保存，每个完整文件写入result。非法回答会保留并标记，不能直接当成0补齐；存在非法回答时不生成最终ZIP。

运行或断点继续，在项目根目录执行同一条命令：

```powershell
.\.venv-qwen2b\Scripts\python.exe -u scripts/qwen2b/predict_test_a.py run
```

已启动时不要重复启动；进程锁阻止并发写同一任务。停止或断电后再次运行会校验输入与权重，跳过已保存回答，不修改原始数据。

产物在runs/qwen2b_test_A：

- status.json：进度、非法输出数量和剩余时间估计。
- responses/：每个候选、每条描述的模型原始回答与耗时。
- result/：已完成的同名TXT。
- submission_A.zip：全部完成并验证通过后生成。
- complete.json：成功标志status=PASS，以及ZIP指纹。
- failure.json：发生异常时的记录；恢复后看status.json判断当前状态，旧错误记录可能仍保留。

ZIP生成时核对全部300个文件、行列数量、0/1取值、原始回答映射及ZIP内部内容。不会自动上传网站；本地没有官方测试答案，因此不报告test正确率或排行榜得分。预计用时仅按此前实测估算，受电脑负载和休眠影响。
