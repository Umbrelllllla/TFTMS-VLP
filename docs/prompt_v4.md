# V4 明确字段对应实验

本轮由用户提出：把类别、颜色、尺寸、距离、位置、朝向、遮挡各自应该匹配哪个字段讲清楚，并加入判断规则。

实验已完成：V4正确89/120，漏选15、错选16；V2为100/120，漏选0、错选20。V4修正6题、退步17题，暂不采用为默认版本。完整记录见[结果报告](../runs/qwen2b_prompt_diagnostic_v4/README.md)。

只在已保存的 V2 系统提示词末尾追加规则，原有120题顺序、描述、候选字段和值全部不变；程序已逐条断言用户消息与 V2 完全一致。没有重新抽样，没有增加例题，没有训练。模型仍为 Qwen3.5-2B、NF4四位量化、非思考模式、每次一题、最多输出8个token。

新增规则的中文含义：

| 描述要求 | 对应字段或规则 |
|---|---|
| car / van / truck / bus | type；明确类别精确匹配，泛称vehicle不限定为car |
| 颜色 | color；排除被明确否定的颜色 |
| 距离 | distance_m；不混用深度z、车长或像素距离 |
| 长、宽、高 | dimensions_m中的length、width、height |
| 不超过、至少、大于、小于、区间 | 分别使用≤、≥、>、<及同时满足上下界 |
| 画面位置 | image_region；沿用框中心与三等分九宫格，不重新用x2/y2或二等分定义 |
| 朝向 | facing_convention；完整匹配朝向短语，不混淆朝向与画面位置 |
| 遮挡 | occlusion或occlusion_code；不把truncation_code当作遮挡 |
| 并且、或者、否定 | 按条件逻辑处理，否定只作用于其修饰条件；其余条件仍需满足 |

未被描述提及的属性不作排除条件；参照物的属性不强行套给候选目标。保持原始数值，不增加人为距离容差。

否定和或者是统一追加的规则，本轮没有专门补充这两类样例，所以不能单独评判这两条规则的效果。整体变好或变差也不能归因于某一条规则。

## 运行与复核

题目已经冻结。项目根目录运行：

```powershell
.\.venv-qwen2b\Scripts\python.exe -u scripts/qwen2b/diagnose_v4.py run
.\.venv\Scripts\python.exe scripts/qwen2b/diagnose_v4.py report
```

输出在 `runs/qwen2b_prompt_diagnostic_v4`，被Git忽略。`added_rules.txt`是新增规则，`full_system_prompt.txt`是完整提示词，`tasks.jsonl`保存实际输入。`README.md`和`metrics.json`在完成后记录结果，`review_cases.json`记录V2/V4逐题对照。

旧版诊断数据和结果保持原状，manifest记录其指纹，报告前再次核对。两轮的模型版本、依赖、量化和生成设置也会逐项核对。

这是已经反复用于提示词设计的训练侧诊断集，不能称为独立验证。重点看V2原先20个错选修正了多少，同时报告新产生的漏选和退步题，避免只看总体正确数。
