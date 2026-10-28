# Rope3D / MonoMulti3D-VLMOD 数据口径（团队统一规范）

> 本文件是团队复现《Beyond Human Perception》（CVPR 2025, MonoMulti3D-ROPE）时的**统一数据口径**。
> 所有坐标、单位、字段语义、属性映射、朝向与距离规则，一律以此处为准；出现分歧以本文件为准并回到本文件更新。
> 全部结论均经**官方来源（Rope3D 论文 / 官方代码 `liyingying0113/rope3d-dataset-tools`）核对**或数据实测交叉验证，逐项标注证据级别。

---

## 1. 数据布局与 JSON 结构

本地数据根目录默认 `D:/VLMOD`（可用环境变量 `VLMOD_DATA_ROOT` 覆盖）：

| 子目录 | 内容 |
|---|---|
| `img/train` | 训练图像（4000 张） |
| `img/test` | 测试图像（300 张） |
| `MonoMulti3D/train` | 训练标注（3999 个 JSON，每个对应一张图 + 3 条表达式） |
| `MonoMulti3D/test` | 测试标注（300 个 JSON，无 GT 标签） |
| `MonoMulti-3DVG-main` | 官方 baseline 开源代码（仅 `libs.py`，含 KANLinear/KAN/LatentParams/reparameterize 四件"积木"，不含模型主体） |

### 训练集 JSON（顶层 `list`）
```
[ [表达式记录0, 表达式记录1, 表达式记录2], [标定] ]
```
- `[0]`：3 条表达式记录，每条含 `ann_id` / `image_file_name` / `public_properties`（结构化的属性键值列表）/
  `public_description`（自然语言描述）/ `label_3`（匹配目标 GT，**字符串形式**，需 `ast.literal_eval` 还原为 16 元素列表）。
- `[1]`：相机标定 `{calib, denorm}`。

### 测试集 JSON（顶层 `dict`）
```
{ public_description: [3 条自然语言], test_data: [N 个候选，每行 16 个空格分隔字段] }
```
- 无 GT 标签、无 `public_properties`——规则 baseline 需从 `public_description` 自由文本自行抽取属性约束。

### 16 字段顺序（KITTI 风格，统一为 `Rope3DObject`）
```
class  trunc  occ  alpha  x1  y1  x2  y2  h  w  l  x  y  z  ry  color
[0]    [1]   [2]   [3]   [4]  [5] [6]  [7]  [8] [9] [10][11][12][13][14] [15]
```

---

## 2. 坐标系与单位（已核对，证据=官方代码 `show_2d3d_box.py` + 数据实测）

- **单位：米（meter）**，所有 h/w/l/x/y/z 均为米制。官方代码加载注释明确 `X [m], Y [m], Z [m]`。
- **坐标系：相机坐标系**（camera coordinates，Rope3D/KITTI 惯例），轴方向：
  - **x 轴**：向右（图像右方），`x>0`=目标在画面右侧，`x<0`=左侧。
  - **y 轴**：向下（垂直方向）。
  - **z 轴**：相机前方深度方向（forward/depth）。
- **存储的 `(x, y, z)` = 目标的 3D 框中心**，非底部中心、非角点。
  - 证据：官方 `project_3d` 中 8 角点以 `(x,y,z)` 为中心平移（y 从 `-h/2` 到 `+h/2`）。
  - 交叉验证：用 `calib` 的 P2 把 3D 中心投到像素，与 2D 框中心误差平均 11.5px(u)/33.1px(v)，量级吻合。

### `calib`
P2 相机内参，**行主序 3×4**：`[fu, 0, cu, 0, 0, fv, cv, 0, 0, 0, 1, 0]`。
投影：`u = fu·x/z + cu`，`v = fv·y/z + cv`。

### `denorm`（重要：它不是坐标归一化参数）
`denorm = (α, β, γ, d)` 是相机坐标系下的**地平面方程 `G: αx + βy + γz + d = 0`** 的四个系数，
即地面平面的**法向量+偏移**。官方 `compute_c2g_trans` 把它当作地面法向量，用于**相机坐标系 ↔ 地面坐标系的旋转**、把 3D 框投到地面、算地面距离（评测指标 ACS/AGD 的做法）。
**结论：Rope3D 标注本身不存在"深度归一化/去归一化公式"——x/y/z 就是可直接使用的真实米制。** `denorm` 只服务于"投地面/算地面距离"，对目标匹配非必需。

---

## 3. 描述属性 ↔ 字段映射（键名实测 + 匹配规则）

训练集 `public_properties` 实测属性键全集：
`length / distance / width / objectname / occlusion / place / facing_direction / height / appearance`

| 描述属性（键） | 用哪个字段 | 精确匹配规则 |
|---|---|---|
| `objectname` | 字段[0] `class` | car / van / bus / truck / pedestrian / cyclist 等值 |
| `occlusion` | 字段[2] `occ` | `no occluded`↔occ=0；`partially occluded`↔occ=1 等 |
| `place`（top/left/right/bottom/center 等 9 值） | 字段[11] `x` 的符号/相对位置 | **left↔x<0、right↔x>0（实测 100% 干净）**；top↔y 偏小、bottom↔y 偏大（纵向符号 y 向下） |
| `length` / `width` / `height` | 字段[10]`l` / [9]`w` / [8]`h`（米） | 描述多为区间 `"4.1 to 4.5 m"`，判断 `值 ∈ [lo−tol, hi+tol]` |
| `distance` | 欧氏 `sqrt(x²+y²+z²)` | ≈描述值，**宽容差 ±20%~50%**（见 §4） |
| `facing_direction` | 字段[14] `ry` | 用**车头向量夹角**判断，非单角度硬判（见 §5） |
| `appearance`（颜色） | 字段[15] `color` | **规范化集合交集**：约束端与目标端都拆成规范化色系集合，交集非空即命中（见下） |

> 说明：描述属性键在训练集是 `facing_direction`/`appearance`，**不是** `orientation`/`color`。

### 颜色（appearance）匹配的精确规则（2026-10-24 修订）

数据里颜色既可能单值也可能复合/带修饰（`silver-grey` / `yellow-orange` / `dark-brown`），约束端与目标端写法可不对称。**唯一正确做法：两端都按 `[-/|空格]` 拆词、逐词映射到规范化色系集合，两端集合交集非空即命中。**

- 规范化映射（`_normalize_color_word`）：`silver→gray`、`grey→gray`、`navy→blue`、`dark→black` 等，见 `attributes.py::_COLOR_SYNONYMS`。
- 例：约束 `silver-grey` → `{gray}`，目标 `silver-grey` → `{gray}` → 命中 ✓
- 例：约束 `gray` → `{gray}`，目标 `silver-grey` → `{gray}` → 命中 ✓
- 例：约束 `yellow-orange` → `{yellow,orange}`，目标 `orange` → `{orange}` → 命中 ✓
- 例：约束 `white` → `{white}`，目标 `black` → `{black}` → 不命中 ✗

> ⚠️ 教训（2026-10-24）：此前一版只规范化**目标端**、约束端用原始串 `in` 集合，导致"复合色约束 ↔ 复合色目标"判不中（`约束=silver-grey` 对 `目标=silver-grey` 反而 False），dev 有 309 条表达式 GT 因此匹配不上。**两端必须同样规范化、判交集**（`methods/baseline/matcher.py::_color_matches`）。

---

## 4. 距离 `distance` 的精确定义

- **定义：目标 3D 中心到相机的欧氏距离 `sqrt(x²+y²+z²)`**（单位米）。
- 证据：数据实测——25% 的表达式存在匹配目标欧氏距离与描述值**精确相等**（相对误差 0.00）；
  中位最佳相对误差 12%；±20% 内覆盖 60%、±50% 内覆盖 83%。
- **含义**：生成描述时是"随机取参考距离 + 宽松匹配"，所以规则匹配必须用**宽容差带**，不能用点值硬判。
  建议容差：`|sqrt(x²+y²+z²) − d| ≤ max(0.20·d, 常数)`（可调）。

---

## 5. 朝向 `ry` 的定义（官方代码盖章）与匹配规则

### ry 的精确定义（官方 `show_2d3d_box.py`，逐字引用）
绕竖直轴 **Y 轴**旋转（标准 KITTI yaw），旋转矩阵：
```
R_y(ry) = [[+cos ry, 0, +sin ry],
           [0,       1,  0      ],
           [−sin ry,  0, +cos ry]]
```
目标的**车头方向（+l，即长轴方向）**旋转后指向相机坐标系的单位向量为 **`heading = (cos ry, 0, −sin ry)`**
（相机坐标系 x 右 / y 下 / z 前）。

### ry → 语义映射（官方公式推导，非经验猜测）
| ry | heading=(cos ry,0,−sin ry) | 语义 |
|---|---|---|
| **90° (π/2)** | (0,0,−1) 指向相机 | **facing me（正对/朝向相机）** |
| **270° (−90°)** | (0,0,+1) 背离相机 | **facing away（背对相机）** |
| **0° / 180°** | (±1,0,0) 侧向 | **side facing（侧面）** |

### 匹配规则（不要用单角度硬判）
数据存在**语义离散**（例如一辆 45° 斜向朝相机驶来的车在语义上确实属于"facing me"），
所以正确做法是**用"车头向量与'指向相机方向'的夹角"做阈值**，而非比较 ry 是否等于某固定角：
- 指向相机方向（归一化）：`toward_cam = normalize(0 − obj_center) = normalize(−x, −y, −z)`（近似取 `(0,0,−1)`）。
- 夹角 `θ = arccos( heading · toward_cam )`，`heading=(cos ry,0,−sin ry)`。
- `facing me`：`θ ≤ 45°`；`facing away`：`θ ≥ 135°`；`side facing`：`45° < θ < 135°`（阈值可按内部验证集微调并冻结）。

---

## 6. 数据划分（图片级，防泄露）

- **按图片（scene）级**划分：同一张图的 3 条表达式必须同组，不允许同一图横跨开发/验证。
- 默认 **80% 开发 / 20% 内部验证**，seed 固定可复现，列表写入 `data/splits/train.txt` / `data/splits/val.txt`。
- 划分脚本：`scripts/split_dataset.py`。
- **冻结规则**：所有匹配阈值、容差、超参**只在开发集上调，调完冻结**，最后只在冻结的验证集上报一次，
  以平台测试集（`MonoMulti3D/test`）为最终裁判。禁止用验证集反复调参（防过拟合/偏置）。

---

## 7. 待确认（需平台 `vlp.chd.edu.cn` / 官方评估工具）

- 比赛最终提交的 JSON schema（目标 ID、预测参数、置信度）。
- 评测指标 `Accuracy@0.7` / `ACE` / `Slot_edit_f1_full` 的精确计算口径与匹配的目标集合定义。
- 预测输出矩阵方向（N 目标 × 3 文本，或反之）。

---

## 8. 复现入口

- 数据加载：`code/datasets/rope3d.py`（16 字段 + 解析）、`code/datasets/vlmod_dataset.py`（训练/测试样本 + 候选池 + 生成器）。
- 运行：`uv run python -m scripts.inspect_json`（校验解析）；环境见 `docs/environment.md`。
