# STCross

基于时空交叉特征的异常客流预测框架，针对节假日、极端天气等特殊事件场景下的客流量预测。

## 项目目标

常规客流预测模型在节假日、极端天气等异常场景下表现不佳，因为这些场景的客流模式与日常显著不同。本项目旨在：

1. 构建节假日、极端天气等外部事件的结构化表征（掩码机制）
2. 将外部事件特征与时空客流数据进行交叉融合
3. 提升模型在异常场景下的客流预测精度
4. 支持跨城市迁移学习（城市 A 训练 → 城市 B 预测）

## 项目结构

```
STCross/
├── README.md                       # 项目说明
├── requirements.txt                # Python 依赖
├── exp_log.md                      # 实验日志
│
├── data/
│   ├── raw_data/                   # 原始数据（未纳入版本管理）
│   │   ├── traffic_flow/           #   交通流量 .pkl（UCTB 格式）
│   │   ├── weather_his/            #   历史天气 .csv
│   │   └── weather/                #   天气预报 .parquet
│   └── processed_data/             # 预处理后数据 .npy
│       ├── traffic_flow/           #   flow_{city}_{start}_{end}_{freq}min.npy
│       ├── holiday/                #   mask_{city}_holiday_*.npy / mask_{city}_weekend_*.npy
│       └── weather/                #   mask_{city}_all_*.npy
│
├── data_preprocess/                # 数据预处理与数据集构建
│   ├── build_flow_data.py          #   客流数据提取与裁剪
│   ├── build_holiday_mask.py       #   节假日/周末掩码生成
│   ├── build_weather_mask.py       #   极端天气掩码生成
│   ├── dataset.py                  #   数据加载、滑动窗口、时间划分、子集筛选
│   ├── normalizer.py               #   PerNodeStandardScaler 标准化
│   └── README.md                   #   预处理模块详细文档
│
├── model/                          # 模型定义
│   ├── __init__.py
│   ├── base.py                     #   BaseModel 抽象基类
│   └── ha.py                       #   HA (Historical Average) baseline
│
├── metrics/                        # 评测指标
│   ├── __init__.py
│   └── metrics.py                  #   MAE, RMSE, MAPE, PCC, R²
│
├── eval/                           # 评测算法与管线
│   ├── __init__.py
│   ├── config.py                   #   EvalConfig 配置数据类
│   ├── evaluator.py                #   主评测管线（数据加载→模型推理→指标计算）
│   └── reporter.py                 #   结果打印 + JSON 保存
│
├── train/                          # 训练算法（待实现）
│   └── __init__.py
│
├── experiments/                    # 实验入口脚本
│   ├── run_ha_baseline.py          #   HA baseline 实验
│   └── run_model.py                #   通用模型评测 CLI
│
├── results/                        # 评测结果输出（JSON）
│
├── papers/                         # 参考文献与阅读笔记
│   ├── ST00_note.md                #   综合阅读笔记
│   └── ST{01-13}_note.md           #   各论文阅读笔记
│
└── utils/                          # 通用工具函数（待完善）
    └── __init__.py
```

## 数据基础

### 1. 交通客流数据

| 城市 | 数据来源 | 原始格式 | 时间粒度 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| NYC (纽约) | Bike sharing (UCTB) | `.pkl` | 60 min | 包含站点信息、流量矩阵、月度交互矩阵 |
| CHI (芝加哥) | Bike sharing (UCTB) | `.pkl` | 60 min | 同上 |

**pkl 数据结构**：
- `TimeRange`: 数据时间跨度
- `TimeFitness`: 时间步长（分钟）
- `Node.TrafficNode`: 流量矩阵，shape = `[time_steps, num_nodes]`
- `Node.StationInfo`: 站点元信息
- `Node.TrafficMonthlyInteraction`: 月度站点间交互矩阵
- `Grid.TrafficGrid`: 网格流量数据
- `Grid.GridLatLng`: 网格经纬度
- `ExternalFeature.Weather`: 天气外部特征

### 2. 天气数据

**历史天气数据**（用于生成极端天气掩码）：

| 城市 | 时间范围 | 原始格式 | 关键字段 |
| :--- | :--- | :--- | :--- |
| NYC | 2013.01.01 - 2024.01.01 | `.csv` | `valid_time_gmt`, `wx_phrase`, `precip_hrly`, `temp`, `wspd`, `gust`, `vis` 等 |
| CHI | 2013.01.01 - 2023.01.01 | `.csv` | 同上 |
| DC | 2013.01.01 - 2018.01.01 | `.csv` | 同上 |
| LA | 2012.01.01 - 2013.01.01 | `.csv` | 同上 |
| BAY | 2017.01.01 - 2018.01.01 | `.csv` | 同上 |
| MEL | 2021.01.01 - 2023.01.01 | `.csv` | 同上 |

**天气预报数据**（预留）：`data/raw_data/weather/*.parquet`，覆盖上述 6 个城市。

### 3. 节假日数据

基于规则生成，无需原始数据文件。覆盖范围：

- **美国联邦假日**（10 类）：New Year's Day, MLK Day, Presidents' Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas
- **选举日**：11 月第一个星期一之后的星期二
- **城市/州级假日**：NYC/CHI 的 Lincoln's Birthday, LA/BAY 的 Cesar Chavez Day
- **调休处理**：节假日遇周末自动调休至邻近工作日

### 4. 已处理数据（当前状态）

目前已完成 **NYC** 和 **CHI** 两个城市的数据预处理，时间范围统一为 `2014-10-01 ~ 2017-09-30`，时间粒度 `60 min`：

```
data/processed_data/
├── traffic_flow/
│   ├── flow_NYC_2014-10-01_2017-09-30_60min.npy    # shape=(26280, 820), int32
│   └── flow_CHI_2014-10-01_2017-09-30_60min.npy    # shape=(26280, 585), int32
├── holiday/
│   ├── mask_NYC_holiday_2014-10-01_2017-09-30_60min.npy   # shape=(26280,), int8
│   ├── mask_NYC_weekend_2014-10-01_2017-09-30_60min.npy
│   ├── mask_CHI_holiday_2014-10-01_2017-09-30_60min.npy
│   └── mask_CHI_weekend_2014-10-01_2017-09-30_60min.npy
└── weather/
    ├── mask_NYC_all_2014-10-01_2017-09-30_60min.npy       # shape=(26280,), int8
    └── mask_CHI_all_2014-10-01_2017-09-30_60min.npy
```

掩码数据均为 `np.int8` 类型，`1` 表示特殊事件，`0` 表示正常状态。

## 评测框架

### 实验范式

**跨城市迁移预测**：在 source 城市训练模型，在 target 城市评测预测性能。支持同城评测（source=target）和跨城迁移（source≠target）。

### 输入输出定义

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `history_len` (H) | 12 | 输入历史步数（12小时） |
| `predict_len` (P) | 12 | 预测未来步数（12小时） |
| `split_ratio` | (0.6, 0.2, 0.2) | train:val:test 划分比例 |
| `anomaly_min_ratio` | 0.5 | 异常标签阈值（见下文） |

- **滑动窗口**：样本 i → X = flow[i:i+H, :]，Y = flow[i+H:i+H+P, :]
- **数据划分**：按时间顺序切分，划分基于原始时间步 T
- **标准化**：Per-node StandardScaler，在 train split 上 fit，std=0 节点设 std=1.0
- 所有指标在**反标准化后的原始尺度**上计算

### 异常定义与标签

**anomaly = holiday OR weather**（weekend 不算异常）

样本级异常标签由 `anomaly_min_ratio` 参数控制：预测窗口 P 步中异常步占比 **超过** 该阈值时，样本标记为异常。

| `anomaly_min_ratio` | 含义 | NYC 异常占比 | CHI 异常占比 |
|---------------------|------|:------------:|:------------:|
| 0.0 | 任意一步异常即标记 | 22.8% | 21.8% |
| 0.25 | >25% 步数异常 | 13.0% | 10.9% |
| **0.5（默认）** | **多数步数异常** | **6.1%** | **4.0%** |
| 0.75 | >75% 步数异常 | 2.9% | 1.8% |
| 1.0 | 全部步数异常 | 1.2% | 0.9% |

默认使用 0.5（多数原则），确保异常子集反映真正的异常时段，符合异常事件稀疏性的研究假设。

### 评测子集

| 子集 | 说明 |
|------|------|
| full | 全部测试样本 |
| anomaly | 仅异常样本 |
| normal | 仅正常样本 |

### 评测指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MAE | mean(\|y - ŷ\|) | 平均绝对误差 |
| RMSE | sqrt(mean((y - ŷ)²)) | 均方根误差 |
| MAPE | mean(\|y - ŷ\|/\|y\|) × 100% | 平均绝对百分比误差（仅对 \|y\|>0.5 计算） |
| PCC | pearsonr(y, ŷ) | 皮尔逊相关系数 |
| R² | 1 - SS_res/SS_tot | 决定系数 |

> **MAPE 说明**：数据零值占比 66%-77%，MAPE 仅在 |y_true| > 0.5 的样本上计算，排除零值站点。因此 MAPE 仅作参考指标，MAE/RMSE 为主要指标。

### HA Baseline 结果

配置：H=12, P=12, split=6:2:2, anomaly_min_ratio=0.5

| 城市 | 子集 | 样本数 | MAE | RMSE | MAPE% | PCC | R² |
|------|------|--------|-----|------|-------|-----|-----|
| NYC | full | 5233 | 1.8926 | 4.2465 | 72.33 | 0.6784 | 0.3903 |
| NYC | anomaly | 318 | 1.5559 | 3.4676 | 78.50 | 0.6069 | 0.3395 |
| NYC | normal | 4915 | 1.9144 | 4.2920 | 71.96 | 0.6829 | 0.3918 |
| CHI | full | 5233 | 0.7379 | 2.2205 | 63.81 | 0.7438 | 0.4557 |
| CHI | anomaly | 210 | 0.7152 | 2.4251 | 72.41 | 0.5162 | 0.2601 |
| CHI | normal | 5023 | 0.7388 | 2.2115 | 63.52 | 0.7525 | 0.4628 |

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 数据预处理（如需重新生成）

```bash
# 提取客流数据（支持时间范围裁剪）
python data_preprocess/build_flow_data.py

# 生成节假日与周末掩码
python data_preprocess/build_holiday_mask.py

# 生成极端天气掩码
python data_preprocess/build_weather_mask.py
```

### 3. 运行 HA Baseline

```bash
# 默认配置：H=12, P=12, split=6:2:2, anomaly_min_ratio=0.5
python experiments/run_ha_baseline.py
```

### 4. 通用模型评测

```bash
python experiments/run_model.py \
    --model HA \
    --source NYC \
    --target NYC \
    --history_len 12 \
    --predict_len 12 \
    --split 0.6 0.2 0.2 \
    --anomaly_min_ratio 0.5 \
    --output_dir results/
```

### 5. 查看结果

评测结果保存为 JSON 文件到 `results/` 目录，命名格式为 `{model}_{source}_to_{target}_{timestamp}.json`。

## 添加新模型

1. 在 `model/` 目录下创建模型文件，继承 `model.base.BaseModel`
2. 实现三个接口：
   - `fit(train_X, train_Y, val_X, val_Y)` — 训练
   - `predict(X) → np.ndarray` — 推理，返回 shape=(num_samples, P, N)
   - `name` 属性 — 模型名称
3. 在 `experiments/run_model.py` 的 `MODEL_REGISTRY` 和 `build_model()` 中注册
4. 运行：`python experiments/run_model.py --model <name> --target NYC`

## 极端天气类型

| 类型 | 标识 | 判断条件 |
| :--- | :--- | :--- |
| 暴雨 | `heavy_rain` | 降水 > 2 in/h 或 wx_phrase 含 "Heavy Rain" |
| 暴雪 | `heavy_snow` | 降雪 > 1 in/h 或 wx_phrase 含 "Heavy Snow" |
| 风暴 | `storm` | 阵风 > 50 mph 或 wx_phrase 含 "T-Storm"/"Tornado" 等 |
| 极端高温 | `extreme_heat` | 温度 > 100°F |
| 极端低温 | `extreme_cold` | 温度 < 0°F |
| 低能见度 | `low_visibility` | 能见度 < 0.5 mi 或 wx_phrase 含 "Fog"/"Haze" 等 |
| 极端紫外线 | `extreme_uv` | UV 指数 > 10 |

更多细节参见 [data_preprocess/README.md](data_preprocess/README.md)。

## 环境依赖

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
tqdm>=4.60.0
```

## 当前进展

- [x] 数据预处理流水线（客流提取 + 节假日掩码 + 天气掩码）
- [x] NYC、CHI 两城市数据预处理完成（2014-10-01 ~ 2017-09-30, 60min）
- [x] 统一评测框架（滑动窗口 + 跨城市迁移 + 异常子集 + 五项指标）
- [x] HA baseline 评测完成
- [ ] 模型设计与实现
- [ ] 跨城市迁移实验
