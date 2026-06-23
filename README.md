# STCross

基于时空交叉特征的异常客流预测框架，针对节假日、极端天气等特殊事件场景下的客流量预测。

## 项目目标

常规客流预测模型在节假日、极端天气等异常场景下表现不佳，因为这些场景的客流模式与日常显著不同。本项目旨在：

1. 构建节假日、极端天气等外部事件的结构化表征（掩码机制）
2. 将外部事件特征与时空客流数据进行交叉融合
3. 提升模型在异常场景下的客流预测精度

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
│   ├── flow_NYC_2014-10-01_2017-09-30_60min.npy    # ~82 MB
│   └── flow_CHI_2014-10-01_2017-09-30_60min.npy    # ~59 MB
├── holiday/
│   ├── mask_NYC_holiday_2014-10-01_2017-09-30_60min.npy
│   ├── mask_NYC_weekend_2014-10-01_2017-09-30_60min.npy
│   ├── mask_CHI_holiday_2014-10-01_2017-09-30_60min.npy
│   └── mask_CHI_weekend_2014-10-01_2017-09-30_60min.npy
└── weather/
    ├── mask_NYC_all_2014-10-01_2017-09-30_60min.npy
    └── mask_CHI_all_2014-10-01_2017-09-30_60min.npy
```

掩码数据均为 `np.int8` 类型，`1` 表示特殊事件，`0` 表示正常状态。时间步数约 26,280（= 1,095 天 × 24 小时）。

## 项目结构

```
STCross/
├── README.md                 # 项目说明
├── exp_log.md                # 实验日志
├── data/
│   ├── raw_data/             # 原始数据（未纳入版本管理）
│   │   ├── traffic_flow/     #   交通流量 .pkl
│   │   ├── weather_his/      #   历史天气 .csv
│   │   └── weather/          #   天气预报 .parquet
│   └── processed_data/       # 预处理后数据 .npy
│       ├── traffic_flow/
│       ├── holiday/
│       └── weather/
├── data_preprocess/          # 数据预处理模块
│   ├── build_flow_data.py    #   客流数据提取与裁剪
│   ├── build_holiday_mask.py #   节假日/周末掩码生成
│   ├── build_weather_mask.py #   极端天气掩码生成
│   └── README.md             #   预处理模块详细文档
└── utils/                    # 工具函数（待完善）
```

## 数据预处理

### 快速开始

```bash
# 1. 提取客流数据（支持时间范围裁剪）
python data_preprocess/build_flow_data.py

# 2. 生成节假日与周末掩码
python data_preprocess/build_holiday_mask.py

# 3. 生成极端天气掩码
python data_preprocess/build_weather_mask.py
```

### 极端天气类型

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
tqdm>=4.60.0
```

## 当前进展

- [x] 数据预处理流水线（客流提取 + 节假日掩码 + 天气掩码）
- [x] NYC、CHI 两城市数据预处理完成（2014-10-01 ~ 2017-09-30, 60min）
- [ ] 模型设计与实现
- [ ] 实验评估
