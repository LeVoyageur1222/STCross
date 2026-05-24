# Data Preprocess 模块说明

## 一、数据处理需求

本模块负责将原始交通数据、节假日数据和天气数据转换为模型训练所需的标准化格式。主要处理任务包括：

### 1.1 核心需求

| 需求编号 | 需求描述 | 对应脚本 |
| :--- | :--- | :--- |
| REQ-001 | 从原始pkl文件中提取城市交通流量数据，转换为numpy数组格式 | `build_flow_data.py` |
| REQ-002 | 根据日期范围生成节假日掩码（包含联邦假日、城市特定假日及调休日） | `build_holiday_mask.py` |
| REQ-003 | 根据历史天气数据生成极端天气事件掩码（暴雨、暴雪、风暴等） | `build_weather_mask.py` |

### 1.2 数据格式要求

- **时间粒度**：支持多种时间间隔（如5分钟、60分钟）
- **输出格式**：`.npy` 格式，便于高效加载和处理
- **掩码格式**：`np.int8` 类型，1表示特殊事件，0表示正常状态

---

## 二、目录结构

```
data_preprocess/
├── __init__.py           # 模块初始化文件
├── build_flow_data.py    # 交通流量数据处理
├── build_holiday_mask.py # 节假日掩码生成
├── build_weather_mask.py # 极端天气掩码生成
└── README.md             # 本说明文档

../data/                  # 数据目录（同级）
├── raw_data/             # 原始数据
│   ├── traffic_flow/     # 交通流量原始数据 (.pkl)
│   ├── holiday/          # 节假日原始数据 (.csv)
│   ├── weather/          # 天气预报数据 (.parquet)
│   └── weather_his/      # 历史天气数据 (.csv)
└── processed_data/       # 处理后的数据
    ├── traffic_flow/     # 处理后的流量数据 (.npy)
    ├── holiday/          # 节假日掩码 (.npy)
    └── weather/          # 天气掩码 (.npy)
```

### 2.1 原始数据说明

| 数据类型 | 文件格式 | 说明 |
| :--- | :--- | :--- |
| 交通流量 | `.pkl` | 包含TimeRange、TimeFitness、Node等字段的字典结构 |
| 节假日 | `.csv` | 各城市节假日日期列表 |
| 历史天气 | `.csv` | 包含valid_time_gmt、wx_phrase、precip_hrly等字段 |
| 天气预报 | `.parquet` | 未来天气预测数据 |

### 2.2 处理后数据说明

| 数据类型 | 文件命名规则 | 示例 |
| :--- | :--- | :--- |
| 流量数据 | `flow_{city}_{start}_{end}_{freq}min.npy` | `flow_NYC_2013-07-01_2017-09-30_60min.npy` |
| 节假日掩码 | `mask_{city}_holiday_{start}_{end}_{freq}min.npy` | `mask_NYC_holiday_2013-07-01_2017-09-30_60min.npy` |
| 周末掩码 | `mask_{city}_weekend_{start}_{end}_{freq}min.npy` | `mask_NYC_weekend_2013-07-01_2017-09-30_60min.npy` |
| 天气掩码 | `mask_{city}_{type}_{start}_{end}_{freq}min.npy` | `mask_NYC_all_2013-07-01_2017-09-30_60min.npy` |

---

## 三、具体设计

### 3.1 build_flow_data.py

**功能**：从pkl文件提取交通流量数据，支持按指定时间范围裁剪

**核心函数**：

| 函数名 | 功能说明 | 参数 | 返回值 |
| :--- | :--- | :--- | :--- |
| `read_bike_pklfile` | 读取并打印pkl文件结构（调试用） | `raw_path`: pkl文件路径 | 无 |
| `get_bike_flow` | 提取流量数据并保存为npy，可选按时间范围裁剪 | `raw_root_path`, `city`, `save_root_path`, `start_date`, `end_date` | 无 |

**数据提取逻辑**：
1. 从pkl文件读取字典数据
2. 提取 `data['Node']['TrafficNode']` 作为流量矩阵（shape: [time_step, num_node]）
3. 获取时间范围 `data['TimeRange']` 和时间步长 `data['TimeFitness']`
4. 如果指定了 `start_date` 和 `end_date`，裁剪流量数据到指定时间范围
5. 保存为 `flow_{city}_{start}_{end}_{freq}min.npy`

**时间范围裁剪说明**：
- `start_date` 和 `end_date` 参数用于指定输出数据的时间范围
- 如果不指定，默认使用原始数据的完整时间范围
- 指定时间范围后，数据会被裁剪以与 `build_holiday_mask.py` 和 `build_weather_mask.py` 生成的掩码对齐

### 3.2 build_holiday_mask.py

**功能**：生成节假日和周末掩码

**核心函数**：

| 函数名 | 功能说明 | 参数 | 返回值 |
| :--- | :--- | :--- | :--- |
| `nth_weekday_of_month` | 获取某月第n个指定星期几 | `year`, `month`, `weekday`, `n` | `date`对象 |
| `last_weekday_of_month` | 获取某月最后一个指定星期几 | `year`, `month`, `weekday` | `date`对象 |
| `add_actual_and_observed` | 处理调休日期（周末节假日调休到工作日） | `d`: date对象 | 日期集合 |
| `get_us_federal_holidays` | 获取美国联邦节假日 | `year` | 日期集合 |
| `get_city_extra_holidays` | 获取城市/州级额外假日 | `year`, `city` | 日期集合 |
| `build_city_holiday_and_weekend_masks` | 构建节假日和周末掩码 | `start_date`, `end_date`, `freq_min`, `city`, `save_dir`, `prefix` | `holiday_mask`, `weekend_mask` |

**节假日覆盖范围**：

| 联邦假日 | 日期规则 |
| :--- | :--- |
| New Year's Day | Jan 1 |
| Martin Luther King Jr. Day | 1月第3个周一 |
| Presidents' Day | 2月第3个周一 |
| Memorial Day | 5月最后一个周一 |
| Independence Day | Jul 4 |
| Labor Day | 9月第1个周一 |
| Columbus Day | 10月第2个周一 |
| Veterans Day | Nov 11 |
| Thanksgiving | 11月第4个周四 |
| Christmas | Dec 25 |

| 城市额外假日 | 城市 | 日期 |
| :--- | :--- | :--- |
| Lincoln's Birthday | NYC, CHI | Feb 12 |
| Cesar Chavez Day | LA, BAY | Mar 31 |

### 3.3 build_weather_mask.py

**功能**：生成极端天气事件掩码

**极端天气类型定义**：

| 天气类型 | 标识 | 判断条件 |
| :--- | :--- | :--- |
| 暴雨 | `heavy_rain` | 每小时降水>2英寸 或 wx_phrase包含"Heavy Rain" |
| 暴雪 | `heavy_snow` | 每小时降雪>1英寸 或 wx_phrase包含"Heavy Snow" |
| 风暴 | `storm` | 阵风>50mph 或 风速>40mph 或 wx_phrase包含"T-Storm"/"Thunder"/"Tornado" |
| 极端高温 | `extreme_heat` | 温度>100°F |
| 极端低温 | `extreme_cold` | 温度<0°F |
| 低能见度 | `low_visibility` | 能见度<0.5英里 或 wx_phrase包含"Fog"/"Haze"/"Smoke" |
| 极端紫外线 | `extreme_uv` | UV指数>10 |

**核心函数**：

| 函数名 | 功能说明 | 参数 | 返回值 |
| :--- | :--- | :--- | :--- |
| `read_historical_weather` | 读取城市历史天气数据 | `city` | `pd.DataFrame` |
| `check_extreme_weather` | 检查单条记录是否为极端天气 | `row`, `weather_types`, `only_wx_phrase` | `bool` |
| `build_extreme_weather_mask` | 构建极端天气掩码 | `start_date`, `end_date`, `freq_min`, `city`, `extreme_weather_types`, `save_dir`, `prefix`, `only_wx_phrase` | `extreme_weather_mask` |

**时间片匹配逻辑**：
- 对于每个时间片，检查所有与该时间片重叠的天气记录
- 只要有一条记录满足极端天气条件，该时间片标记为1

---

## 四、运行说明

### 4.1 环境依赖

```bash
# 依赖包
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.60.0
```

### 4.2 运行方式

**方式一：直接运行各脚本**

```bash
# 处理流量数据
python build_flow_data.py

# 处理节假日掩码
python build_holiday_mask.py

# 处理天气掩码
python build_weather_mask.py
```

**方式二：作为模块导入使用**

```python
from data_preprocess import build_flow_data, build_holiday_mask, build_weather_mask

# 处理流量数据（指定时间范围，与掩码对齐）
build_flow_data.get_bike_flow(
    raw_root_path="data/raw_data/traffic_flow/",
    city="NYC",
    save_root_path="data/processed_data/traffic_flow/",
    start_date="2014-10-01",
    end_date="2017-09-30"
)

# 构建节假日掩码
holiday_mask, weekend_mask = build_holiday_mask.build_city_holiday_and_weekend_masks(
    start_date="2014-10-01",
    end_date="2017-09-30",
    freq_min=60,
    city="NYC",
    save_dir="data/processed_data/holiday"
)

# 构建天气掩码
weather_mask = build_weather_mask.build_extreme_weather_mask(
    start_date="2014-10-01",
    end_date="2017-09-30",
    freq_min=60,
    city="NYC",
    extreme_weather_types=['heavy_rain', 'heavy_snow', 'storm', 'low_visibility'],
    save_dir="data/processed_data/weather",
    only_wx_phrase=True
)
```

### 4.3 参数说明

#### get_bike_flow

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `raw_root_path` | str | `data/raw_data/traffic_flow/` | 原始数据根目录 |
| `city` | str | `NYC` | 城市标识：'NYC', 'CHI' |
| `save_root_path` | str | `data/processed_data/traffic_flow/` | 保存目录 |
| `start_date` | str | None | 开始日期，格式'YYYY-MM-DD'，None表示使用数据最早日期 |
| `end_date` | str | None | 结束日期，格式'YYYY-MM-DD'（不含尾），None表示使用数据最晚日期 |

#### build_city_holiday_and_weekend_masks

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `start_date` | str | - | 起始日期，格式'YYYY-MM-DD'（含首） |
| `end_date` | str | - | 结束日期，格式'YYYY-MM-DD'（不含尾） |
| `freq_min` | int | - | 时间片间隔（分钟） |
| `city` | str | - | 城市标识：'NYC', 'CHI', 'LA', 'DC', 'BAY' |
| `save_dir` | str | - | 保存目录 |
| `prefix` | str | None | 文件名前缀，默认'mask_{city}' |

#### build_extreme_weather_mask

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `start_date` | str | - | 起始日期，格式'YYYY-MM-DD' |
| `end_date` | str | - | 结束日期，格式'YYYY-MM-DD' |
| `freq_min` | int | - | 时间片间隔（分钟） |
| `city` | str | - | 城市标识 |
| `extreme_weather_types` | list | - | 要检测的极端天气类型列表 |
| `save_dir` | str | - | 保存目录 |
| `prefix` | str | None | 文件名前缀 |
| `only_wx_phrase` | bool | False | 是否仅基于wx_phrase字段判断 |

### 4.4 注意事项

1. **时间跨度要求**：`end_date` 必须晚于 `start_date`，且总分钟数必须能被 `freq_min` 整除
2. **城市标识**：大小写不敏感，支持 'NYC', 'CHI', 'LA', 'BAY', 'DC'
3. **数据文件**：确保 `data/raw_data/` 目录下存在对应城市的数据文件
4. **输出目录**：脚本会自动创建不存在的输出目录

---

## 五、输出示例

### 5.1 流量数据输出

```
Flow saved to: data/processed_data/traffic_flow/flow_NYC_2013-07-01_2017-09-30_60min.npy, shape = (37728, 128)
```

### 5.2 节假日掩码输出

```
Holiday mask saved to: data/processed_data/holiday/mask_NYC_holiday_2013-07-01_2017-09-30_60min.npy, shape = (37728,)
Weekend mask saved to: data/processed_data/holiday/mask_NYC_weekend_2013-07-01_2017-09-30_60min.npy, shape = (37728,)
```

### 5.3 天气掩码输出

```
100%|██████████| 37728/37728 [00:45<00:00, 830.57it/s]
Extreme weather mask saved to: data/processed_data/weather/mask_NYC_all_2013-07-01_2017-09-30_60min.npy, shape = (37728,)
仅基于wx_phrase的极端天气时间片占比: 3.25%
```