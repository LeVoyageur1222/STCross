# Data
## traffic_flow

## weather

- `data/raw_data/weather/*.parquet`: 在 BAY, CHI, DC, LA, MEL, NYC 六个城市的天气预报数据；
  - 时间范围：
    - BAY: 2017.01.01 - 2018.01.01
    - CHI: 2013.01.01 - 2023.01.01
    - DC: 2013.01.01 - 2018.01.01
    - LA: 2012.01.01 - 2013.01.01
    - MEL: 2021.01.01 - 2023.01.01
    - NYC: 2013.01.01 - 2024.01.01
  - 数据表形式与字段：parquet表格数据

## holiday

- `data_preprocess/build_holiday_mask.py`: 基于规则生成。输入城市名称、开始时间、结束时间、时间片长度；