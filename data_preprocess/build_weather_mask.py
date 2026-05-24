import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 定义极端天气类型及其判断条件
EXTREME_WEATHER_TYPES = {
    'heavy_rain': {
        'description': '暴雨',
        'conditions': [
            {'field': 'precip_hrly', 'operator': '>', 'threshold': 2.0},  # 每小时降水量 > 2英寸
            {'field': 'wx_phrase', 'operator': 'contains', 'threshold': ['Heavy Rain', 'Heavy Rain / Windy']}
        ]
    },
    'heavy_snow': {
        'description': '暴雪',
        'conditions': [
            {'field': 'snow_hrly', 'operator': '>', 'threshold': 1.0},  # 每小时降雪量 > 1英寸
            {'field': 'wx_phrase', 'operator': 'contains', 'threshold': ['Heavy Snow', 'Heavy Snow / Windy']}
        ]
    },
    'storm': {
        'description': '风暴',
        'conditions': [
            {'field': 'gust', 'operator': '>', 'threshold': 50.0},  # 阵风 > 50英里/小时
            {'field': 'wspd', 'operator': '>', 'threshold': 40.0},  # 持续风速 > 40英里/小时
            {'field': 'wx_phrase', 'operator': 'contains',
             'threshold': ['T-Storm', 'Thunder', 'Thunder / Windy', 'Light Rain with Thunder',
                          'T-Storm / Windy', 'Heavy T-Storm', 'Heavy T-Storm / Windy',
                          'Tornado', 'Thunder in the Vicinity', 'Funnel Cloud']}
        ]
    },
    'extreme_heat': {
        'description': '极端高温',
        'conditions': [
            {'field': 'temp', 'operator': '>', 'threshold': 100.0}  # 温度 > 100°F
        ]
    },
    'extreme_cold': {
        'description': '极端低温',
        'conditions': [
            {'field': 'temp', 'operator': '<', 'threshold': 0.0}  # 温度 < 0°F
        ]
    },
    'low_visibility': {
        'description': '低能见度',
        'conditions': [
            {'field': 'vis', 'operator': '<', 'threshold': 0.5},  # 能见度 < 0.5英里
            {'field': 'wx_phrase', 'operator': 'contains',
             'threshold': ['Fog', 'Fog / Windy', 'Patches of Fog', 'Haze', 'Haze / Windy',
                          'Smoke', 'Widespread Dust / Windy', 'Blowing Snow', 'Blowing Snow / Windy']}
        ]
    },
    'extreme_uv': {
        'description': '极端紫外线',
        'conditions': [
            {'field': 'uv_index', 'operator': '>', 'threshold': 10.0}  # UV指数 > 10
        ]
    }
}


def read_historical_weather(city):
    """
    读取指定城市的历史天气数据

    Args:
        city (str): 城市标识，如 'NYC', 'CHI', 'LA', 'DC', 'BAY'

    Returns:
        pd.DataFrame: 历史天气数据，已解析时间索引
    """
    # 确定历史天气数据文件路径
    city_upper = city.upper()
    weather_files = os.listdir('data/raw_data/weather_his')
    city_files = [f for f in weather_files if city_upper in f and 'HistoricalWeather.csv' in f]

    if not city_files:
        raise FileNotFoundError(f"未找到{city}的历史天气数据文件")

    # 读取数据
    file_path = os.path.join('data/raw_data/weather_his', city_files[0])
    df = pd.read_csv(file_path)

    # 解析时间字段
    df['valid_time_gmt'] = pd.to_datetime(df['valid_time_gmt'])
    df['expire_time_gmt'] = pd.to_datetime(df['expire_time_gmt'])

    # 按valid_time_gmt排序
    df = df.sort_values('valid_time_gmt').reset_index(drop=True)

    return df


def check_extreme_weather(row, weather_types, only_wx_phrase=False):
    """
    检查一行天气数据是否符合任一极端天气类型的条件

    Args:
        row (pd.Series): 一行天气数据
        weather_types (list): 要检查的极端天气类型列表
        only_wx_phrase (bool): 是否仅基于wx_phrase字段进行判断

    Returns:
        bool: 是否为极端天气
    """
    for weather_type in weather_types:
        if weather_type not in EXTREME_WEATHER_TYPES:
            continue

        conditions = EXTREME_WEATHER_TYPES[weather_type]['conditions']

        # 如果仅基于wx_phrase，则过滤出仅包含wx_phrase的条件
        if only_wx_phrase:
            conditions = [cond for cond in conditions if cond['field'] == 'wx_phrase']

        for cond in conditions:
            field = cond['field']
            operator = cond['operator']
            threshold = cond['threshold']

            # 检查字段是否存在且非空
            if field not in row or pd.isna(row[field]):
                continue

            if operator == '>':
                if row[field] > threshold:
                    return True
            elif operator == '<':
                if row[field] < threshold:
                    return True
            elif operator == 'contains':
                if isinstance(row[field], str):
                    for keyword in threshold:
                        if keyword.lower() in row[field].lower():
                            return True
            elif operator == '==':
                if row[field] == threshold:
                    return True

    return False


def build_extreme_weather_mask(start_date: str,
                               end_date: str,
                               freq_min: int,
                               city: str,
                               extreme_weather_types: list,
                               save_dir: str,
                               prefix: str = None,
                               only_wx_phrase: bool = False):
    """
    构建城市极端天气 mask（时间粒度 freq_min），并保存为 .npy 文件。

    Args:
        start_date (str): 起始日期 'YYYY-MM-DD'，含首。
        end_date   (str): 结束日期 'YYYY-MM-DD'，不含尾（即 [start_date, end_date)）。
        freq_min   (int): 时间片间隔（分钟），如 5。
        city       (str): 城市标识，如 'NYC', 'CHI', 'LA', 'DC', 'BAY'。
        extreme_weather_types (list): 要统计的极端天气类型列表，如 ['heavy_rain', 'storm']。
        save_dir   (str): 保存目录。
        prefix     (str, optional): 文件名前缀，默认为 'mask_{city}'。
        only_wx_phrase (bool, optional): 是否仅基于wx_phrase字段进行判断，默认为False。

    Returns:
        extreme_weather_mask (np.ndarray): [seq_num,]，1 表示极端天气，0 表示正常天气。
    """
    # 读取历史天气数据
    weather_df = read_historical_weather(city)

    # 转换日期格式
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成完整的时间序列
    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    if total_minutes % freq_min != 0:
        raise ValueError("时间跨度不能被 freq_min 整除，请检查输入参数。")
    seq_num = total_minutes // freq_min

    times = [start_dt + timedelta(minutes=freq_min * i) for i in range(seq_num)]

    # 创建极端天气mask
    extreme_weather_mask = np.zeros(seq_num, dtype=np.int8)

    # 对每个时间片检查是否存在极端天气
    for i, time_slice_start in tqdm(enumerate(times)):
        # 确定当前时间片的结束时间
        time_slice_end = time_slice_start + timedelta(minutes=freq_min)

        # 获取与当前时间片有重叠的所有天气数据
        # 条件：记录的valid_time_gmt < 当前时间片结束时间 且 记录的expire_time_gmt > 当前时间片开始时间
        mask = (weather_df['valid_time_gmt'] < time_slice_end) & (weather_df['expire_time_gmt'] > time_slice_start)
        overlapping_data = weather_df[mask]

        if not overlapping_data.empty:
            # 检查该时间片内是否有任何一条记录符合极端天气条件
            for _, row in overlapping_data.iterrows():
                if check_extreme_weather(row, extreme_weather_types, only_wx_phrase):
                    extreme_weather_mask[i] = 1
                    break

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    city_tag = city.upper()
    if prefix is None:
        prefix = f"mask_{city_tag}"

    # 生成天气类型标识字符串
    if len(extreme_weather_types) > 1:
        weather_types_str = "all"
    else:
        weather_types_str = extreme_weather_types[0]
    # weather_types_str = "_".join(extreme_weather_types)

    # 根据only_wx_phrase参数添加标识
    # wx_phrase_flag = "_only_wxphrase" if only_wx_phrase else ""

    mask_path = os.path.join(
        save_dir,
        f"{prefix}_{weather_types_str}_{start_date}_{end_date}_{freq_min}min.npy"
    )

    np.save(mask_path, extreme_weather_mask)
    print(f"Extreme weather mask saved to: {mask_path}, shape = {extreme_weather_mask.shape}")

    return extreme_weather_mask


if __name__ == '__main__':
    extreme_weather_types = ['heavy_rain', 'heavy_snow', 'storm', 'low_visibility']
    save_dir = 'data/processed_data/weather'

    # start_date = '2016-09-30'
    # end_date = '2017-09-30'
    # freq_min = 5
    # mask_only_wxphrase = build_extreme_weather_mask(
    #     start_date=start_date,
    #     end_date=end_date,
    #     freq_min=freq_min,
    #     city='NYC',
    #     extreme_weather_types=extreme_weather_types,
    #     save_dir=save_dir,
    #     only_wx_phrase=True
    # )
    # print(f"仅基于wx_phrase的极端天气时间片占比: {np.mean(mask_only_wxphrase):.2%}")
    #
    # mask_only_wxphrase = build_extreme_weather_mask(
    #     start_date=start_date,
    #     end_date=end_date,
    #     freq_min=freq_min,
    #     city='CHI',
    #     extreme_weather_types=extreme_weather_types,
    #     save_dir=save_dir,
    #     only_wx_phrase=True
    # )
    # print(f"仅基于wx_phrase的极端天气时间片占比: {np.mean(mask_only_wxphrase):.2%}")

    start_date = '2014-10-01'
    end_date = '2017-09-30'
    freq_min = 60
    mask_only_wxphrase = build_extreme_weather_mask(
        start_date=start_date,
        end_date=end_date,
        freq_min=freq_min,
        city='NYC',
        extreme_weather_types=extreme_weather_types,
        save_dir=save_dir,
        only_wx_phrase=True
    )
    print(f"仅基于wx_phrase的极端天气时间片占比: {np.mean(mask_only_wxphrase):.2%}")

    mask_only_wxphrase = build_extreme_weather_mask(
        start_date=start_date,
        end_date=end_date,
        freq_min=freq_min,
        city='CHI',
        extreme_weather_types=extreme_weather_types,
        save_dir=save_dir,
        only_wx_phrase=True
    )
    print(f"仅基于wx_phrase的极端天气时间片占比: {np.mean(mask_only_wxphrase):.2%}")
