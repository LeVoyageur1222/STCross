import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Extreme weather types and their detection conditions
EXTREME_WEATHER_TYPES = {
    'heavy_rain': {
        'description': 'heavy rain',
        'conditions': [
            {'field': 'precip_hrly', 'operator': '>', 'threshold': 2.0},  # hourly precipitation > 2 inches
            {'field': 'wx_phrase', 'operator': 'contains', 'threshold': ['Heavy Rain', 'Heavy Rain / Windy']}
        ]
    },
    'heavy_snow': {
        'description': 'heavy snow',
        'conditions': [
            {'field': 'snow_hrly', 'operator': '>', 'threshold': 1.0},  # hourly snowfall > 1 inch
            {'field': 'wx_phrase', 'operator': 'contains', 'threshold': ['Heavy Snow', 'Heavy Snow / Windy']}
        ]
    },
    'storm': {
        'description': 'storm',
        'conditions': [
            {'field': 'gust', 'operator': '>', 'threshold': 50.0},  # gust > 50 mph
            {'field': 'wspd', 'operator': '>', 'threshold': 40.0},  # sustained wind > 40 mph
            {'field': 'wx_phrase', 'operator': 'contains',
             'threshold': ['T-Storm', 'Thunder', 'Thunder / Windy', 'Light Rain with Thunder',
                          'T-Storm / Windy', 'Heavy T-Storm', 'Heavy T-Storm / Windy',
                          'Tornado', 'Thunder in the Vicinity', 'Funnel Cloud']}
        ]
    },
    'extreme_heat': {
        'description': 'extreme heat',
        'conditions': [
            {'field': 'temp', 'operator': '>', 'threshold': 100.0}  # temperature > 100°F
        ]
    },
    'extreme_cold': {
        'description': 'extreme cold',
        'conditions': [
            {'field': 'temp', 'operator': '<', 'threshold': 0.0}  # temperature < 0°F
        ]
    },
    'low_visibility': {
        'description': 'low visibility',
        'conditions': [
            {'field': 'vis', 'operator': '<', 'threshold': 0.5},  # visibility < 0.5 mile
            {'field': 'wx_phrase', 'operator': 'contains',
             'threshold': ['Fog', 'Fog / Windy', 'Patches of Fog', 'Haze', 'Haze / Windy',
                          'Smoke', 'Widespread Dust / Windy', 'Blowing Snow', 'Blowing Snow / Windy']}
        ]
    },
    'extreme_uv': {
        'description': 'extreme UV',
        'conditions': [
            {'field': 'uv_index', 'operator': '>', 'threshold': 10.0}  # UV index > 10
        ]
    }
}


def read_historical_weather(city):
    """
    Load the historical weather records of a city.

    Args:
        city (str): city identifier, e.g. 'NYC', 'CHI', 'LA', 'DC', 'BAY'

    Returns:
        pd.DataFrame: historical weather records with parsed timestamps
    """
    # Locate the historical weather file
    city_upper = city.upper()
    weather_files = os.listdir('data/raw_data/weather_his')
    city_files = [f for f in weather_files if city_upper in f and 'HistoricalWeather.csv' in f]

    if not city_files:
        raise FileNotFoundError(f"No historical weather file found for {city}")

    # Read the data
    file_path = os.path.join('data/raw_data/weather_his', city_files[0])
    df = pd.read_csv(file_path)

    # Parse time fields
    df['valid_time_gmt'] = pd.to_datetime(df['valid_time_gmt'])
    df['expire_time_gmt'] = pd.to_datetime(df['expire_time_gmt'])

    # Sort by valid_time_gmt
    df = df.sort_values('valid_time_gmt').reset_index(drop=True)

    return df


def check_extreme_weather(row, weather_types, only_wx_phrase=False):
    """
    Check whether one weather record matches any extreme weather type.

    Args:
        row (pd.Series): one weather record
        weather_types (list): extreme weather types to check
        only_wx_phrase (bool): whether to judge only by the wx_phrase field

    Returns:
        bool: True if the record is extreme weather
    """
    for weather_type in weather_types:
        if weather_type not in EXTREME_WEATHER_TYPES:
            continue

        conditions = EXTREME_WEATHER_TYPES[weather_type]['conditions']

        # Keep only wx_phrase conditions when only_wx_phrase is set
        if only_wx_phrase:
            conditions = [cond for cond in conditions if cond['field'] == 'wx_phrase']

        for cond in conditions:
            field = cond['field']
            operator = cond['operator']
            threshold = cond['threshold']

            # Skip missing or NaN fields
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
    Build the city extreme-weather mask at granularity freq_min and save it
    as an .npy file.

    Args:
        start_date (str): start date 'YYYY-MM-DD', inclusive.
        end_date   (str): end date 'YYYY-MM-DD', exclusive (i.e. [start_date, end_date)).
        freq_min   (int): step size in minutes, e.g. 5.
        city       (str): city identifier, e.g. 'NYC', 'CHI', 'LA', 'DC', 'BAY'.
        extreme_weather_types (list): types to include, e.g. ['heavy_rain', 'storm'].
        save_dir   (str): output directory.
        prefix     (str, optional): filename prefix, default 'mask_{city}'.
        only_wx_phrase (bool, optional): judge only by the wx_phrase field, default False.

    Returns:
        extreme_weather_mask (np.ndarray): [seq_num,], 1 = extreme weather, 0 = normal.
    """
    # Load historical weather records
    weather_df = read_historical_weather(city)

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Build the full time index
    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    if total_minutes % freq_min != 0:
        raise ValueError("The time span is not divisible by freq_min; check the inputs.")
    seq_num = total_minutes // freq_min

    times = [start_dt + timedelta(minutes=freq_min * i) for i in range(seq_num)]

    # Initialize the extreme weather mask
    extreme_weather_mask = np.zeros(seq_num, dtype=np.int8)

    # Check every time step for extreme weather
    for i, time_slice_start in tqdm(enumerate(times)):
        # End of the current time step
        time_slice_end = time_slice_start + timedelta(minutes=freq_min)

        # All weather records overlapping the current step, i.e. records with
        # valid_time_gmt < step end AND expire_time_gmt > step start
        mask = (weather_df['valid_time_gmt'] < time_slice_end) & (weather_df['expire_time_gmt'] > time_slice_start)
        overlapping_data = weather_df[mask]

        if not overlapping_data.empty:
            # Flag the step if any overlapping record is extreme
            for _, row in overlapping_data.iterrows():
                if check_extreme_weather(row, extreme_weather_types, only_wx_phrase):
                    extreme_weather_mask[i] = 1
                    break

    # Save
    os.makedirs(save_dir, exist_ok=True)
    city_tag = city.upper()
    if prefix is None:
        prefix = f"mask_{city_tag}"

    # Weather-type tag for the filename
    if len(extreme_weather_types) > 1:
        weather_types_str = "all"
    else:
        weather_types_str = extreme_weather_types[0]
    # weather_types_str = "_".join(extreme_weather_types)

    # Optional flag for only_wx_phrase
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
    # print(f"Extreme-weather step ratio (wx_phrase only): {np.mean(mask_only_wxphrase):.2%}")
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
    # print(f"Extreme-weather step ratio (wx_phrase only): {np.mean(mask_only_wxphrase):.2%}")

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
    print(f"Extreme-weather step ratio (wx_phrase only): {np.mean(mask_only_wxphrase):.2%}")

    mask_only_wxphrase = build_extreme_weather_mask(
        start_date=start_date,
        end_date=end_date,
        freq_min=freq_min,
        city='CHI',
        extreme_weather_types=extreme_weather_types,
        save_dir=save_dir,
        only_wx_phrase=True
    )
    print(f"Extreme-weather step ratio (wx_phrase only): {np.mean(mask_only_wxphrase):.2%}")
