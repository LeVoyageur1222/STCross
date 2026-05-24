import os
import numpy as np
from datetime import datetime, timedelta, date

def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    返回某年某月第 n 个指定 weekday 的日期。
    weekday: Monday=0, Sunday=6
    例如：第三个星期一 => weekday=0, n=3
    """
    d = date(year, month, 1)
    # 找到当月第一个目标 weekday
    days_ahead = (weekday - d.weekday() + 7) % 7
    d = d + timedelta(days=days_ahead)
    # 再加 (n-1) 周
    d = d + timedelta(weeks=n - 1)
    return d

def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """
    返回某年某月最后一个指定 weekday 的日期。
    weekday: Monday=0, Sunday=6
    例如：五月最后一个星期一 => weekday=0, month=5
    """
    # 下个月的第一天 - 1 天 = 当月最后一天
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    # 从最后一天往前找指定 weekday
    days_back = (d.weekday() - weekday + 7) % 7
    d = d - timedelta(days=days_back)
    return d

def add_actual_and_observed(d: date) -> set:
    """
    对于固定日期假日：返回“实际日期 + 遇到周末时的调休日期”。
    - 周六：调休到前一个周五
    - 周日：调休到下一个周一
    为了建 mask，我们既把实际日期也当作节假日（人流模式往往也不一样）。
    """
    res = {d}
    if d.weekday() == 5:      # Saturday
        res.add(d - timedelta(days=1))
    elif d.weekday() == 6:    # Sunday
        res.add(d + timedelta(days=1))
    return res

def get_election_day(year: int) -> date:
    """
    返回某年美国大选日：
    通常定义为“11 月中，第一个星期一之后的星期二”。
    """
    first_nov = date(year, 11, 1)
    # 找到 11 月的第一个星期一
    days_ahead = (0 - first_nov.weekday() + 7) % 7  # Monday = 0
    first_monday = first_nov + timedelta(days=days_ahead)
    # Election Day = 第一个星期一之后的星期二
    election_day = first_monday + timedelta(days=1)
    return election_day

def get_us_federal_holidays(year: int) -> set:
    """
    返回某年全美假日（日期集合），包含实际日和遇周末的调休日。
    主要包括：
    - New Year's Day              (Jan 1)
    - Martin Luther King Jr. Day  (third Monday of Jan)
    - Washington's Birthday /
      Presidents' Day             (third Monday of Feb)
    - Memorial Day                (last Monday of May)
    - Independence Day            (Jul 4)
    - Labor Day                   (first Monday of Sep)
    - Columbus / Indigenous Day   (second Monday of Oct)
    - Veterans Day                (Nov 11)
    - Thanksgiving Day            (fourth Thursday of Nov)
    - Christmas Day               (Dec 25)
    额外加入：
    - Election Day                (Tuesday following the first Monday of Nov)
      （严格来说不是联邦法定假日，但对出行/用车模式有明显影响，这里作为“节假日”处理）
    """
    holidays = set()

    # New Year's Day
    holidays |= add_actual_and_observed(date(year, 1, 1))

    # Martin Luther King Jr. Day - third Monday of Jan
    holidays.add(nth_weekday_of_month(year, 1, weekday=0, n=3))

    # Washington's Birthday / Presidents' Day - third Monday of Feb
    holidays.add(nth_weekday_of_month(year, 2, weekday=0, n=3))

    # Memorial Day - last Monday of May
    holidays.add(last_weekday_of_month(year, 5, weekday=0))

    # Independence Day - Jul 4
    holidays |= add_actual_and_observed(date(year, 7, 4))

    # Labor Day - first Monday of Sep
    holidays.add(nth_weekday_of_month(year, 9, weekday=0, n=1))

    # Columbus Day / Indigenous Peoples' Day - second Monday of Oct
    holidays.add(nth_weekday_of_month(year, 10, weekday=0, n=2))

    # Veterans Day - Nov 11
    holidays |= add_actual_and_observed(date(year, 11, 11))

    # Election Day - Tuesday following the first Monday of Nov
    holidays.add(get_election_day(year))

    # Thanksgiving Day - fourth Thursday of Nov
    holidays.add(nth_weekday_of_month(year, 11, weekday=3, n=4))  # Thursday=3

    # Christmas Day - Dec 25
    holidays |= add_actual_and_observed(date(year, 12, 25))

    return holidays

def get_city_extra_holidays(year: int, city: str) -> set:
    """
    城市/州级额外假日（可能不是联邦假日）。
    city: 'NYC', 'CHI', 'LA', 'BAY', 'DC' 等（大小写不敏感）。
    """
    city = city.upper()
    holidays = set()

    # NYC / New York
    if city in {"NYC", "NEW_YORK", "NEW YORK"}:
        # Lincoln's Birthday - Feb 12（纽约州假日）
        holidays |= add_actual_and_observed(date(year, 2, 12))

    # CHI / Chicago（Illinois）
    if city in {"CHI", "CHICAGO"}:
        # Lincoln's Birthday - Feb 12（伊州假日）
        holidays |= add_actual_and_observed(date(year, 2, 12))

    # LA / Los Angeles & BAY（Bay Area / SF 等，加州）
    if city in {"LA", "LOS_ANGELES", "LOS ANGELES",
                "BAY", "BAYAREA",
                "SF", "SAN_FRANCISCO", "SAN FRANCISCO"}:
        # Cesar Chavez Day - Mar 31（加州假日）
        holidays |= add_actual_and_observed(date(year, 3, 31))

    # DC：目前只使用联邦假日，就不额外添加
    if city in {"DC", "WASHINGTON_DC", "WASHINGTON DC"}:
        pass

    return holidays

def get_city_holidays_for_year_range(city: str, start_year: int, end_year: int) -> set:
    """
    对 [start_year, end_year] 每一年，生成：
    - 全美（联邦 + Election Day 等）节假日
    - 城市额外假日
    并合并成一个日期集合。
    """
    all_holidays = set()
    for y in range(start_year, end_year + 1):
        all_holidays |= get_us_federal_holidays(y)
        all_holidays |= get_city_extra_holidays(y, city)
    return all_holidays

def build_city_holiday_and_weekend_masks(start_date: str,
                                         end_date: str,
                                         freq_min: int,
                                         city: str,
                                         save_dir: str,
                                         prefix: str = None):
    """
    构建“城市节假日 mask + weekend mask”（时间粒度 freq_min），并保存为 .npy 文件。

    Args:
        start_date (str): 起始日期 'YYYY-MM-DD'，含首。
        end_date   (str): 结束日期 'YYYY-MM-DD'，不含尾（即 [start_date, end_date)）。
        freq_min   (int): 时间片间隔（分钟），如 5。
        city       (str): 城市标识，如 'NYC', 'CHI', 'LA', 'DC', 'BAY'。
        save_dir   (str): 保存目录。
        prefix     (str, optional): 文件名前缀，默认为 'mask_{city}'。

    Returns:
        holiday_mask (np.ndarray): [seq_num,]，1 表示节假日（含调休），0 表示非节假日。
        weekend_mask (np.ndarray): [seq_num,]，1 表示周六/周日，0 表示工作日。
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")

    if end_dt <= start_dt:
        raise ValueError("end_date 必须晚于 start_date")

    # 年份范围（例如 2017-01-01 ~ 2019-06-01 -> [2017, 2018, 2019]）
    start_year = start_dt.year
    end_year   = end_dt.year

    # 获取该城市在这一段年份范围内的节假日集合
    holidays = get_city_holidays_for_year_range(city, start_year, end_year)

    # 时间片数
    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    if total_minutes % freq_min != 0:
        raise ValueError("时间跨度不能被 freq_min 整除，请检查输入参数。")
    seq_num = total_minutes // freq_min

    # 每个时间片对应一个 datetime
    times = [start_dt + timedelta(minutes=freq_min * i) for i in range(seq_num)]

    # 节假日 mask：只要 t.date() 在 holidays 集合中就为 1
    holiday_mask = np.array(
        [1 if t.date() in holidays else 0 for t in times],
        dtype=np.int8
    )

    # weekend mask：周六=5，周日=6
    weekend_mask = np.array(
        [1 if t.weekday() >= 5 else 0 for t in times],
        dtype=np.int8
    )

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    city_tag = city.upper()
    if prefix is None:
        prefix = f"mask_{city_tag}"

    holiday_path = os.path.join(
        save_dir,
        f"{prefix}_holiday_{start_date}_{end_date}_{freq_min}min.npy"
    )
    weekend_path = os.path.join(
        save_dir,
        f"{prefix}_weekend_{start_date}_{end_date}_{freq_min}min.npy"
    )

    np.save(holiday_path, holiday_mask)
    np.save(weekend_path, weekend_mask)

    print(f"Holiday mask saved to: {holiday_path}, shape = {holiday_mask.shape}")
    print(f"Weekend mask saved to: {weekend_path}, shape = {weekend_mask.shape}")

    return holiday_mask, weekend_mask


if __name__ == '__main__':
    # 2016-09-30 ~ 2017-09-30
    # holiday_mask, weekend_mask = build_city_holiday_and_weekend_masks(
    #     start_date="2016-09-30",
    #     end_date="2017-09-30",
    #     freq_min=5,
    #     city="NYC",
    #     save_dir="data/processed_data/holiday"
    # )
    # holiday_mask, weekend_mask = build_city_holiday_and_weekend_masks(
    #     start_date="2016-09-30",
    #     end_date="2017-09-30",
    #     freq_min=5,
    #     city="CHI",
    #     save_dir="data/processed_data/holiday"
    # )

    # 2013-07-01 ~ 2017-09-30
    holiday_mask, weekend_mask = build_city_holiday_and_weekend_masks(
        start_date="2014-10-01",
        end_date="2017-09-30",
        freq_min=60,
        city="NYC",
        save_dir="data/processed_data/holiday"
    )
    holiday_mask, weekend_mask = build_city_holiday_and_weekend_masks(
        start_date="2014-10-01",
        end_date="2017-09-30",
        freq_min=60,
        city="CHI",
        save_dir="data/processed_data/holiday"
    )
