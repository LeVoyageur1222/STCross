import os
import numpy as np
from datetime import datetime, timedelta, date

def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    Return the n-th given weekday of a month.
    weekday: Monday=0, Sunday=6
    e.g. third Monday => weekday=0, n=3
    """
    d = date(year, month, 1)
    # First target weekday of the month
    days_ahead = (weekday - d.weekday() + 7) % 7
    d = d + timedelta(days=days_ahead)
    # Advance (n-1) weeks
    d = d + timedelta(weeks=n - 1)
    return d

def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """
    Return the last given weekday of a month.
    weekday: Monday=0, Sunday=6
    e.g. last Monday of May => weekday=0, month=5
    """
    # First day of next month minus one day = last day of this month
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    # Walk backwards to the target weekday
    days_back = (d.weekday() - weekday + 7) % 7
    d = d - timedelta(days=days_back)
    return d

def add_actual_and_observed(d: date) -> set:
    """
    For fixed-date holidays: return the actual date plus the observed date
    when it falls on a weekend.
    - Saturday: observed on the preceding Friday
    - Sunday: observed on the following Monday
    For mask building we also keep the actual date as a holiday, since flow
    patterns usually differ on it as well.
    """
    res = {d}
    if d.weekday() == 5:      # Saturday
        res.add(d - timedelta(days=1))
    elif d.weekday() == 6:    # Sunday
        res.add(d + timedelta(days=1))
    return res

def get_election_day(year: int) -> date:
    """
    Return the U.S. Election Day of a year:
    conventionally the Tuesday following the first Monday of November.
    """
    first_nov = date(year, 11, 1)
    # First Monday of November
    days_ahead = (0 - first_nov.weekday() + 7) % 7  # Monday = 0
    first_monday = first_nov + timedelta(days=days_ahead)
    # Election Day = the Tuesday after the first Monday
    election_day = first_monday + timedelta(days=1)
    return election_day

def get_us_federal_holidays(year: int) -> set:
    """
    Return the U.S. holidays of a year (set of dates), including actual dates
    and weekend-observed dates. Covers:
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
    Additionally:
    - Election Day                (Tuesday following the first Monday of Nov)
      (strictly not a federal holiday, but it visibly changes travel patterns,
      so it is treated as a holiday here)
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
    City/state-level extra holidays (not necessarily federal).
    city: 'NYC', 'CHI', 'LA', 'BAY', 'DC', etc. (case-insensitive).
    """
    city = city.upper()
    holidays = set()

    # NYC / New York
    if city in {"NYC", "NEW_YORK", "NEW YORK"}:
        # Lincoln's Birthday - Feb 12 (New York State holiday)
        holidays |= add_actual_and_observed(date(year, 2, 12))

    # CHI / Chicago (Illinois)
    if city in {"CHI", "CHICAGO"}:
        # Lincoln's Birthday - Feb 12 (Illinois holiday)
        holidays |= add_actual_and_observed(date(year, 2, 12))

    # LA / Los Angeles & BAY (Bay Area / SF, California)
    if city in {"LA", "LOS_ANGELES", "LOS ANGELES",
                "BAY", "BAYAREA",
                "SF", "SAN_FRANCISCO", "SAN FRANCISCO"}:
        # Cesar Chavez Day - Mar 31 (California holiday)
        holidays |= add_actual_and_observed(date(year, 3, 31))

    # DC: federal holidays only, nothing extra
    if city in {"DC", "WASHINGTON_DC", "WASHINGTON DC"}:
        pass

    return holidays

def get_city_holidays_for_year_range(city: str, start_year: int, end_year: int) -> set:
    """
    For every year in [start_year, end_year], collect:
    - nationwide holidays (federal + Election Day, etc.)
    - city-specific extra holidays
    and merge them into one set of dates.
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
    Build the city holiday mask and weekend mask at granularity freq_min and
    save them as .npy files.

    Args:
        start_date (str): start date 'YYYY-MM-DD', inclusive.
        end_date   (str): end date 'YYYY-MM-DD', exclusive (i.e. [start_date, end_date)).
        freq_min   (int): step size in minutes, e.g. 5.
        city       (str): city identifier, e.g. 'NYC', 'CHI', 'LA', 'DC', 'BAY'.
        save_dir   (str): output directory.
        prefix     (str, optional): filename prefix, default 'mask_{city}'.

    Returns:
        holiday_mask (np.ndarray): [seq_num,], 1 = holiday (incl. observed dates), 0 = non-holiday.
        weekend_mask (np.ndarray): [seq_num,], 1 = Saturday/Sunday, 0 = weekday.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")

    if end_dt <= start_dt:
        raise ValueError("end_date must be later than start_date")

    # Year range (e.g. 2017-01-01 ~ 2019-06-01 -> [2017, 2018, 2019])
    start_year = start_dt.year
    end_year   = end_dt.year

    # Holiday set of this city over the year range
    holidays = get_city_holidays_for_year_range(city, start_year, end_year)

    # Number of time steps
    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    if total_minutes % freq_min != 0:
        raise ValueError("The time span is not divisible by freq_min; check the inputs.")
    seq_num = total_minutes // freq_min

    # One datetime per time step
    times = [start_dt + timedelta(minutes=freq_min * i) for i in range(seq_num)]

    # Holiday mask: 1 whenever t.date() is in the holiday set
    holiday_mask = np.array(
        [1 if t.date() in holidays else 0 for t in times],
        dtype=np.int8
    )

    # Weekend mask: Saturday=5, Sunday=6
    weekend_mask = np.array(
        [1 if t.weekday() >= 5 else 0 for t in times],
        dtype=np.int8
    )

    # Save
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
