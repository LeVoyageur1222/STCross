import numpy as np
import pickle
from datetime import datetime, timedelta

def read_bike_pklfile(raw_path):
    f = open(raw_path, 'rb')
    raw = pickle.load(f)
    print(f"raw keys: {raw.keys()}")

    print(f"data['TimeRange'] = {raw['TimeRange']}")
    print(f"data['TimeFitness'] = {raw['TimeFitness']}")
    # print(f"data['LenTimeSlots'] = {raw['LenTimeSlots']}")

    print(
        f"data['Node'] = {raw['Node'].keys()}")  # dict_keys(['TrafficNode', 'StationInfo', 'TrafficMonthlyInteraction'])
    print(f"data['Node']['TrafficNode'] = {type(raw['Node']['TrafficNode'])}")
    print(f"data['Node']['TrafficNode'] = {raw['Node']['TrafficNode'].shape}")
    print(f"data['Node']['StationInfo'] = {type(raw['Node']['StationInfo'])}")
    print(f"data['Node']['StationInfo'] = {len(raw['Node']['StationInfo'])}")
    print(f"data['Node']['StationInfo'][:5] = {raw['Node']['StationInfo'][:5]}")
    print(f"data['Node']['TrafficMonthlyInteraction'] = {type(raw['Node']['TrafficMonthlyInteraction'])}")
    print(f"data['Node']['TrafficMonthlyInteraction'] = {raw['Node']['TrafficMonthlyInteraction'].shape}")

    print(f"data['Grid'] = {raw['Grid'].keys()}")  # dict_keys(['TrafficGrid', 'GridLatLng', 'POI'])
    print(f"data['Grid']['TrafficGrid'] = {type(raw['Grid']['TrafficGrid'])}")
    print(f"data['Grid']['TrafficGrid'] = {len(raw['Grid']['TrafficGrid'])}")
    print(f"data['Grid']['GridLatLng'] = {type(raw['Grid']['GridLatLng'])}")
    print(f"data['Grid']['GridLatLng'] = {len(raw['Grid']['GridLatLng'])}")
    print(f"data['Grid']['POI'] = {type(raw['Grid']['POI'])}")
    print(f"data['Grid']['POI'] = {len(raw['Grid']['POI'])}")   # 0

    print(f"data['ExternalFeature'] = {raw['ExternalFeature'].keys()}")
    print(f"data['ExternalFeature']['Weather'] = {type(raw['ExternalFeature']['Weather'])}")
    print(f"data['ExternalFeature']['CheckInFeature'] = {len(raw['ExternalFeature']['CheckInFeature'])}")   # list
    print(f"data['ExternalFeature']['CheckInFeature'][0] = {len(raw['ExternalFeature']['CheckInFeature'][0])}")    # list
    print(f"data['ExternalFeature']['CheckInFeature'][0][0] = {raw['ExternalFeature']['CheckInFeature'][0][0].shape}")
    print(f"data['ExternalFeature']['WeatherStateDim'] = {raw['ExternalFeature']['WeatherStateDim']}")

def get_bike_flow(raw_root_path="data/raw_data/traffic_flow/", city="NYC", save_root_path="data/processed_data/traffic_flow/",
                  start_date=None, end_date=None):
    """
    Extract traffic flow data from the raw pkl file, optionally crop it to a
    given time range, and save it as an .npy file.

    Args:
        raw_root_path (str): raw data root, default 'data/raw_data/traffic_flow/'
        city (str): city identifier such as 'NYC' or 'CHI', default 'NYC'
        save_root_path (str): output directory, default 'data/processed_data/traffic_flow/'
        start_date (str, optional): start date 'YYYY-MM-DD'; None = earliest date in the data
        end_date (str, optional): end date 'YYYY-MM-DD' (exclusive); None = latest date in the data

    Returns:
        None
    """
    if city == "CHI":
        bike_path = raw_root_path + f"Bike_Chicago.pkl"
    else:
        bike_path = raw_root_path + f"Bike_{city}.pkl"
    f = open(bike_path, 'rb')
    data = pickle.load(f)

    # data['TimeRange'] = ['2013-07-01', '2017-09-30'] gives the time span
    data_start_date = data['TimeRange'][0]
    data_end_date = data['TimeRange'][-1]
    # data['TimeFitness'] = 60 gives the step size in minutes
    freq_min = data['TimeFitness']
    # data['Node']['TrafficNode'] holds the flow data, shape [time_step, num_node]
    flow = data['Node']['TrafficNode']

    # Crop when the caller specified a time range
    if start_date is not None or end_date is not None:
        # Fall back to the range recorded in the data
        actual_start = start_date if start_date is not None else data_start_date
        actual_end = end_date if end_date is not None else data_end_date

        # Parse dates
        actual_start_dt = datetime.strptime(actual_start, "%Y-%m-%d")
        actual_end_dt = datetime.strptime(actual_end, "%Y-%m-%d")
        data_start_dt = datetime.strptime(data_start_date, "%Y-%m-%d")

        # Offsets in minutes
        total_minutes_start = int((actual_start_dt - data_start_dt).total_seconds() // 60)
        total_minutes_end = int((actual_end_dt - data_start_dt).total_seconds() // 60)

        # Slice indices
        start_idx = total_minutes_start // freq_min
        end_idx = total_minutes_end // freq_min

        # Crop the flow array
        flow = flow[start_idx:end_idx]

        print(f"Time range cropped from [{data_start_date}, {data_end_date}] to [{actual_start}, {actual_end}]")
        print(f"Flow data cropped from shape {data['Node']['TrafficNode'].shape} to {flow.shape}")
    else:
        actual_start = data_start_date
        actual_end = data_end_date

    save_path = save_root_path + f"flow_{city}_{actual_start}_{actual_end}_{freq_min}min.npy"
    np.save(save_path, flow)
    print(f"Flow saved to: {save_path}, shape = {flow.shape}")

if __name__ == '__main__':
    # Inspect the raw NYC / Chicago bike pkl files (optional)
    # NYC_bike_path = "data/raw_data/traffic_flow/Bike_NYC.pkl"
    # CHI_bike_path = "data/raw_data/traffic_flow/Bike_Chicago.pkl"
    # read_bike_pklfile(NYC_bike_path)

    # Save bike flow arrays for the study period (aligned with the holiday masks)
    get_bike_flow(
        raw_root_path="data/raw_data/traffic_flow/",
        city="NYC",
        save_root_path="data/processed_data/traffic_flow/",
        start_date="2014-10-01",
        end_date="2017-09-30"
    )
    get_bike_flow(
        raw_root_path="data/raw_data/traffic_flow/",
        city="CHI",
        save_root_path="data/processed_data/traffic_flow/",
        start_date="2014-10-01",
        end_date="2017-09-30"
    )