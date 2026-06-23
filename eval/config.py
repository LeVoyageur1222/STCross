from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EvalConfig:
    flow_dir: str = "data/processed_data/traffic_flow"
    mask_dir_holiday: str = "data/processed_data/holiday"
    mask_dir_weather: str = "data/processed_data/weather"

    source_city: str = "NYC"
    target_city: str = "NYC"

    history_len: int = 12
    predict_len: int = 12

    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)

    # Minimum fraction of anomalous steps in prediction window to label sample as anomaly.
    # 0.0 = ANY step (broad), 0.5 = majority (default), 1.0 = ALL steps (strict).
    anomaly_min_ratio: float = 0.5

    start_date: str = "2014-10-01"
    end_date: str = "2017-09-30"
    freq_min: int = 60

    eval_subsets: List[str] = field(default_factory=lambda: ["full", "anomaly", "normal"])

    def flow_path(self, city: str) -> str:
        return f"{self.flow_dir}/flow_{city}_{self.start_date}_{self.end_date}_{self.freq_min}min.npy"

    def holiday_mask_path(self, city: str) -> str:
        return f"{self.mask_dir_holiday}/mask_{city}_holiday_{self.start_date}_{self.end_date}_{self.freq_min}min.npy"

    def weather_mask_path(self, city: str) -> str:
        return f"{self.mask_dir_weather}/mask_{city}_all_{self.start_date}_{self.end_date}_{self.freq_min}min.npy"
