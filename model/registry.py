"""Shared model registry. All experiment scripts import from here."""

from eval.config import EvalConfig
from train.trainer import TrainConfig
from model.ha import HAModel

MODEL_REGISTRY = {}


def register(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


@register("HA")
def build_ha(config: EvalConfig, train_config: TrainConfig):
    return HAModel(
        target_city=config.target_city,
        flow_path=config.flow_path(config.target_city),
        start_date=config.start_date,
        freq_min=config.freq_min,
        train_steps=config.train_steps,
    )
