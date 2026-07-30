"""Model registry for the ProtoDRIFT reproduction package.

Only two entries are registered:
  - "ProtoDRIFT": the proposed model.
  - "HA": the historical-average lookup. It is kept because the evaluation
    pipeline uses historical averages to build normal references; it is not
    a compared baseline here.
Baselines from the full research repository are intentionally excluded.
"""

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


@register("ProtoDRIFT")
def build_protodrift(config: EvalConfig, train_config: TrainConfig, **kwargs):
    from model.protodrift import ProtoDRIFTModel
    return ProtoDRIFTModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
        k_protos=kwargs.get('protodrift_k', 8),
        alpha=kwargs.get('protodrift_alpha', 0.3),
        beta=kwargs.get('protodrift_beta', 0.01),
        gamma=kwargs.get('protodrift_gamma', 0.1),
        use_proto=kwargs.get('protodrift_use_proto', True),
        use_city_branch=kwargs.get('protodrift_use_city_branch', True),
        use_deviation_space=kwargs.get('protodrift_use_deviation_space', True),
    )
