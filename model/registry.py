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


@register("LSTM")
def build_lstm(config: EvalConfig, train_config: TrainConfig):
    from model.lstm import LSTMModel
    return LSTMModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("DLinear")
def build_dlinear(config: EvalConfig, train_config: TrainConfig):
    from model.dlinear import DLinearModel
    return DLinearModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("PatchTST")
def build_patchtst(config: EvalConfig, train_config: TrainConfig):
    from model.patchtst import PatchTSTModel
    return PatchTSTModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("TimesNet")
def build_timesnet(config: EvalConfig, train_config: TrainConfig):
    from model.timesnet import TimesNetModel
    return TimesNetModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


# Group B models (STGCN, DCRNN, GWNET, STGODE) removed from registry.
# Reason: cross-city crashes because NYC(820 nodes) ≠ CHI(585 nodes).
# These models contain N-dependent adjacency buffers that cannot be
# reshaped for a different-city input. Verified 2026-06-25.
# See TODO.md Phase 2 for details.

@register("TimeAlign")
def build_timealign(config: EvalConfig, train_config: TrainConfig):
    from model.timealign import TimeAlignModel
    return TimeAlignModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("ProtoTS")
def build_protots(config: EvalConfig, train_config: TrainConfig):
    from model.protots import ProtoTSModel
    return ProtoTSModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("ModernTCN")
def build_moderntcn(config: EvalConfig, train_config: TrainConfig):
    from model.moderntcn import ModernTCNModel
    return ModernTCNModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("EVL")
def build_evl(config: EvalConfig, train_config: TrainConfig):
    from model.evl import EVLModel
    return EVLModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("EALGAP")
def build_ealgap(config: EvalConfig, train_config: TrainConfig):
    from model.ealgap import EALGAPModel
    return EALGAPModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("DAN")
def build_dan(config: EvalConfig, train_config: TrainConfig):
    from model.dan import DANModel
    return DANModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("STSSDL")
def build_stssdl(config: EvalConfig, train_config: TrainConfig):
    from model.stssdl import STSSDLModel
    return STSSDLModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("STCross")
def build_stcross(config: EvalConfig, train_config: TrainConfig, **kwargs):
    from model.stcross import STCrossModel
    return STCrossModel(
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
        k_protos=kwargs.get('stcross_k', 8),
        alpha=kwargs.get('stcross_alpha', 0.01),
        beta=kwargs.get('stcross_beta', 0.01),
        gamma=kwargs.get('stcross_gamma', 0.1),
        use_proto=kwargs.get('stcross_use_proto', True),
        use_city_branch=kwargs.get('stcross_use_city_branch', True),
        use_deviation_space=kwargs.get('stcross_use_deviation_space', True),
    )
