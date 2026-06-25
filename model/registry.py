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


@register("STGCN")
def build_stgcn(config: EvalConfig, train_config: TrainConfig):
    import numpy as np
    from model.stgcn import STGCNModel
    adj = np.load(config.adj_path(config.target_city))
    N = adj.shape[0]
    return STGCNModel(
        N=N, adj=adj,
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("DCRNN")
def build_dcrnn(config: EvalConfig, train_config: TrainConfig):
    import numpy as np
    from model.dcrnn import DCRNNModel
    adj = np.load(config.adj_path(config.target_city))
    N = adj.shape[0]
    return DCRNNModel(
        N=N, adj=adj,
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )


@register("GWNET")
def build_gwnet(config: EvalConfig, train_config: TrainConfig):
    import numpy as np
    from model.gwnet import GWNETModel
    adj = np.load(config.adj_path(config.target_city))
    N = adj.shape[0]
    return GWNETModel(
        N=N, adj=adj,
        H=config.history_len, P=config.predict_len,
        train_config=train_config,
    )
