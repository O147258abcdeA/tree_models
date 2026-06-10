"""模型工厂：由 (model_family, objective_type) 返回统一接口的运行器与保存/加载函数。

统一接口（run_one_window）：
  run(train_df, valid_df, test_df, feature_cols, label_col, params,
      objective_type, eval_label_col, seed) -> dict
返回 dict 含：model, best_iteration, ic_valid, rankic_valid, icir_valid,
rankicir_valid, predictions, feature_importance。
"""
from __future__ import annotations

from typing import Callable

from src.models import cat_model, lgbm_model, xgb_model

FAMILIES = ("lightgbm", "xgboost", "catboost")
OBJECTIVE_TYPES = ("regression", "ranking")

_RUNNERS: dict[str, Callable] = {
    "lightgbm": lgbm_model.run_lgb_one_window,
    "xgboost": xgb_model.run_xgb_one_window,
    "catboost": cat_model.run_cat_one_window,
}
_SAVERS: dict[str, Callable] = {
    "lightgbm": lgbm_model.save_lgb_model,
    "xgboost": xgb_model.save_xgb_model,
    "catboost": cat_model.save_cat_model,
}
_LOADERS: dict[str, Callable] = {
    "lightgbm": lgbm_model.load_lgb_model,
    "xgboost": xgb_model.load_xgb_model,
    "catboost": cat_model.load_cat_model,
}
_PREDICTORS: dict[str, Callable] = {
    "lightgbm": lgbm_model.predict_lgb,
    "xgboost": xgb_model.predict_xgb,
    "catboost": cat_model.predict_cat,
}
_CONFIG_FILES: dict[str, str] = {
    "lightgbm": "lgbm_config.yaml",
    "xgboost": "xgb_config.yaml",
    "catboost": "cat_config.yaml",
}
MODEL_FILE_EXT: dict[str, str] = {
    "lightgbm": "model.txt",
    "xgboost": "model.json",
    "catboost": "model.cbm",
}


def _check(model_family: str, objective_type: str | None = None) -> None:
    if model_family not in FAMILIES:
        raise ValueError(f"unknown model_family: {model_family}; expect {FAMILIES}")
    if objective_type is not None and objective_type not in OBJECTIVE_TYPES:
        raise ValueError(f"unknown objective_type: {objective_type}; expect {OBJECTIVE_TYPES}")


def get_runner(model_family: str, objective_type: str) -> Callable:
    """返回单窗口运行器。"""
    _check(model_family, objective_type)
    return _RUNNERS[model_family]


def get_saver(model_family: str) -> Callable:
    _check(model_family)
    return _SAVERS[model_family]


def get_loader(model_family: str) -> Callable:
    _check(model_family)
    return _LOADERS[model_family]


def get_predictor(model_family: str) -> Callable:
    _check(model_family)
    return _PREDICTORS[model_family]


def get_config_file(model_family: str) -> str:
    """返回 family 对应的默认超参配置文件名。"""
    _check(model_family)
    return _CONFIG_FILES[model_family]


def build_model_name(model_family: str, objective_type: str, label_type: str,
                     horizon: int, model_version: str) -> str:
    """模型命名规范：{family}_{obj}_{label}_h{h}_v{ver}，如 lgbm_reg_excess_h5_v01。"""
    fam = {"lightgbm": "lgbm", "xgboost": "xgb", "catboost": "cat"}[model_family]
    obj = {"regression": "reg", "ranking": "rank", "classification": "cls"}[objective_type]
    lab = {
        "raw_return": "raw", "excess_return": "excess",
        "industry_neutral_return": "indneu", "residual_return": "residual",
        "rank_return": "rankret", "group_label": "group",
    }[label_type]
    ver = model_version if model_version.startswith("v") else f"v{model_version}"
    return f"{fam}_{obj}_{lab}_h{horizon}_{ver}"
