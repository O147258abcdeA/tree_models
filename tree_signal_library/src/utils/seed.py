"""统一随机种子：python / numpy / LightGBM / XGBoost / CatBoost。"""
from __future__ import annotations

import os
import random

import numpy as np

GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """设置全局随机种子。树模型库的种子通过参数注入（见 seed_params）。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_params(model_family: str, seed: int = GLOBAL_SEED) -> dict:
    """返回各模型库的种子参数字典，训练时 merge 进 params。"""
    if model_family == "lightgbm":
        return {"seed": seed, "deterministic": True, "force_row_wise": True}
    if model_family == "xgboost":
        return {"seed": seed}
    if model_family == "catboost":
        return {"random_seed": seed}
    raise ValueError(f"unknown model_family: {model_family}")
