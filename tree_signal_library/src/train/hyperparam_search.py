"""超参搜索（可选）：只允许使用 train/valid 窗口，禁止用测试集调参。

实现一个轻量随机搜索：在给定参数空间内采样，对每组参数在单个
(train, valid) 窗口上训练并以 valid RankIC（每日横截面）为目标排序。
若安装了 optuna 可平滑替换 _sample 为 TPE 采样器，接口不变。
"""
from __future__ import annotations

import random
from typing import Any

import polars as pl

from src.models.model_factory import get_runner
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sample(space: dict[str, list[Any]], rng: random.Random) -> dict:
    """从离散参数空间各取一个值。"""
    return {k: rng.choice(v) for k, v in space.items()}


def hyperparam_search(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                      feature_cols: list[str], label_col: str,
                      base_params: dict, search_space: dict[str, list[Any]],
                      model_family: str, objective_type: str,
                      n_trials: int = 20, eval_label_col: str | None = None,
                      seed: int = 42) -> tuple[dict, pl.DataFrame]:
    """随机搜索超参，目标为 valid 每日横截面 RankIC 均值。

    注意：只使用 train/valid；test 数据严禁传入本函数。

    Returns
    -------
    (best_params, trials_df)：最优参数与全部 trial 记录。
    """
    rng = random.Random(seed)
    runner = get_runner(model_family, objective_type)
    trials: list[dict] = []
    best_params, best_score = dict(base_params), float("-inf")
    for i in range(n_trials):
        params = {**base_params, **_sample(search_space, rng)}
        result = runner(train_df, valid_df, valid_df, feature_cols, label_col,
                        params, objective_type=objective_type,
                        eval_label_col=eval_label_col, seed=seed)
        score = result["rankic_valid"] if result["rankic_valid"] is not None else float("-inf")
        trials.append({"trial": i, "rankic_valid": result["rankic_valid"],
                       "ic_valid": result["ic_valid"],
                       "best_iteration": result["best_iteration"],
                       **{f"p_{k}": v for k, v in params.items()
                          if k in search_space}})
        if score > best_score:
            best_score, best_params = score, params
        logger.info("trial %d/%d rankic=%.4f best=%.4f", i + 1, n_trials,
                    score, best_score)
    return best_params, pl.DataFrame(trials)
