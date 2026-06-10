"""LightGBM 模型模块：Regression + Ranking（可选 Classification）。

统一约定：
- 特征矩阵 float32；
- ranking 每个 date 一个 query group（rank_utils 强制 date 排序）；
- 训练用 valid early stopping，并输出 valid 每日横截面 IC / RankIC；
- 模型保存为文本 model.txt + meta.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models.rank_utils import (build_group_sizes, drop_null_labels,
                                   to_feature_matrix, to_label_array,
                                   to_rank_relevance)
from src.utils.logger import get_logger
from src.utils.metrics import daily_ic, daily_rank_ic, ic_summary
from src.utils.seed import seed_params

logger = get_logger(__name__)

DATASET_KEYS = {"num_boost_round", "early_stopping_rounds", "label_gain_max",
                "ndcg_eval_at"}


def _booster_params(params: dict) -> dict:
    """剔除非 booster 参数。"""
    return {k: v for k, v in params.items() if k not in DATASET_KEYS}


def prepare_lgb_dataset(df: pl.DataFrame, feature_cols: list[str],
                        label_col: str,
                        reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """构造回归 / 分类用 lgb.Dataset（先剔除 null 标签）。"""
    df = drop_null_labels(df, label_col)
    return lgb.Dataset(
        to_feature_matrix(df, feature_cols),
        label=to_label_array(df, label_col),
        feature_name=feature_cols,
        reference=reference,
        free_raw_data=True,
    )


def prepare_lgb_rank_dataset(df: pl.DataFrame, feature_cols: list[str],
                             label_col: str,
                             reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """构造 ranking 用 lgb.Dataset：每个 date 一个 query group，按 date 排序传入。"""
    df = drop_null_labels(df, label_col).sort(["date", "stock_id"])
    group = build_group_sizes(df)
    return lgb.Dataset(
        to_feature_matrix(df, feature_cols),
        label=to_rank_relevance(df, label_col),
        group=group,
        feature_name=feature_cols,
        reference=reference,
        free_raw_data=True,
    )


def _train(params: dict, dtrain: lgb.Dataset, dvalid: lgb.Dataset,
           seed: int) -> lgb.Booster:
    """通用训练（early stopping on valid）。"""
    booster_params = {**_booster_params(params), **seed_params("lightgbm", seed)}
    model = lgb.train(
        booster_params,
        dtrain,
        num_boost_round=params.get("num_boost_round", 5000),
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(params.get("early_stopping_rounds", 100), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    logger.info("lgb trained: best_iteration=%d best_score=%s",
                model.best_iteration, dict(model.best_score.get("valid", {})))
    return model


def train_lgb_regression(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                         feature_cols: list[str], label_col: str,
                         params: dict, seed: int = 42) -> lgb.Booster:
    """LightGBM Regression（objective: regression/regression_l1/huber/quantile）。"""
    dtrain = prepare_lgb_dataset(train_df, feature_cols, label_col)
    dvalid = prepare_lgb_dataset(valid_df, feature_cols, label_col, reference=dtrain)
    return _train(params, dtrain, dvalid, seed)


def train_lgb_ranking(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                      feature_cols: list[str], label_col: str,
                      params: dict, seed: int = 42) -> lgb.Booster:
    """LightGBM Ranking（objective: lambdarank/rank_xendcg），按 date 构造 group。"""
    params = dict(params)
    max_rel = int(params.pop("label_gain_max", 9))
    params.setdefault("label_gain", list(range(max_rel + 1)))
    if "ndcg_eval_at" in params:
        params["eval_at"] = params.pop("ndcg_eval_at")
    dtrain = prepare_lgb_rank_dataset(train_df, feature_cols, label_col)
    dvalid = prepare_lgb_rank_dataset(valid_df, feature_cols, label_col, reference=dtrain)
    return _train(params, dtrain, dvalid, seed)


def predict_lgb(model: lgb.Booster, df: pl.DataFrame,
                feature_cols: list[str]) -> np.ndarray:
    """样本外预测（使用 best_iteration）。"""
    return model.predict(to_feature_matrix(df, feature_cols),
                         num_iteration=model.best_iteration or None)


def save_lgb_model(model: lgb.Booster, path: str | Path) -> Path:
    """保存模型（文本格式，保留 best_iteration）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path), num_iteration=model.best_iteration or None)
    return path


def load_lgb_model(path: str | Path) -> lgb.Booster:
    """加载模型。"""
    return lgb.Booster(model_file=str(path))


def get_lgb_feature_importance(model: lgb.Booster,
                               importance_type: str = "gain") -> pl.DataFrame:
    """特征重要性，按重要性降序。"""
    return pl.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type=importance_type),
    }).sort("importance", descending=True)


def run_lgb_one_window(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                       test_df: pl.DataFrame, feature_cols: list[str],
                       label_col: str, params: dict,
                       objective_type: str = "regression",
                       eval_label_col: str | None = None,
                       seed: int = 42) -> dict:
    """单窗口完整流程：训练 + valid IC/RankIC + test 样本外预测。

    Returns
    -------
    dict: model, best_iteration, ic_valid, rankic_valid, icir_valid,
          predictions (test_df + raw_score), feature_importance。
    eval_label_col: 计算 IC 用的连续收益标签（ranking 用整数标签训练时，
    IC 仍应对连续未来收益计算）。
    """
    if objective_type == "regression":
        model = train_lgb_regression(train_df, valid_df, feature_cols, label_col, params, seed)
    elif objective_type == "ranking":
        model = train_lgb_ranking(train_df, valid_df, feature_cols, label_col, params, seed)
    else:
        raise ValueError(f"unsupported objective_type for lightgbm: {objective_type}")

    eval_col = eval_label_col or label_col
    vpred = valid_df.with_columns(
        pl.Series("pred", predict_lgb(model, valid_df, feature_cols)))
    ic = ic_summary(daily_ic(vpred, "pred", eval_col))
    ric = ic_summary(daily_rank_ic(vpred, "pred", eval_col), "rank_ic")

    preds = test_df.select(["date", "stock_id"]).with_columns(
        pl.Series("raw_score", predict_lgb(model, test_df, feature_cols)))
    return {
        "model": model,
        "best_iteration": model.best_iteration,
        "ic_valid": ic["ic_mean"],
        "rankic_valid": ric["ic_mean"],
        "icir_valid": ic["icir"],
        "rankicir_valid": ric["icir"],
        "predictions": preds,
        "feature_importance": get_lgb_feature_importance(model),
    }
