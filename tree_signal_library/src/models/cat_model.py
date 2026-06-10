"""CatBoost 模型模块：Regression + Ranking（可选 Classification）。

ranking：使用 CatBoost Pool，正确传入 group_id（每个 date 一个 group）与
cat_features，不允许跨日期混合 ranking group。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoost, Pool

from src.models.rank_utils import (build_group_ids, drop_null_labels,
                                   to_feature_matrix, to_label_array,
                                   to_rank_relevance)
from src.utils.logger import get_logger
from src.utils.metrics import daily_ic, daily_rank_ic, ic_summary
from src.utils.seed import seed_params

logger = get_logger(__name__)


def prepare_cat_pool(df: pl.DataFrame, feature_cols: list[str],
                     label_col: str,
                     cat_features: list[str] | None = None) -> Pool:
    """构造回归 / 分类用 Pool（先剔除 null 标签）。

    cat_features 为分类特征列名（须包含在 feature_cols 内）。
    """
    df = drop_null_labels(df, label_col)
    return Pool(
        to_feature_matrix(df, feature_cols),
        label=to_label_array(df, label_col),
        feature_names=feature_cols,
        cat_features=cat_features,
    )


def prepare_cat_rank_pool(df: pl.DataFrame, feature_cols: list[str],
                          label_col: str,
                          cat_features: list[str] | None = None) -> Pool:
    """构造 ranking 用 Pool：每个 date 编码为一个 group_id，按 date 排序传入。"""
    df = drop_null_labels(df, label_col).sort(["date", "stock_id"])
    return Pool(
        to_feature_matrix(df, feature_cols),
        label=to_rank_relevance(df, label_col),
        group_id=build_group_ids(df),
        feature_names=feature_cols,
        cat_features=cat_features,
    )


def _train(params: dict, train_pool: Pool, valid_pool: Pool, seed: int) -> CatBoost:
    p = {**params, **seed_params("catboost", seed)}
    model = CatBoost(p)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True,
              early_stopping_rounds=params.get("early_stopping_rounds", 100))
    logger.info("cat trained: best_iteration=%s", model.get_best_iteration())
    return model


def train_cat_regression(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                         feature_cols: list[str], label_col: str,
                         params: dict, seed: int = 42,
                         cat_features: list[str] | None = None) -> CatBoost:
    """CatBoost Regression（RMSE / MAE / Huber / Quantile）。"""
    train_pool = prepare_cat_pool(train_df, feature_cols, label_col, cat_features)
    valid_pool = prepare_cat_pool(valid_df, feature_cols, label_col, cat_features)
    return _train(params, train_pool, valid_pool, seed)


def train_cat_ranking(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                      feature_cols: list[str], label_col: str,
                      params: dict, seed: int = 42,
                      cat_features: list[str] | None = None) -> CatBoost:
    """CatBoost Ranking（YetiRank / PairLogit / QueryRMSE），按 date 设置 group_id。"""
    train_pool = prepare_cat_rank_pool(train_df, feature_cols, label_col, cat_features)
    valid_pool = prepare_cat_rank_pool(valid_df, feature_cols, label_col, cat_features)
    return _train(params, train_pool, valid_pool, seed)


def predict_cat(model: CatBoost, df: pl.DataFrame,
                feature_cols: list[str]) -> np.ndarray:
    """样本外预测（use_best_model=True 训练，已截断到 best_iteration）。"""
    return np.asarray(
        model.predict(Pool(to_feature_matrix(df, feature_cols),
                           feature_names=feature_cols))
    ).reshape(-1)


def save_cat_model(model: CatBoost, path: str | Path) -> Path:
    """保存模型（cbm 二进制格式）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_cat_model(path: str | Path) -> CatBoost:
    """加载模型。"""
    model = CatBoost()
    model.load_model(str(path))
    return model


def get_cat_feature_importance(model: CatBoost,
                               train_pool: Pool | None = None) -> pl.DataFrame:
    """特征重要性，降序。

    ranking 模型（LossFunctionChange 口径）必须传入训练 Pool；
    regression 默认 PredictionValuesChange，无需数据。
    """
    names = model.feature_names_
    imp = model.get_feature_importance(data=train_pool)
    return pl.DataFrame({"feature": names, "importance": imp}).sort(
        "importance", descending=True)


def run_cat_one_window(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                       test_df: pl.DataFrame, feature_cols: list[str],
                       label_col: str, params: dict,
                       objective_type: str = "regression",
                       eval_label_col: str | None = None,
                       seed: int = 42,
                       cat_features: list[str] | None = None) -> dict:
    """单窗口完整流程：训练 + valid IC/RankIC + test 样本外预测（接口同 lgbm/xgb）。"""
    if objective_type == "regression":
        model = train_cat_regression(train_df, valid_df, feature_cols, label_col,
                                     params, seed, cat_features)
        importance = get_cat_feature_importance(model)
    elif objective_type == "ranking":
        model = train_cat_ranking(train_df, valid_df, feature_cols, label_col,
                                  params, seed, cat_features)
        importance = get_cat_feature_importance(
            model, prepare_cat_rank_pool(train_df, feature_cols, label_col,
                                         cat_features))
    else:
        raise ValueError(f"unsupported objective_type for catboost: {objective_type}")

    eval_col = eval_label_col or label_col
    vpred = valid_df.with_columns(
        pl.Series("pred", predict_cat(model, valid_df, feature_cols)))
    ic = ic_summary(daily_ic(vpred, "pred", eval_col))
    ric = ic_summary(daily_rank_ic(vpred, "pred", eval_col), "rank_ic")

    preds = test_df.select(["date", "stock_id"]).with_columns(
        pl.Series("raw_score", predict_cat(model, test_df, feature_cols)))
    return {
        "model": model,
        "best_iteration": int(model.get_best_iteration() or 0),
        "ic_valid": ic["ic_mean"],
        "rankic_valid": ric["ic_mean"],
        "icir_valid": ic["icir"],
        "rankicir_valid": ric["icir"],
        "predictions": preds,
        "feature_importance": importance,
    }
