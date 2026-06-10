"""XGBoost 模型模块：Regression + Ranking（可选 Classification）。

ranking：必须构造 DMatrix 并 set_group，group 顺序与训练数据 date 排序一致，
不允许跨日期混合 ranking group。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from src.models.rank_utils import (build_group_sizes, drop_null_labels,
                                   to_feature_matrix, to_label_array,
                                   to_rank_relevance)
from src.utils.logger import get_logger
from src.utils.metrics import daily_ic, daily_rank_ic, ic_summary
from src.utils.seed import seed_params

logger = get_logger(__name__)

NON_BOOSTER_KEYS = {"num_boost_round", "early_stopping_rounds"}


def _booster_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in NON_BOOSTER_KEYS}


def prepare_xgb_dmatrix(df: pl.DataFrame, feature_cols: list[str],
                        label_col: str) -> xgb.DMatrix:
    """构造回归 / 分类用 DMatrix（先剔除 null 标签）。"""
    df = drop_null_labels(df, label_col)
    return xgb.DMatrix(to_feature_matrix(df, feature_cols),
                       label=to_label_array(df, label_col),
                       feature_names=feature_cols)


def prepare_xgb_rank_dmatrix(df: pl.DataFrame, feature_cols: list[str],
                             label_col: str) -> xgb.DMatrix:
    """构造 ranking 用 DMatrix：每个 date 一个 query group。"""
    df = drop_null_labels(df, label_col).sort(["date", "stock_id"])
    group = build_group_sizes(df)
    dm = xgb.DMatrix(to_feature_matrix(df, feature_cols),
                     label=to_rank_relevance(df, label_col),
                     feature_names=feature_cols)
    dm.set_group(group)
    return dm


def _train(params: dict, dtrain: xgb.DMatrix, dvalid: xgb.DMatrix,
           seed: int) -> xgb.Booster:
    booster_params = {**_booster_params(params), **seed_params("xgboost", seed)}
    model = xgb.train(
        booster_params,
        dtrain,
        num_boost_round=params.get("num_boost_round", 3000),
        evals=[(dvalid, "valid")],
        early_stopping_rounds=params.get("early_stopping_rounds", 100),
        verbose_eval=False,
    )
    logger.info("xgb trained: best_iteration=%d", model.best_iteration)
    return model


def train_xgb_regression(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                         feature_cols: list[str], label_col: str,
                         params: dict, seed: int = 42) -> xgb.Booster:
    """XGBoost Regression（reg:squarederror / reg:pseudohubererror / reg:absoluteerror）。"""
    dtrain = prepare_xgb_dmatrix(train_df, feature_cols, label_col)
    dvalid = prepare_xgb_dmatrix(valid_df, feature_cols, label_col)
    return _train(params, dtrain, dvalid, seed)


def train_xgb_ranking(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                      feature_cols: list[str], label_col: str,
                      params: dict, seed: int = 42) -> xgb.Booster:
    """XGBoost Ranking（rank:pairwise / rank:ndcg / rank:map），按 date 设置 group。"""
    dtrain = prepare_xgb_rank_dmatrix(train_df, feature_cols, label_col)
    dvalid = prepare_xgb_rank_dmatrix(valid_df, feature_cols, label_col)
    return _train(params, dtrain, dvalid, seed)


def predict_xgb(model: xgb.Booster, df: pl.DataFrame,
                feature_cols: list[str]) -> np.ndarray:
    """样本外预测（使用 best_iteration）。"""
    dm = xgb.DMatrix(to_feature_matrix(df, feature_cols), feature_names=feature_cols)
    best = getattr(model, "best_iteration", None)
    if best is not None:
        return model.predict(dm, iteration_range=(0, best + 1))
    return model.predict(dm)


def save_xgb_model(model: xgb.Booster, path: str | Path) -> Path:
    """保存模型（json 格式，保留 best_iteration 属性）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_xgb_model(path: str | Path) -> xgb.Booster:
    """加载模型。"""
    model = xgb.Booster()
    model.load_model(str(path))
    return model


def get_xgb_feature_importance(model: xgb.Booster,
                               importance_type: str = "gain") -> pl.DataFrame:
    """特征重要性，按重要性降序。"""
    score = model.get_score(importance_type=importance_type)
    return pl.DataFrame({
        "feature": list(score.keys()),
        "importance": list(score.values()),
    }).sort("importance", descending=True)


def run_xgb_one_window(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                       test_df: pl.DataFrame, feature_cols: list[str],
                       label_col: str, params: dict,
                       objective_type: str = "regression",
                       eval_label_col: str | None = None,
                       seed: int = 42) -> dict:
    """单窗口完整流程：训练 + valid IC/RankIC + test 样本外预测（接口同 lgbm）。"""
    if objective_type == "regression":
        model = train_xgb_regression(train_df, valid_df, feature_cols, label_col, params, seed)
    elif objective_type == "ranking":
        model = train_xgb_ranking(train_df, valid_df, feature_cols, label_col, params, seed)
    else:
        raise ValueError(f"unsupported objective_type for xgboost: {objective_type}")

    eval_col = eval_label_col or label_col
    vpred = valid_df.with_columns(
        pl.Series("pred", predict_xgb(model, valid_df, feature_cols)))
    ic = ic_summary(daily_ic(vpred, "pred", eval_col))
    ric = ic_summary(daily_rank_ic(vpred, "pred", eval_col), "rank_ic")

    preds = test_df.select(["date", "stock_id"]).with_columns(
        pl.Series("raw_score", predict_xgb(model, test_df, feature_cols)))
    return {
        "model": model,
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
        "ic_valid": ic["ic_mean"],
        "rankic_valid": ric["ic_mean"],
        "icir_valid": ic["icir"],
        "rankicir_valid": ric["icir"],
        "predictions": preds,
        "feature_importance": get_xgb_feature_importance(model),
    }
