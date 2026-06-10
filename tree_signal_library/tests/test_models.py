"""模型模块测试：三家族 regression / ranking 单窗口闭环 + 信号 schema。"""
import datetime as dt

import polars as pl
import pytest

from src.data.label_builder import build_all_labels
from src.models.model_factory import build_model_name, get_runner
from src.signals.signal_writer import (build_signal_frame, validate_signal_frame,
                                       SIGNAL_SCHEMA)
from src.utils.config import load_config

FAST_PARAMS = {
    "lightgbm": {"objective": "regression", "metric": "l2", "learning_rate": 0.1,
                 "num_leaves": 15, "min_data_in_leaf": 20, "verbosity": -1,
                 "num_boost_round": 50, "early_stopping_rounds": 10},
    "xgboost": {"objective": "reg:squarederror", "eval_metric": "rmse",
                "eta": 0.1, "max_depth": 3, "min_child_weight": 10,
                "tree_method": "hist", "num_boost_round": 50,
                "early_stopping_rounds": 10},
    "catboost": {"loss_function": "RMSE", "eval_metric": "RMSE",
                 "iterations": 50, "learning_rate": 0.1, "depth": 3,
                 "early_stopping_rounds": 10, "verbose": False},
}
FAST_RANK_PARAMS = {
    "lightgbm": {**FAST_PARAMS["lightgbm"], "objective": "lambdarank",
                 "metric": "ndcg", "eval_at": [10], "label_gain": list(range(5))},
    "xgboost": {**FAST_PARAMS["xgboost"], "objective": "rank:pairwise",
                "eval_metric": "ndcg"},
    "catboost": {**FAST_PARAMS["catboost"], "loss_function": "YetiRank",
                 "eval_metric": "NDCG"},
}


@pytest.fixture(scope="module")
def model_data(synthetic_panel, factor_cols):
    cfg = load_config("label_config.yaml")
    cfg["horizons"] = [5]
    labels, _ = build_all_labels(synthetic_panel, cfg)
    df = synthetic_panel.join(labels, on=["date", "stock_id"], how="inner") \
        .sort(["date", "stock_id"])
    dates = df.get_column("date").unique().sort().to_list()
    train = df.filter(pl.col("date") <= dates[150])
    valid = df.filter((pl.col("date") > dates[160]) & (pl.col("date") <= dates[200]))
    test = df.filter(pl.col("date") > dates[210])
    return train, valid, test


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost"])
def test_regression_one_window(model_data, factor_cols, family):
    train, valid, test = model_data
    runner = get_runner(family, "regression")
    result = runner(train, valid, test, factor_cols, "y_excess_5",
                    FAST_PARAMS[family], objective_type="regression",
                    eval_label_col="y_excess_5", seed=42)
    preds = result["predictions"]
    assert preds.columns == ["date", "stock_id", "raw_score"]
    assert preds.height == test.height
    assert result["best_iteration"] >= 0
    assert result["ic_valid"] is not None
    # factor_001 含真实 alpha：IC 应为正
    assert result["ic_valid"] > 0
    assert result["feature_importance"].height > 0


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost"])
def test_ranking_one_window(model_data, factor_cols, family):
    train, valid, test = model_data
    runner = get_runner(family, "ranking")
    result = runner(train, valid, test, factor_cols, "y_group_5",
                    FAST_RANK_PARAMS[family], objective_type="ranking",
                    eval_label_col="y_excess_5", seed=42)
    assert result["predictions"].height == test.height
    assert result["rankic_valid"] is not None


def test_save_load_roundtrip(model_data, factor_cols, tmp_path):
    """可回放性：保存后重新加载预测，与库中信号逐位一致。"""
    import numpy as np
    from src.models.lgbm_model import (load_lgb_model, predict_lgb,
                                       save_lgb_model)
    train, valid, test = model_data
    runner = get_runner("lightgbm", "regression")
    result = runner(train, valid, test, factor_cols, "y_excess_5",
                    FAST_PARAMS["lightgbm"], objective_type="regression")
    path = save_lgb_model(result["model"], tmp_path / "m.txt")
    reloaded = load_lgb_model(path)
    p2 = predict_lgb(reloaded, test, factor_cols)
    np.testing.assert_allclose(
        result["predictions"].get_column("raw_score").to_numpy(), p2, rtol=1e-10)


def test_build_model_name():
    assert build_model_name("lightgbm", "regression", "excess_return", 5, "v01") \
        == "lgbm_reg_excess_h5_v01"


def test_signal_frame_schema_and_validation(model_data, factor_cols):
    train, valid, test = model_data
    runner = get_runner("lightgbm", "regression")
    result = runner(train, valid, test, factor_cols, "y_excess_5",
                    FAST_PARAMS["lightgbm"], objective_type="regression")
    meta = {
        "model_family": "lightgbm", "model_name": "lgbm_reg_excess_h5_v01",
        "model_version": "v01", "feature_version": "f01", "label_version": "l01",
        "label_type": "excess_return", "horizon": 5,
        "objective_type": "regression", "window_id": "w000_20201001",
        "train_start": dt.date(2020, 1, 1), "train_end": dt.date(2020, 8, 1),
        "valid_start": dt.date(2020, 8, 2), "valid_end": dt.date(2020, 10, 1),
        "test_start": dt.date(2020, 10, 2), "test_end": dt.date(2020, 12, 31),
        "ic_valid": result["ic_valid"], "rankic_valid": result["rankic_valid"],
        "icir_valid": result["icir_valid"],
        "best_iteration": result["best_iteration"],
    }
    frame = build_signal_frame(result["predictions"], meta)
    assert list(frame.columns) == list(SIGNAL_SCHEMA)
    validate_signal_frame(frame)  # 不应抛错
    # 缺版本必须拒绝入库
    bad = frame.with_columns(pl.lit("").alias("model_version"))
    with pytest.raises(ValueError):
        validate_signal_frame(bad)
