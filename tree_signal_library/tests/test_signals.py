"""信号处理 / 中性化 / 评估 / 融合测试。"""
import numpy as np
import polars as pl
import pytest

from src.signals.signal_ensemble import (compute_rolling_icir,
                                         equal_weight_ensemble, family_ensemble,
                                         icir_weighted_ensemble, run_ensemble)
from src.signals.signal_neutralizer import (industry_neutralize_signal,
                                            style_neutralize_signal)
from src.signals.signal_processor import (PROCESSED_COLS,
                                          process_one_model_signal,
                                          rank_signal, winsorize_signal,
                                          zscore_signal)
from src.utils.metrics import daily_ic, group_returns, ic_summary


@pytest.fixture(scope="module")
def raw_signal(synthetic_panel):
    """用 factor_001 当作 raw_score 构造两个模型的信号表。"""
    base = synthetic_panel.select(["date", "stock_id"]).with_columns(
        synthetic_panel.get_column("factor_001").alias("raw_score"))
    frames = []
    for name, fam in [("lgbm_reg_excess_h5_v01", "lightgbm"),
                      ("xgb_reg_excess_h5_v01", "xgboost")]:
        noise = np.random.default_rng(1).normal(0, 0.5, base.height)
        frames.append(base.with_columns(
            (pl.col("raw_score") + noise).alias("raw_score"),
            pl.lit(fam).alias("model_family"), pl.lit(name).alias("model_name"),
            pl.lit("v01").alias("model_version"),
            pl.lit("excess_return").alias("label_type"),
            pl.lit(5).cast(pl.Int32).alias("horizon"),
            pl.lit("regression").alias("objective_type"),
        ))
    return pl.concat(frames)


@pytest.fixture(scope="module")
def exposure(synthetic_panel):
    return synthetic_panel.select(["date", "stock_id", "industry", "log_mktcap",
                                   "beta", "turnover", "volatility"])


@pytest.fixture(scope="module")
def returns(synthetic_panel):
    return synthetic_panel.sort(["stock_id", "date"]).with_columns([
        (pl.col("close").shift(-h).over("stock_id") / pl.col("close") - 1.0)
        .alias(f"future_return_{h}d") for h in (1, 5)
    ]).select(["date", "stock_id", "future_return_1d", "future_return_5d"])


def test_zscore_and_rank_daily(raw_signal):
    one = raw_signal.filter(pl.col("model_name") == "lgbm_reg_excess_h5_v01")
    z = zscore_signal(one, "raw_score", "zscore_score")
    daily = z.group_by("date").agg(pl.col("zscore_score").mean().alias("m"),
                                   pl.col("zscore_score").std().alias("s"))
    assert daily.get_column("m").abs().max() < 1e-6
    assert (daily.get_column("s") - 1.0).abs().max() < 1e-6
    r = rank_signal(one, "raw_score")
    assert r.get_column("rank_score").min() >= 0.0
    assert r.get_column("rank_score").max() <= 1.0


def test_winsorize_signal(raw_signal):
    one = raw_signal.filter(pl.col("model_name") == "lgbm_reg_excess_h5_v01")
    w = winsorize_signal(one, "raw_score", 0.05, 0.95, out_col="w")
    d = one.get_column("date").min()
    day_w = w.filter(pl.col("date") == d)
    day_r = one.filter(pl.col("date") == d)
    assert day_w.get_column("w").max() <= \
        day_r.get_column("raw_score").quantile(0.95, "linear") + 1e-9


def test_industry_neutralize(raw_signal, exposure):
    one = raw_signal.filter(pl.col("model_name") == "lgbm_reg_excess_h5_v01") \
        .join(exposure, on=["date", "stock_id"])
    out = industry_neutralize_signal(one, "raw_score", out_col="n")
    means = out.group_by(["date", "industry"]).agg(pl.col("n").mean().alias("m"))
    assert means.get_column("m").abs().max() < 1e-9


def test_style_neutralize_kills_correlation(raw_signal, exposure):
    one = raw_signal.filter(pl.col("model_name") == "lgbm_reg_excess_h5_v01") \
        .join(exposure, on=["date", "stock_id"]) \
        .with_columns((pl.col("raw_score") + 0.5 * pl.col("log_mktcap")).alias("raw_score"))
    out = style_neutralize_signal(one, "raw_score", ["log_mktcap", "beta"],
                                  "industry", out_col="n")
    d = out.get_column("date").min()
    day = out.filter(pl.col("date") == d)
    assert abs(day.select(pl.corr("n", "log_mktcap")).item()) < 1e-6


def test_process_one_model_signal_full(raw_signal, exposure):
    one = raw_signal.filter(pl.col("model_name") == "lgbm_reg_excess_h5_v01")
    cfg = {"winsorize": {"enabled": True}, "industry_neutralize": True,
           "style_neutralize": True,
           "style_cols": ["log_mktcap", "beta", "turnover", "volatility"]}
    out = process_one_model_signal(one, exposure, cfg)
    assert out.columns == PROCESSED_COLS
    assert out.height == one.height
    # neutral_score 再标准化：日内均值 0 / std 1
    daily = out.group_by("date").agg(pl.col("neutral_score").mean().alias("m"))
    assert daily.get_column("m").abs().max() < 1e-6


def test_metrics_perfect_signal(returns):
    """signal == future return 时 IC=1；分组收益单调。"""
    df = returns.drop_nulls("future_return_1d").with_columns(
        pl.col("future_return_1d").alias("sig"))
    ic = daily_ic(df, "sig", "future_return_1d")
    assert (ic.get_column("ic") - 1.0).abs().max() < 1e-9
    grp = group_returns(df, "sig", "future_return_1d", 5)
    mean_by_grp = grp.group_by("group").agg(
        pl.col("group_return").mean()).sort("group").get_column("group_return")
    assert mean_by_grp.to_list() == sorted(mean_by_grp.to_list())


def test_equal_weight_ensemble(raw_signal):
    sig = rank_signal(raw_signal, "raw_score")
    ens = equal_weight_ensemble(sig, "rank_score")
    assert ens.columns == ["date", "stock_id", "ensemble_score"]
    n_pairs = raw_signal.select(["date", "stock_id"]).unique().height
    assert ens.height == n_pairs


def test_icir_weights_no_future_info(raw_signal, returns):
    sig = zscore_signal(raw_signal, "raw_score", "neutral_score")
    icir = compute_rolling_icir(sig, returns, horizon=5, lookback_days=60)
    valid = icir.drop_nulls("info_as_of_date")
    # 时序审计：权重信息日期必须严格早于应用日期
    assert valid.filter(pl.col("info_as_of_date") >= pl.col("date")).height == 0


def test_icir_weighted_ensemble(raw_signal, returns):
    sig = zscore_signal(raw_signal, "raw_score", "neutral_score")
    ens, weights = icir_weighted_ensemble(sig, returns, horizon=5,
                                          lookback_days=60, min_history_days=10)
    daily_w = weights.group_by("date").agg(pl.col("weight").sum().alias("s"))
    assert (daily_w.get_column("s") - 1.0).abs().max() < 1e-9
    assert ens.get_column("ensemble_score").null_count() == 0


def test_family_ensemble(raw_signal):
    sig = zscore_signal(raw_signal, "raw_score", "neutral_score")
    ens = family_ensemble(sig, "neutral_score", {"lightgbm": 0.5, "xgboost": 0.5})
    assert "ensemble_score" in ens.columns


def test_run_ensemble_dispatch(raw_signal, returns):
    sig = zscore_signal(rank_signal(raw_signal, "raw_score"),
                        "raw_score", "neutral_score")
    for method in ("equal_rank", "equal_neutral", "family", "stacking",
                   "icir_weighted"):
        ens, _ = run_ensemble(sig, {"method": method, "lookback_days": 60,
                                    "min_history_days": 10,
                                    "stacking": {"refit_freq_days": 21}},
                              returns, horizon=5)
        assert ens.height > 0
    with pytest.raises(ValueError):
        run_ensemble(sig, {"method": "nope"})
