"""组合优化与多空回测测试。"""
import numpy as np
import polars as pl
import pytest

from src.portfolio.backtest import (build_long_short_weights,
                                    calculate_long_short_return,
                                    neutralize_long_short_weights,
                                    run_long_short_backtest)
from src.portfolio.optimizer import calculate_turnover, optimize_portfolio_one_day
from src.portfolio.risk_model import SimpleRiskModel, estimate_tracking_error


@pytest.fixture(scope="module")
def one_day(synthetic_panel):
    d = synthetic_panel.get_column("date").min()
    return synthetic_panel.filter(pl.col("date") == d).with_columns(
        pl.col("factor_001").alias("alpha_score"))


def test_optimizer_one_day(one_day):
    cfg = {"risk_aversion": 5.0, "turnover_penalty": 1.0, "cost_rate": 0.0015,
           "constraints": {"max_weight": 0.10, "max_active_weight": 0.08,
                           "max_industry_active": 0.10}}
    out = optimize_portfolio_one_day(one_day, cfg)
    w = out.get_column("weight").to_numpy()
    assert abs(w.sum() - 1.0) < 1e-6
    assert w.min() >= -1e-8
    assert w.max() <= 0.10 + 1e-6
    active = out.get_column("active_weight").to_numpy()
    assert np.abs(active).max() <= 0.08 + 1e-6
    # 有 alpha 的优化应偏向高 alpha 股票
    alpha = one_day.get_column("alpha_score").to_numpy()
    assert np.corrcoef(alpha, active)[0, 1] > 0


def test_optimizer_turnover_constraint(one_day):
    cfg = {"risk_aversion": 5.0,
           "constraints": {"max_weight": 0.10, "max_turnover": 0.05}}
    n = one_day.height
    prev = np.full(n, 1.0 / n)
    out = optimize_portfolio_one_day(one_day, cfg, prev_w=prev)
    assert calculate_turnover(prev, out.get_column("weight").to_numpy()) <= 0.05 + 1e-6


def test_risk_model_tracking_error():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 3))
    rm = SimpleRiskModel(b, np.full(50, 1e-4))
    a = rng.normal(size=50) * 0.01
    te = estimate_tracking_error(a, rm)
    assert te > 0
    # 结构化方差与显式协方差一致
    assert abs(rm.portfolio_variance(a) - a @ rm.covariance() @ a) < 1e-12


def test_long_short_weights(synthetic_panel):
    sig = synthetic_panel.select(["date", "stock_id"]).with_columns(
        synthetic_panel.get_column("factor_001").alias("signal_score"))
    w = build_long_short_weights(sig, 0.1, 0.1, "equal")
    daily = w.group_by("date").agg(
        pl.col("weight").filter(pl.col("weight") > 0).sum().alias("long"),
        pl.col("weight").filter(pl.col("weight") < 0).sum().alias("short"))
    assert (daily.get_column("long") - 1.0).abs().max() < 1e-9
    assert (daily.get_column("short") + 1.0).abs().max() < 1e-9


def test_long_short_backtest_positive_for_true_alpha(synthetic_panel):
    """factor_001 驱动收益 -> 多空策略应有正收益（信号纯度检验）。"""
    sig = synthetic_panel.select(["date", "stock_id"]).with_columns(
        synthetic_panel.get_column("factor_001").alias("signal_score"))
    rets = synthetic_panel.select(["date", "stock_id", "return"])
    cfg = {"long_short": {"top_quantile": 0.2, "bottom_quantile": 0.2,
                          "weighting": "equal", "rebalance_freq": "daily",
                          "cost_rate": 0.0}}
    result = run_long_short_backtest(sig, rets, cfg)
    assert result["summary"]["ann_return"] > 0
    assert 0 <= result["summary"]["max_drawdown"] <= 1
    assert result["daily"].height > 200


def test_neutralized_weights_zero_beta_exposure(synthetic_panel):
    sig = synthetic_panel.select(["date", "stock_id"]).with_columns(
        synthetic_panel.get_column("factor_001").alias("signal_score"))
    expo = synthetic_panel.select(["date", "stock_id", "industry",
                                   "log_mktcap", "beta"])
    w = build_long_short_weights(sig, 0.2, 0.2)
    nw = neutralize_long_short_weights(w, expo, beta_neutral=True)
    chk = nw.join(expo, on=["date", "stock_id"]).group_by("date").agg(
        ((pl.col("beta") - pl.col("beta").mean()) * pl.col("weight")).sum().alias("e"))
    assert chk.get_column("e").abs().max() < 1e-9
