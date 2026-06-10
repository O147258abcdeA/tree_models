"""中证1000指数增强组合优化（cvxpy）。

目标：
  maximize  active' alpha - lambda * active' Sigma active - gamma * turnover_cost
约束：
  预算 sum(w)=1、w>=0、个股上限、个股主动偏离、行业主动偏离、
  风格暴露带、跟踪误差上限（SOC）、换手上限、成交额容量。
Sigma 采用 risk_model.SimpleRiskModel 的结构化形式（B F B' + D），
避免显式 5000x5000 矩阵。
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import polars as pl

from src.portfolio.risk_model import (SimpleRiskModel, build_exposure_matrix,
                                      estimate_tracking_error, from_panel)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


def build_constraints(w: cp.Variable, df_day: pl.DataFrame, cfg: dict,
                      bench: np.ndarray, prev_w: np.ndarray | None,
                      risk: SimpleRiskModel) -> list:
    """构造线性 / SOC 约束列表（缺列的约束自动跳过并告警，不静默假装满足）。"""
    c = cfg.get("constraints", {})
    cons = [cp.sum(w) == 1, w >= 0]
    active = w - bench

    if c.get("max_weight") is not None:
        cons.append(w <= float(c["max_weight"]))
    if c.get("max_active_weight") is not None:
        cons.append(cp.abs(active) <= float(c["max_active_weight"]))

    # 行业主动偏离
    if c.get("max_industry_active") is not None and "industry" in df_day.columns:
        dummies = (df_day.select(pl.col("industry").cast(pl.Utf8).fill_null("UNKNOWN"))
                   .to_dummies("industry").to_numpy().astype(float))
        cons.append(cp.abs(dummies.T @ active) <= float(c["max_industry_active"]))

    # 风格暴露带（含市值 / Beta / 波动率 / 流动性 / 估值 / 成长 / 质量）
    for col, bound in (c.get("style_bounds") or {}).items():
        if col in df_day.columns:
            x = df_day.get_column(col).cast(pl.Float64).fill_null(0.0).to_numpy()
            cons.append(cp.abs(x @ active) <= float(bound))
        else:
            logger.warning("style bound skipped, column missing: %s", col)

    # 跟踪误差上限：||[F^{1/2} B' a; sqrt(D) a]||_2 <= TE / sqrt(252)
    if c.get("max_tracking_error") is not None:
        f_half = np.linalg.cholesky(risk.f + 1e-12 * np.eye(risk.f.shape[0]))
        te_daily = float(c["max_tracking_error"]) / np.sqrt(TRADING_DAYS)
        cons.append(
            cp.norm(cp.hstack([f_half.T @ (risk.b.T @ active),
                               cp.multiply(np.sqrt(risk.d), active)])) <= te_daily)

    # 换手上限（双边）
    if prev_w is not None and c.get("max_turnover") is not None:
        cons.append(cp.norm1(w - prev_w) <= float(c["max_turnover"]))

    # 成交额容量：w * portfolio_value <= participation * ADV
    if c.get("adv_participation") is not None and "adv" in df_day.columns:
        adv = df_day.get_column("adv").cast(pl.Float64).fill_null(0.0).to_numpy()
        pv = float(c.get("portfolio_value", 1e8))
        cons.append(cp.multiply(pv, w) <= float(c["adv_participation"]) * adv)
    return cons


def optimize_portfolio_one_day(df_day: pl.DataFrame, cfg: dict,
                               prev_w: np.ndarray | None = None,
                               style_cols: list[str] | None = None,
                               factor_cov: np.ndarray | None = None) -> pl.DataFrame:
    """单日指数增强优化。

    df_day 必含：stock_id, alpha_score, benchmark_weight；可含 industry、
    风格列、volatility、adv。返回 [stock_id, weight, benchmark_weight,
    active_weight]。求解失败时抛错，不返回不可用解。
    """
    n = df_day.height
    if n == 0:
        raise ValueError("empty cross-section for optimization")
    alpha = df_day.get_column("alpha_score").cast(pl.Float64).fill_null(0.0).to_numpy()
    bench = df_day.get_column("benchmark_weight").cast(pl.Float64).fill_null(0.0).to_numpy()
    bench = bench / bench.sum() if bench.sum() > 0 else np.full(n, 1.0 / n)

    style_cols = [c for c in (style_cols or
                              ["log_mktcap", "beta", "volatility", "liquidity",
                               "value", "growth", "quality"]) if c in df_day.columns]
    risk = from_panel(df_day, style_cols, factor_cov=factor_cov)

    w = cp.Variable(n)
    active = w - bench
    lam = float(cfg.get("risk_aversion", 10.0))
    gamma = float(cfg.get("turnover_penalty", 1.0))
    cost_rate = float(cfg.get("cost_rate", 0.0015))

    # 结构化主动风险：a'Sigma a = ||F^{1/2} B'a||^2 + sum(d a^2)
    f_half = np.linalg.cholesky(risk.f + 1e-12 * np.eye(risk.f.shape[0]))
    risk_term = cp.sum_squares(f_half.T @ (risk.b.T @ active)) \
        + cp.sum(cp.multiply(risk.d, cp.square(active)))
    objective = alpha @ active - lam * risk_term
    if prev_w is not None:
        objective = objective - gamma * cost_rate * cp.norm1(w - prev_w)

    cons = build_constraints(w, df_day, cfg, bench, prev_w, risk)
    prob = cp.Problem(cp.Maximize(objective), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"optimization failed: status={prob.status}")

    w_opt = np.clip(np.asarray(w.value).reshape(-1), 0.0, None)
    w_opt = w_opt / w_opt.sum()
    out = df_day.select(["stock_id"]).with_columns(
        pl.Series("weight", w_opt),
        pl.Series("benchmark_weight", bench),
        pl.Series("active_weight", w_opt - bench),
    )
    logger.info("optimized: n=%d active_names=%d TE=%.4f",
                n, int((np.abs(w_opt - bench) > 1e-6).sum()),
                estimate_tracking_error(w_opt - bench, risk))
    return out


def calculate_turnover(prev_w: np.ndarray | None, new_w: np.ndarray) -> float:
    """双边换手率：sum(|w_new - w_old|)。首期为建仓，换手 = sum(w)。"""
    if prev_w is None:
        return float(np.abs(new_w).sum())
    return float(np.abs(new_w - prev_w).sum())
