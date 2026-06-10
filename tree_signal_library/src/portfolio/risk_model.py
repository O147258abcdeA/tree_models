"""简化结构化风险模型。

无完整个股协方差矩阵时，用「风格因子协方差 + 对角特异方差」结构：
  Sigma = B F B' + D
- B: 个股风格/行业暴露矩阵
- F: 因子收益协方差（由历史因子收益估计或给定）
- D: 特异方差对角阵（用个股波动率近似）
并提供主动组合跟踪误差估计 estimate_tracking_error。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


def build_exposure_matrix(df_day: pl.DataFrame, style_cols: list[str],
                          industry_col: str | None = "industry") -> tuple[np.ndarray, list[str]]:
    """单日个股因子暴露矩阵 B（风格 + 行业 dummy），返回 (B, 因子名)。"""
    parts, names = [], []
    if style_cols:
        x = df_day.select([
            pl.col(c).cast(pl.Float64).fill_null(0.0) for c in style_cols
        ]).to_numpy()
        parts.append(x)
        names += style_cols
    if industry_col and industry_col in df_day.columns:
        dummies = (
            df_day.select(pl.col(industry_col).cast(pl.Utf8).fill_null("UNKNOWN"))
            .to_dummies(industry_col)
        )
        parts.append(dummies.to_numpy().astype(np.float64))
        names += dummies.columns
    if not parts:
        raise ValueError("no exposure columns available")
    return np.hstack(parts), names


class SimpleRiskModel:
    """Sigma = B F B' + D 的简化风险模型。

    Parameters
    ----------
    factor_cov : 因子协方差 F（日频）。None 时用单位阵 * factor_var。
    factor_var : 因子方差默认值（日频）。
    """

    def __init__(self, exposures: np.ndarray, specific_var: np.ndarray,
                 factor_cov: np.ndarray | None = None,
                 factor_var: float = 1e-4):
        n, k = exposures.shape
        if specific_var.shape != (n,):
            raise ValueError("specific_var shape mismatch")
        self.b = exposures
        self.f = factor_cov if factor_cov is not None else np.eye(k) * factor_var
        self.d = np.clip(specific_var, 1e-8, None)

    def covariance(self) -> np.ndarray:
        """显式协方差矩阵（小规模或调试用；优化器内部直接用结构化形式）。"""
        return self.b @ self.f @ self.b.T + np.diag(self.d)

    def portfolio_variance(self, w: np.ndarray) -> float:
        """w' Sigma w，利用结构避免显式 N x N 矩阵。"""
        bw = self.b.T @ w
        return float(bw @ self.f @ bw + (self.d * w ** 2).sum())


def from_panel(df_day: pl.DataFrame, style_cols: list[str],
               industry_col: str | None = "industry",
               vol_col: str = "volatility",
               factor_cov: np.ndarray | None = None) -> SimpleRiskModel:
    """从单日面板构造简化风险模型；特异方差用个股波动率平方（日频）近似。"""
    b, _ = build_exposure_matrix(df_day, style_cols, industry_col)
    if vol_col in df_day.columns:
        vol = df_day.get_column(vol_col).cast(pl.Float64).fill_null(
            df_day.get_column(vol_col).cast(pl.Float64).median()).to_numpy()
        spec = (vol / np.sqrt(TRADING_DAYS)) ** 2
    else:
        logger.warning("no %s column; using flat specific variance", vol_col)
        spec = np.full(df_day.height, (0.02) ** 2)
    return SimpleRiskModel(b, spec, factor_cov)


def estimate_tracking_error(active_weight: np.ndarray,
                            risk_model: SimpleRiskModel) -> float:
    """年化跟踪误差：sqrt(252 * w_a' Sigma w_a)。"""
    return float(np.sqrt(TRADING_DAYS * risk_model.portfolio_variance(active_weight)))
