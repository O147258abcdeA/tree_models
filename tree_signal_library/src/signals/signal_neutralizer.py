"""信号中性化模块：行业中性化 + 风格中性化（逐日横截面回归取残差）。

全部按 date 分组、单日内计算，禁止跨日期、禁止未来数据。
"""
from __future__ import annotations

import polars as pl

from src.data.preprocess import build_design_matrix, regression_residual
from src.utils.logger import get_logger

logger = get_logger(__name__)


def industry_neutralize_signal(df: pl.DataFrame, score_col: str,
                               industry_col: str = "industry",
                               out_col: str = "neutral_score",
                               date_col: str = "date") -> pl.DataFrame:
    """行业中性化：signal_i - mean(signal within (date, industry))。"""
    return df.with_columns(
        (pl.col(score_col)
         - pl.col(score_col).mean().over([date_col, industry_col]))
        .alias(out_col)
    )


def style_neutralize_signal(df: pl.DataFrame, score_col: str,
                            style_cols: list[str],
                            industry_col: str | None = "industry",
                            out_col: str = "neutral_score",
                            date_col: str = "date") -> pl.DataFrame:
    """风格中性化：每日横截面回归
    signal_i = a + b'style_i + industry_dummies + e_i，取 e_i。

    回归用 numpy lstsq；行业 dummy 由 polars to_dummies 生成。
    """
    def _one_day(day: pl.DataFrame) -> pl.DataFrame:
        x = build_design_matrix(
            day, style_cols,
            industry_col if industry_col and industry_col in day.columns else None)
        y = day.get_column(score_col).cast(pl.Float64).to_numpy()
        res = regression_residual(y, x)
        return day.with_columns(pl.Series(out_col, res, dtype=pl.Float64))

    return (
        df.sort(date_col)
        .group_by(date_col, maintain_order=True)
        .map_groups(_one_day)
    )
