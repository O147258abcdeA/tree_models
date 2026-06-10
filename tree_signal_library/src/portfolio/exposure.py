"""组合行业 / 风格暴露计算。"""
from __future__ import annotations

import polars as pl

STYLE_COLS_DEFAULT = ["log_mktcap", "beta", "volatility", "liquidity",
                      "value", "growth", "quality"]


def industry_exposure(weights: pl.DataFrame, exposure: pl.DataFrame,
                      weight_col: str = "weight",
                      industry_col: str = "industry") -> pl.DataFrame:
    """组合行业权重：sum(weight) by (date, industry)。"""
    df = weights.join(exposure.select(["date", "stock_id", industry_col]),
                      on=["date", "stock_id"], how="left")
    return (
        df.group_by(["date", industry_col], maintain_order=True)
        .agg(pl.col(weight_col).sum().alias("industry_weight"))
        .sort(["date", industry_col])
    )


def active_industry_exposure(weights: pl.DataFrame, benchmark: pl.DataFrame,
                             exposure: pl.DataFrame,
                             industry_col: str = "industry") -> pl.DataFrame:
    """主动行业暴露：组合行业权重 - 基准行业权重。"""
    port = industry_exposure(weights, exposure).rename({"industry_weight": "port_w"})
    bench = industry_exposure(
        benchmark.rename({"benchmark_weight": "weight"}), exposure
    ).rename({"industry_weight": "bench_w"})
    return (
        port.join(bench, on=["date", industry_col], how="full", coalesce=True)
        .fill_null(0.0)
        .with_columns((pl.col("port_w") - pl.col("bench_w")).alias("active_w"))
    )


def style_exposure(weights: pl.DataFrame, exposure: pl.DataFrame,
                   style_cols: list[str] | None = None,
                   weight_col: str = "weight") -> pl.DataFrame:
    """组合风格暴露：sum(weight * style) by date（暴露应已横截面标准化）。"""
    style_cols = [c for c in (style_cols or STYLE_COLS_DEFAULT)
                  if c in exposure.columns]
    df = weights.join(exposure.select(["date", "stock_id", *style_cols]),
                      on=["date", "stock_id"], how="left")
    return (
        df.group_by("date", maintain_order=True)
        .agg([
            (pl.col(c).cast(pl.Float64).fill_null(0.0) * pl.col(weight_col)).sum().alias(c)
            for c in style_cols
        ])
        .sort("date")
    )
