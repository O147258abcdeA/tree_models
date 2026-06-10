"""信号标准化处理模块（第十二节）。

处理顺序：raw -> winsorize -> zscore -> rank（另存）-> 行业中性 ->
风格中性（逐日横截面回归残差）-> 再 zscore = neutral_score。
全部按 date 分组处理，禁止跨日期统计、禁止使用未来数据。

输出 12 字段：date, stock_id, model_family, model_name, model_version,
label_type, horizon, objective_type, raw_score, rank_score, zscore_score,
neutral_score。
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.signals.signal_neutralizer import (industry_neutralize_signal,
                                            style_neutralize_signal)
from src.utils.io import write_partitioned
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_COLS = ["date", "stock_id", "model_family", "model_name",
                  "model_version", "label_type", "horizon", "objective_type",
                  "raw_score", "rank_score", "zscore_score", "neutral_score"]


def rank_signal(df: pl.DataFrame, score_col: str = "raw_score",
                out_col: str = "rank_score", date_col: str = "date") -> pl.DataFrame:
    """每日横截面 rank，归一到 [0, 1]。"""
    return df.with_columns(
        (
            (pl.col(score_col).rank("average").over(date_col) - 1)
            / (pl.col(score_col).count().over(date_col) - 1).clip(1, None)
        ).cast(pl.Float64).alias(out_col)
    )


def zscore_signal(df: pl.DataFrame, score_col: str,
                  out_col: str, date_col: str = "date") -> pl.DataFrame:
    """每日横截面 zscore：(x - mean) / std，std 为 0 时输出 0。"""
    mean = pl.col(score_col).mean().over(date_col)
    std = pl.col(score_col).std().over(date_col)
    return df.with_columns(
        pl.when(std.is_null() | (std == 0)).then(0.0)
        .otherwise((pl.col(score_col) - mean) / std)
        .cast(pl.Float64).alias(out_col)
    )


def winsorize_signal(df: pl.DataFrame, score_col: str,
                     lower: float = 0.01, upper: float = 0.99,
                     out_col: str | None = None,
                     date_col: str = "date") -> pl.DataFrame:
    """每日横截面去极值（分位数 clip）。"""
    out_col = out_col or score_col
    return df.with_columns(
        pl.col(score_col).clip(
            pl.col(score_col).quantile(lower, "linear").over(date_col),
            pl.col(score_col).quantile(upper, "linear").over(date_col),
        ).alias(out_col)
    )


def process_one_model_signal(signal_df: pl.DataFrame,
                             exposure_df: pl.DataFrame | None,
                             config: dict) -> pl.DataFrame:
    """处理单个模型的信号表：winsorize -> zscore -> rank -> 中性化 -> 再 zscore。

    Parameters
    ----------
    signal_df : 含 raw_score 的统一信号表（单个 model_name）。
    exposure_df : 风险暴露表 [date, stock_id, industry, style...]；
                  为 None 时跳过中性化（neutral_score = zscore_score）。
    """
    w = config.get("winsorize", {})
    industry_col = config.get("industry_col", "industry")
    style_cols = config.get("style_cols", ["log_mktcap", "beta", "turnover", "volatility"])

    df = signal_df.sort(["date", "stock_id"])
    work_col = "raw_score"
    if w.get("enabled", True):
        df = winsorize_signal(df, "raw_score", w.get("lower", 0.01),
                              w.get("upper", 0.99), out_col="_w")
        work_col = "_w"
    df = zscore_signal(df, work_col, "zscore_score")
    df = rank_signal(df, work_col, "rank_score")

    if exposure_df is not None:
        join_cols = ["date", "stock_id"]
        avail_styles = [c for c in style_cols if c in exposure_df.columns]
        keep = join_cols + ([industry_col] if industry_col in exposure_df.columns else []) + avail_styles
        df = df.join(exposure_df.select(keep), on=join_cols, how="left")
        score = "zscore_score"
        if config.get("industry_neutralize", True) and industry_col in df.columns:
            df = industry_neutralize_signal(df, score, industry_col, out_col="_n1")
            score = "_n1"
        if config.get("style_neutralize", True) and avail_styles:
            df = style_neutralize_signal(
                df, score, avail_styles,
                industry_col if industry_col in df.columns else None,
                out_col="_n2")
            score = "_n2"
        df = zscore_signal(df, score, "neutral_score")
    else:
        logger.warning("no exposure table: neutral_score falls back to zscore_score")
        df = df.with_columns(pl.col("zscore_score").alias("neutral_score"))

    return df.select(PROCESSED_COLS)


def batch_process_all_signals(signals_raw_root: str | Path,
                              signals_processed_root: str | Path,
                              exposure_df: pl.DataFrame | None,
                              config: dict) -> list[str]:
    """遍历信号库中全部 model_name，逐模型处理并写出 processed 信号库。"""
    raw_root = Path(signals_raw_root)
    model_names = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    if not model_names:
        raise ValueError(f"no model signals under {raw_root}")
    for name in model_names:
        from src.signals.signal_writer import read_signals
        raw = read_signals(raw_root, name)
        logger.info("processing signals: %s rows=%d", name, raw.height)
        out = process_one_model_signal(raw, exposure_df, config)
        write_partitioned(out, Path(signals_processed_root) / name)
    return model_names
