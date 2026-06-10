"""横截面预处理模块。

铁律：
- 所有统计量只来自当日横截面（group_by date 或在单日 DataFrame 内计算）；
- 禁止跨日期标准化；
- 特征处理不接触任何未来收益 / y_ 前缀列；
- 缺失率剔列在面板起点（train 窗口起点）统计并固化列表。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.data.load_data import get_factor, get_factor_cols, iter_dates
from src.utils.io import enforce_dtypes, write_partitioned
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 静态防护：特征处理禁止接触的列名前缀（防未来函数）
FORBIDDEN_FEATURE_PREFIXES = ("y_", "future_")


def assert_no_future_cols(factor_cols: list[str]) -> None:
    """静态断言：因子列中不允许出现标签 / 未来收益列。"""
    bad = [c for c in factor_cols if c.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if bad:
        raise ValueError(f"future-looking columns leaked into features: {bad}")


def filter_universe(df_day: pl.DataFrame, config: dict) -> pl.DataFrame:
    """单日 universe 过滤：剔除不可交易 / ST / 停牌 /（可选）涨跌停股票。"""
    f = config.get("filter", {})
    n0 = df_day.height
    rules = [
        ("drop_st", "is_st"),
        ("drop_suspended", "is_suspended"),
        ("drop_limit_up", "is_limit_up"),
        ("drop_limit_down", "is_limit_down"),
    ]
    for key, col in rules:
        if f.get(key, False) and col in df_day.columns:
            df_day = df_day.filter(
                pl.col(col).cast(pl.Float64).fill_null(0.0) == 0.0
            )
    logger.debug("universe filter: %d -> %d", n0, df_day.height)
    return df_day


def compute_missing_rate(df_day: pl.DataFrame, factor_cols: list[str]) -> dict[str, float]:
    """统计每个因子在当日横截面的缺失率。"""
    if df_day.is_empty():
        return {c: 1.0 for c in factor_cols}
    row = df_day.select(
        [pl.col(c).null_count().alias(c) for c in factor_cols]
    ).row(0)
    n = df_day.height
    return {c: cnt / n for c, cnt in zip(factor_cols, row)}


def select_valid_factors(df_day: pl.DataFrame, factor_cols: list[str],
                         max_missing_rate: float) -> list[str]:
    """删除缺失率过高的因子，返回保留的因子列表（在面板起点调用一次并固化）。"""
    rates = compute_missing_rate(df_day, factor_cols)
    keep = [c for c in factor_cols if rates[c] <= max_missing_rate]
    dropped = sorted(set(factor_cols) - set(keep))
    if dropped:
        logger.info("drop %d factors with missing rate > %.2f (e.g. %s)",
                    len(dropped), max_missing_rate, dropped[:5])
    return keep


def winsorize_cross_section(df_day: pl.DataFrame, factor_cols: list[str],
                            lower: float = 0.01, upper: float = 0.99) -> pl.DataFrame:
    """单日横截面去极值：按 [lower, upper] 分位数 clip。"""
    return df_day.with_columns([
        pl.col(c).clip(pl.col(c).quantile(lower, "linear"),
                       pl.col(c).quantile(upper, "linear"))
        for c in factor_cols
    ])


def fill_missing_cross_section(df_day: pl.DataFrame, factor_cols: list[str],
                               method: str = "median",
                               industry_col: str = "industry") -> pl.DataFrame:
    """单日横截面缺失值填充。

    method:
    - "median": 当日全市场中位数；
    - "industry_median": 当日行业中位数，行业全缺再退化为全市场中位数；
    - "zero": 填 0（适用于已标准化因子）。
    """
    if method == "median":
        return df_day.with_columns([
            pl.col(c).fill_null(pl.col(c).median()) for c in factor_cols
        ])
    if method == "industry_median":
        if industry_col not in df_day.columns:
            raise ValueError(f"industry column '{industry_col}' not found")
        return df_day.with_columns([
            pl.col(c)
            .fill_null(pl.col(c).median().over(industry_col))
            .fill_null(pl.col(c).median())
            for c in factor_cols
        ])
    if method == "zero":
        return df_day.with_columns([pl.col(c).fill_null(0.0) for c in factor_cols])
    raise ValueError(f"unknown fill method: {method}")


def standardize_cross_section(df_day: pl.DataFrame, factor_cols: list[str],
                              by_industry: bool = False,
                              industry_col: str = "industry") -> pl.DataFrame:
    """单日横截面 zscore 标准化；by_industry=True 时行业内标准化。

    std 为 0 或 null 时输出 0，避免产生 inf / nan。
    """
    def _z(c: str, over: str | None) -> pl.Expr:
        mean = pl.col(c).mean().over(over) if over else pl.col(c).mean()
        std = pl.col(c).std().over(over) if over else pl.col(c).std()
        return (
            pl.when(std.is_null() | (std == 0))
            .then(0.0)
            .otherwise((pl.col(c) - mean) / std)
            .cast(pl.Float32)
            .alias(c)
        )

    over = industry_col if by_industry else None
    if by_industry and industry_col not in df_day.columns:
        raise ValueError(f"industry column '{industry_col}' not found")
    return df_day.with_columns([_z(c, over) for c in factor_cols])


def build_design_matrix(df_day: pl.DataFrame, neutral_cols: list[str],
                        industry_col: str | None = None) -> np.ndarray:
    """构造单日横截面回归设计矩阵：截距 + 数值风险暴露 + 行业 dummy。"""
    parts = [np.ones((df_day.height, 1), dtype=np.float64)]
    if neutral_cols:
        x = df_day.select([
            pl.col(c).cast(pl.Float64).fill_null(pl.col(c).cast(pl.Float64).median())
            .fill_null(0.0)
            for c in neutral_cols
        ]).to_numpy()
        parts.append(x)
    if industry_col is not None:
        dummies = (
            df_day.select(pl.col(industry_col).cast(pl.Utf8).fill_null("UNKNOWN"))
            .to_dummies(industry_col, drop_first=True)
            .to_numpy()
            .astype(np.float64)
        )
        parts.append(dummies)
    return np.hstack(parts)


def regression_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS 残差（numpy lstsq）。y 中的 nan 行不参与拟合，残差对应位置为 nan。"""
    res = np.full_like(y, np.nan, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if mask.sum() <= x.shape[1]:
        return res  # 样本量不足，全部置 nan，不隐藏问题
    beta, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
    res[mask] = y[mask] - x[mask] @ beta
    return res


def neutralize_cross_section(df_day: pl.DataFrame, factor_cols: list[str],
                             neutral_cols: list[str],
                             industry_col: str | None = None) -> pl.DataFrame:
    """单日横截面中性化：每个因子对（风险暴露 + 行业 dummy）回归取残差。

    Parameters
    ----------
    neutral_cols : 数值风险暴露列，如 [log_mktcap, beta, turnover, volatility]。
    industry_col : 不为 None 时加入行业 dummy。
    """
    x = build_design_matrix(df_day, neutral_cols, industry_col)
    y_mat = df_day.select([pl.col(c).cast(pl.Float64) for c in factor_cols]).to_numpy()
    out = {}
    for j, c in enumerate(factor_cols):
        out[c] = regression_residual(y_mat[:, j], x).astype(np.float32)
    return df_day.with_columns([pl.Series(c, v, dtype=pl.Float32) for c, v in out.items()])


def preprocess_one_day(df_day: pl.DataFrame, config: dict,
                       factor_cols: list[str] | None = None) -> pl.DataFrame:
    """单日横截面预处理流水线。

    顺序：universe 过滤 -> 去极值 -> 缺失填充 -> 标准化（可选行业内）-> 可选中性化。
    factor_cols 应为面板起点固化的列表；None 时按正则现取。
    """
    if df_day.is_empty():
        return df_day
    if factor_cols is None:
        factor_cols = get_factor_cols(df_day)
    assert_no_future_cols(factor_cols)
    industry_col = config.get("industry_col", "industry")

    df_day = filter_universe(df_day, config)
    if df_day.is_empty():
        return df_day

    w = config.get("winsorize", {})
    df_day = winsorize_cross_section(df_day, factor_cols,
                                     w.get("lower", 0.01), w.get("upper", 0.99))
    df_day = fill_missing_cross_section(
        df_day, factor_cols,
        config.get("fill_missing", {}).get("method", "median"), industry_col)
    if config.get("standardize", True):
        df_day = standardize_cross_section(
            df_day, factor_cols,
            by_industry=config.get("standardize_by_industry", False),
            industry_col=industry_col)
    neu = config.get("neutralize", {})
    if neu.get("enabled", False):
        df_day = neutralize_cross_section(
            df_day, factor_cols, neu.get("neutral_cols", []),
            industry_col if neu.get("use_industry", True) else None)
    return enforce_dtypes(df_day, factor_cols)


def preprocess_panel(input_path: str | Path, output_path: str | Path,
                     config: dict,
                     start_date: str | dt.date | None = None,
                     end_date: str | dt.date | None = None) -> list[str]:
    """整段面板预处理：逐日 get_factor(date) -> preprocess_one_day -> 按年追加写出。

    全程内存只保留单日数据 + 单年缓冲。返回固化的因子列列表。
    """
    dates = iter_dates(input_path, start_date, end_date)
    if not dates:
        raise ValueError(f"no trading dates in [{start_date}, {end_date}] under {input_path}")
    logger.info("preprocess panel: %d days [%s .. %s]", len(dates), dates[0], dates[-1])

    # 在面板起点统计缺失率并固化因子列表
    first_day = get_factor(dates[0], input_path)
    all_factors = get_factor_cols(first_day)
    assert_no_future_cols(all_factors)
    factor_cols = select_valid_factors(
        filter_universe(first_day, config), all_factors,
        config.get("max_missing_rate", 0.3))
    logger.info("fixed factor list: %d / %d factors kept", len(factor_cols), len(all_factors))

    buffer: list[pl.DataFrame] = []
    buffer_year: int | None = None
    for d in dates:
        df_day = get_factor(d, input_path)
        out_day = preprocess_one_day(df_day, config, factor_cols)
        if out_day.is_empty():
            logger.warning("empty cross-section after preprocess on %s", d)
            continue
        if buffer_year is not None and d.year != buffer_year:
            write_partitioned(pl.concat(buffer, how="diagonal_relaxed"), output_path)
            buffer = []
        buffer_year = d.year
        buffer.append(out_day)
    if buffer:
        write_partitioned(pl.concat(buffer, how="diagonal_relaxed"), output_path)
    logger.info("preprocess panel done -> %s", output_path)
    return factor_cols
