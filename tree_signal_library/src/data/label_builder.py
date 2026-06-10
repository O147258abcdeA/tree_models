"""多标签构造模块。

防未来函数原则：
- 标签使用 t+1..t+h 的未来收益，但行索引在 t（特征日）；
- 序列末尾不足 h 天的标签置 null；
- 标签的去极值 / rank / 分组 / 标准化全部按 date 横截面；
- 特征处理代码（preprocess.py）物理上接触不到任何 y_ 前缀列。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.data.preprocess import build_design_matrix, regression_residual
from src.utils.io import write_partitioned
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_TYPE_COL = {
    "raw_return": "y_raw_{h}",
    "excess_return": "y_excess_{h}",
    "industry_neutral_return": "y_ind_neutral_{h}",
    "residual_return": "y_residual_{h}",
    "rank_return": "y_rank_{h}",
    "group_label": "y_group_{h}",
}


def label_col(label_type: str, horizon: int) -> str:
    """label_type + horizon -> 标签列名。"""
    if label_type not in LABEL_TYPE_COL:
        raise ValueError(f"unknown label_type: {label_type}")
    return LABEL_TYPE_COL[label_type].format(h=horizon)


def build_forward_return(df: pl.DataFrame, horizon: int,
                         price_col: str = "close",
                         date_col: str = "date",
                         stock_col: str = "stock_id") -> pl.DataFrame:
    """未来 h 日原始收益：y_raw_h = close_{t+h} / close_t - 1，对齐到 t 日。

    按 stock_id 分组、date 排序后 shift(-h)；股票序列末尾不足 h 天自动为 null。
    注意：基于股票自身交易日序列，停牌缺行的跨度会大于 h 个日历交易日，
    入库前已剔除停牌股票，影响可控。
    """
    out = f"y_raw_{horizon}"
    return (
        df.sort([stock_col, date_col])
        .with_columns(
            (pl.col(price_col).shift(-horizon).over(stock_col) / pl.col(price_col) - 1.0)
            .cast(pl.Float32)
            .alias(out)
        )
    )


def build_benchmark_return(df: pl.DataFrame, horizon: int,
                           return_col: str = "return",
                           weight_col: str = "benchmark_weight",
                           date_col: str = "date") -> pl.DataFrame:
    """基准未来 h 日收益（按 benchmark_weight 加权的当日收益复合）。

    返回 [date, bench_fwd_{h}]，对齐到 t 日（t+1..t+h 复合收益）。
    """
    daily = (
        df.filter(pl.col(weight_col).fill_null(0.0) > 0)
        .group_by(date_col, maintain_order=True)
        .agg(
            (
                (pl.col(return_col) * pl.col(weight_col)).sum()
                / pl.col(weight_col).sum()
            ).alias("bench_ret")
        )
        .sort(date_col)
    )
    out = f"bench_fwd_{horizon}"
    # t 日的未来 h 日收益 = (1+r_{t+1})*...*(1+r_{t+h}) - 1
    daily = daily.with_columns(
        (1.0 + pl.col("bench_ret")).log().alias("_lr")
    ).with_columns(
        (
            pl.col("_lr").cum_sum().shift(-horizon) - pl.col("_lr").cum_sum()
        ).exp().sub(1.0).cast(pl.Float32).alias(out)
    )
    return daily.select(date_col, out)


def build_excess_return(df: pl.DataFrame, horizon: int,
                        return_col: str = "return",
                        weight_col: str = "benchmark_weight",
                        date_col: str = "date") -> pl.DataFrame:
    """未来 h 日相对基准（中证1000）超额收益：y_excess_h = y_raw_h - bench_fwd_h。

    要求 df 已含 y_raw_{h} 列（先调用 build_forward_return）。
    """
    raw = f"y_raw_{horizon}"
    if raw not in df.columns:
        raise ValueError(f"{raw} missing; call build_forward_return first")
    bench = build_benchmark_return(df, horizon, return_col, weight_col, date_col)
    return (
        df.join(bench, on=date_col, how="left")
        .with_columns(
            (pl.col(raw) - pl.col(f"bench_fwd_{horizon}"))
            .cast(pl.Float32)
            .alias(f"y_excess_{horizon}")
        )
        .drop(f"bench_fwd_{horizon}")
    )


def build_industry_neutral_label(df: pl.DataFrame, horizon: int,
                                 industry_col: str = "industry",
                                 date_col: str = "date") -> pl.DataFrame:
    """未来 h 日行业中性收益：y_ind_neutral_h = y_raw_h - 当日行业中位数。"""
    raw = f"y_raw_{horizon}"
    if raw not in df.columns:
        raise ValueError(f"{raw} missing; call build_forward_return first")
    return df.with_columns(
        (pl.col(raw) - pl.col(raw).median().over([date_col, industry_col]))
        .cast(pl.Float32)
        .alias(f"y_ind_neutral_{horizon}")
    )


def build_residual_label(df: pl.DataFrame, horizon: int,
                         style_cols: list[str],
                         industry_col: str = "industry",
                         date_col: str = "date") -> pl.DataFrame:
    """未来 h 日风格残差收益：每日横截面对风格暴露 + 行业 dummy 回归取残差。

    future_return_i = a + b'style_i + industry_dummies + e_i, y_residual_h = e_i
    """
    raw = f"y_raw_{horizon}"
    if raw not in df.columns:
        raise ValueError(f"{raw} missing; call build_forward_return first")
    out = f"y_residual_{horizon}"

    def _one_day(day: pl.DataFrame) -> pl.DataFrame:
        x = build_design_matrix(day, style_cols,
                                industry_col if industry_col in day.columns else None)
        y = day.get_column(raw).cast(pl.Float64).to_numpy()
        res = regression_residual(y, x)
        return day.with_columns(pl.Series(out, res, dtype=pl.Float32))

    return (
        df.sort(date_col)
        .group_by(date_col, maintain_order=True)
        .map_groups(_one_day)
    )


def build_rank_label(df: pl.DataFrame, horizon: int,
                     date_col: str = "date") -> pl.DataFrame:
    """未来 h 日横截面 rank 标签，归一到 [0, 1]：rank(y_raw_h) within date。"""
    raw = f"y_raw_{horizon}"
    return df.with_columns(
        (
            (pl.col(raw).rank("average").over(date_col) - 1)
            / (pl.col(raw).count().over(date_col) - 1).clip(1, None)
        )
        .cast(pl.Float32)
        .alias(f"y_rank_{horizon}")
    )


def build_group_label(df: pl.DataFrame, horizon: int, n_groups: int = 5,
                      date_col: str = "date") -> pl.DataFrame:
    """未来 h 日分组标签：每日横截面按未来收益分位数分成 n_groups 组（0..n_groups-1）。

    同时作为 ranking 模型的整数 relevance。
    """
    raw = f"y_raw_{horizon}"
    return df.with_columns(
        pl.when(pl.col(raw).is_null())
        .then(None)
        .otherwise(
            (
                (pl.col(raw).rank("ordinal").over(date_col) - 1)
                * n_groups
                // pl.col(raw).count().over(date_col)
            ).clip(0, n_groups - 1)
        )
        .cast(pl.Int8)
        .alias(f"y_group_{horizon}")
    )


def build_classification_label(df: pl.DataFrame, horizon: int,
                               top_quantile: float = 0.2,
                               bottom_quantile: float = 0.2,
                               keep_middle: bool = True,
                               date_col: str = "date") -> pl.DataFrame:
    """top-bottom 分类标签：当日横截面 top q -> 1，bottom q -> 0。

    keep_middle=True 时中间样本标记为中性类别 2；否则置 null（训练时丢弃）。
    """
    raw = f"y_raw_{horizon}"
    pct = (
        (pl.col(raw).rank("average").over(date_col) - 1)
        / (pl.col(raw).count().over(date_col) - 1).clip(1, None)
    )
    middle = pl.lit(2, dtype=pl.Int8) if keep_middle else pl.lit(None, dtype=pl.Int8)
    return df.with_columns(
        pl.when(pl.col(raw).is_null()).then(None)
        .when(pct >= 1.0 - top_quantile).then(1)
        .when(pct <= bottom_quantile).then(0)
        .otherwise(middle)
        .cast(pl.Int8)
        .alias(f"y_class_{horizon}")
    )


def build_ranking_group(df: pl.DataFrame, date_col: str = "date") -> np.ndarray:
    """构造 ranking query group：每个 date 内的股票构成一个 group。

    要求 df 已按 date 排序（强制断言），返回按 date 顺序的 group sizes。
    禁止把不同日期的股票混进同一个 group。
    """
    dates = df.get_column(date_col)
    if not dates.is_sorted():
        raise ValueError("df must be sorted by date before building ranking groups")
    sizes = (
        df.group_by(date_col, maintain_order=True).agg(pl.len().alias("n"))
        .get_column("n").to_numpy()
    )
    assert sizes.sum() == df.height
    return sizes


def winsorize_label(df: pl.DataFrame, cols: list[str], lower: float, upper: float,
                    date_col: str = "date") -> pl.DataFrame:
    """标签按 date 横截面去极值（分位数 clip）。"""
    return df.with_columns([
        pl.col(c).clip(
            pl.col(c).quantile(lower, "linear").over(date_col),
            pl.col(c).quantile(upper, "linear").over(date_col),
        ).alias(c)
        for c in cols
    ])


def standardize_label(df: pl.DataFrame, cols: list[str],
                      date_col: str = "date") -> pl.DataFrame:
    """标签按 date 横截面 zscore 标准化。"""
    return df.with_columns([
        ((pl.col(c) - pl.col(c).mean().over(date_col))
         / pl.col(c).std().over(date_col)).cast(pl.Float32).alias(c)
        for c in cols
    ])


def label_quality_report(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """输出每个标签的覆盖率、均值、标准差、极端值比例（|z|>3 的占比）。"""
    rows = []
    n = df.height
    for c in cols:
        s = df.get_column(c).cast(pl.Float64)
        valid = s.drop_nulls()
        mean = valid.mean() if valid.len() else None
        std = valid.std() if valid.len() else None
        extreme = None
        if std and std > 0:
            extreme = ((valid - mean).abs() > 3 * std).mean()
        rows.append({
            "label": c,
            "coverage": valid.len() / n if n else 0.0,
            "mean": mean,
            "std": std,
            "extreme_ratio": extreme,
        })
    return pl.DataFrame(rows)


def build_all_labels(df: pl.DataFrame, config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造全部 horizon 的全部标签，返回 (标签面板, 质量报告)。

    标签面板字段：date, stock_id 以及各 y_*_{h}。
    """
    horizons = config.get("horizons", [1, 5, 10, 20])
    style_cols = config.get("style_cols", ["log_mktcap", "beta", "turnover", "volatility"])
    industry_col = config.get("industry_col", "industry")
    n_groups = config.get("n_groups", 5)
    cls = config.get("classification", {})
    w = config.get("winsorize", {})

    out = df
    cont_cols: list[str] = []
    for h in horizons:
        logger.info("building labels for horizon=%d", h)
        out = build_forward_return(out, h, config.get("price_col", "close"))
        out = build_excess_return(out, h,
                                  weight_col=config.get("benchmark_weight_col", "benchmark_weight"))
        out = build_industry_neutral_label(out, h, industry_col)
        out = build_residual_label(out, h, style_cols, industry_col)
        cont_cols += [f"y_raw_{h}", f"y_excess_{h}", f"y_ind_neutral_{h}", f"y_residual_{h}"]
        # 去极值需在 rank / 分组 / 分类之前，不影响序关系，但保持口径一致
        if w.get("enabled", True):
            out = winsorize_label(out, [f"y_raw_{h}", f"y_excess_{h}",
                                        f"y_ind_neutral_{h}", f"y_residual_{h}"],
                                  w.get("lower", 0.01), w.get("upper", 0.99))
        out = build_rank_label(out, h)
        out = build_group_label(out, h, n_groups)
        out = build_classification_label(out, h,
                                         cls.get("top_quantile", 0.2),
                                         cls.get("bottom_quantile", 0.2),
                                         cls.get("keep_middle", True))
    if config.get("standardize", False):
        out = standardize_label(out, cont_cols)

    label_cols = [c for c in out.columns if c.startswith("y_")]
    labels = out.select(["date", "stock_id", *label_cols]).sort(["date", "stock_id"])
    report = label_quality_report(labels, label_cols)
    logger.info("label quality report:\n%s", report)
    return labels, report


def build_labels_to_parquet(input_path: str | Path, output_path: str | Path,
                            config: dict,
                            start_date: str | dt.date | None = None,
                            end_date: str | dt.date | None = None) -> pl.DataFrame:
    """从原始面板读取必要列，构造标签并按年分区保存为 parquet。返回质量报告。

    标签构造需要跨日 shift，因此读取整段必要列（仅约 10 列而非上千因子列，
    内存可控），不读取任何 factor_ 列。
    """
    from src.utils.io import scan_parquet_dataset

    need = ["date", "stock_id", config.get("price_col", "close"), "return",
            config.get("benchmark_weight_col", "benchmark_weight"),
            config.get("industry_col", "industry"),
            *config.get("style_cols", [])]
    lf = scan_parquet_dataset(input_path, start_date, end_date)
    avail = [c for c in dict.fromkeys(need) if c in lf.collect_schema().names()]
    missing = set(need) - set(avail)
    if missing:
        raise ValueError(f"label inputs missing columns: {sorted(missing)}")
    df = lf.select(avail).collect().sort(["date", "stock_id"])
    labels, report = build_all_labels(df, config)
    write_partitioned(labels, output_path)
    report.write_csv(Path(output_path) / "label_quality_report.csv")
    return report
