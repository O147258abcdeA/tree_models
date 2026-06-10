"""量化评估指标（全 polars 实现，全部按 date 横截面计算，禁止跨日 pooled 统计）。

- daily_ic / daily_rank_ic: 每日横截面 pearson / spearman 相关
- ic_summary: IC 均值、std、ICIR、t 值、正比例
- group_returns: 每日分组收益
- long_short_returns: 多空收益
- top_turnover: Top 组换手率
- max_drawdown / annualize: 净值类指标
"""
from __future__ import annotations

import math

import polars as pl

TRADING_DAYS = 252


def daily_ic(df: pl.DataFrame, signal_col: str, return_col: str,
             date_col: str = "date") -> pl.DataFrame:
    """每日横截面 pearson IC。返回 [date, ic]。"""
    return (
        df.drop_nulls([signal_col, return_col])
        .group_by(date_col, maintain_order=True)
        .agg(pl.corr(signal_col, return_col).alias("ic"))
        .sort(date_col)
    )


def daily_rank_ic(df: pl.DataFrame, signal_col: str, return_col: str,
                  date_col: str = "date") -> pl.DataFrame:
    """每日横截面 spearman Rank IC（先日内 rank 再 pearson）。返回 [date, rank_ic]。"""
    return (
        df.drop_nulls([signal_col, return_col])
        .group_by(date_col, maintain_order=True)
        .agg(
            pl.corr(pl.col(signal_col).rank(), pl.col(return_col).rank())
            .alias("rank_ic")
        )
        .sort(date_col)
    )


def ic_summary(ic_df: pl.DataFrame, ic_col: str = "ic") -> dict:
    """IC 时间序列汇总：mean / std / ICIR / t 值 / 正比例。"""
    s = ic_df.get_column(ic_col).drop_nulls()
    n = s.len()
    if n == 0:
        return {"ic_mean": None, "ic_std": None, "icir": None,
                "ic_tstat": None, "ic_positive_ratio": None, "n_days": 0}
    mean, std = s.mean(), s.std()
    icir = mean / std if std and std > 0 else None
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_tstat": icir * math.sqrt(n) if icir is not None else None,
        "ic_positive_ratio": (s > 0).mean(),
        "n_days": n,
    }


def add_group_col(df: pl.DataFrame, signal_col: str, n_groups: int = 5,
                  date_col: str = "date", group_col: str = "group") -> pl.DataFrame:
    """每日按 signal 分位数分组（0=Bottom, n_groups-1=Top），日内独立分组。"""
    return df.with_columns(
        (
            (pl.col(signal_col).rank("ordinal").over(date_col) - 1)
            * n_groups
            // pl.col(signal_col).count().over(date_col)
        )
        .clip(0, n_groups - 1)
        .cast(pl.Int8)
        .alias(group_col)
    )


def group_returns(df: pl.DataFrame, signal_col: str, return_col: str,
                  n_groups: int = 5, date_col: str = "date") -> pl.DataFrame:
    """每日按 signal 分组的平均未来收益。返回 [date, group, group_return, n_stocks]。"""
    g = add_group_col(df.drop_nulls([signal_col, return_col]), signal_col,
                      n_groups, date_col)
    return (
        g.group_by([date_col, "group"], maintain_order=True)
        .agg(pl.col(return_col).mean().alias("group_return"),
             pl.len().alias("n_stocks"))
        .sort([date_col, "group"])
    )


def long_short_returns(grp: pl.DataFrame, n_groups: int = 5,
                       date_col: str = "date") -> pl.DataFrame:
    """Top 组 - Bottom 组的每日多空收益。输入为 group_returns 输出。"""
    top = grp.filter(pl.col("group") == n_groups - 1).select(date_col, pl.col("group_return").alias("long_ret"))
    bot = grp.filter(pl.col("group") == 0).select(date_col, pl.col("group_return").alias("short_ret"))
    return (
        top.join(bot, on=date_col, how="inner")
        .with_columns((pl.col("long_ret") - pl.col("short_ret")).alias("long_short_ret"))
        .sort(date_col)
    )


def top_turnover(df: pl.DataFrame, signal_col: str, top_quantile: float = 0.2,
                 date_col: str = "date", stock_col: str = "stock_id") -> pl.DataFrame:
    """Top 组持仓换手率：1 - |今日Top ∩ 上日Top| / |今日Top|。返回 [date, turnover]。"""
    n_groups = max(int(round(1.0 / top_quantile)), 2)
    g = add_group_col(df.drop_nulls([signal_col]), signal_col, n_groups, date_col)
    top = g.filter(pl.col("group") == n_groups - 1).select(date_col, stock_col)
    dates = top.get_column(date_col).unique().sort().to_list()
    rows = []
    prev: set | None = None
    for d in dates:
        cur = set(top.filter(pl.col(date_col) == d).get_column(stock_col).to_list())
        if prev is not None and cur:
            rows.append({"date": d, "turnover": 1.0 - len(cur & prev) / len(cur)})
        prev = cur
    return pl.DataFrame(rows, schema={"date": pl.Date, "turnover": pl.Float64})


def max_drawdown(returns: pl.Series) -> float:
    """单期收益序列的最大回撤（净值口径，返回正数）。"""
    nav = (1.0 + returns.fill_null(0.0)).cum_prod()
    dd = 1.0 - nav / nav.cum_max()
    return float(dd.max()) if dd.len() else 0.0


def annualize(returns: pl.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """年化收益 / 年化波动 / Sharpe / 最大回撤 / 胜率。"""
    r = returns.drop_nulls()
    n = r.len()
    if n == 0:
        return {"ann_return": None, "ann_vol": None, "sharpe": None,
                "max_drawdown": None, "win_rate": None}
    mean, std = r.mean(), r.std()
    ann_ret = mean * periods_per_year
    ann_vol = std * math.sqrt(periods_per_year) if std is not None else None
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol and ann_vol > 0 else None,
        "max_drawdown": max_drawdown(r),
        "win_rate": (r > 0).mean(),
    }
