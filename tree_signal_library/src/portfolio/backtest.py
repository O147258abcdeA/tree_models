"""回测模块：中证1000指数增强回测 + 多空策略回测。

时序纪律：
- 调仓日 t 只使用 t 日（含）之前已生成的 signal；
- 持仓在 t 日收盘建立，收益取 t+1 日（用 return 表的 next_return 对齐）；
- 全程不读取未来数据。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.portfolio.exposure import active_industry_exposure, style_exposure
from src.portfolio.optimizer import calculate_turnover, optimize_portfolio_one_day
from src.utils.logger import get_logger
from src.utils.metrics import annualize, max_drawdown

logger = get_logger(__name__)

TRADING_DAYS = 252


def _next_returns(return_df: pl.DataFrame, return_col: str = "return") -> pl.DataFrame:
    """t 日持仓对应 t+1 日收益：按 stock 把 return 前移一日对齐到 t。"""
    return (
        return_df.sort(["stock_id", "date"])
        .with_columns(pl.col(return_col).shift(-1).over("stock_id").alias("next_return"))
    )


# ---------------------------------------------------------------- 指数增强

def run_portfolio_backtest(alpha_df: pl.DataFrame, return_df: pl.DataFrame,
                           cfg: dict,
                           exposure_df: pl.DataFrame | None = None) -> dict:
    """中证1000指数增强回测。

    alpha_df : date, stock_id, alpha_score, benchmark_weight（+ 可选行业/风格列）。
    return_df : date, stock_id, return。

    Returns
    -------
    dict: holdings（每日持仓与主动权重）、daily（每日组合/基准/超额收益、
    换手、交易额、预期alpha）、summary（年化超额、TE、IR、最大回撤）、
    industry_exposure / style_exposure。
    """
    nxt = _next_returns(return_df)
    dates = alpha_df.get_column("date").unique().sort().to_list()
    prev_w: np.ndarray | None = None
    prev_ids: list[str] = []
    cost_rate = float(cfg.get("cost_rate", 0.0015))
    pv = float(cfg.get("constraints", {}).get("portfolio_value", 1e8))

    holdings, daily = [], []
    for d in dates:
        day = alpha_df.filter(pl.col("date") == d).sort("stock_id")
        # 把上期权重映射到今日股票集合（退市/换股票池的权重视为 0）
        prev_vec = None
        if prev_w is not None:
            m = dict(zip(prev_ids, prev_w))
            prev_vec = np.array([m.get(s, 0.0) for s in day.get_column("stock_id")])
        opt = optimize_portfolio_one_day(day, cfg, prev_vec)
        w = opt.get_column("weight").to_numpy()
        turnover = calculate_turnover(prev_vec, w)
        ret_day = opt.join(
            nxt.filter(pl.col("date") == d).select(["stock_id", "next_return"]),
            on="stock_id", how="left").with_columns(pl.col("next_return").fill_null(0.0))
        port_ret = float((ret_day.get_column("weight")
                          * ret_day.get_column("next_return")).sum())
        bench_ret = float((ret_day.get_column("benchmark_weight")
                           * ret_day.get_column("next_return")).sum())
        cost = turnover * cost_rate
        exp_alpha = float((day.get_column("alpha_score").fill_null(0.0).to_numpy()
                           * (w - opt.get_column("benchmark_weight").to_numpy())).sum())
        holdings.append(opt.with_columns(pl.lit(d).alias("date")))
        daily.append({"date": d, "portfolio_return": port_ret - cost,
                      "benchmark_return": bench_ret,
                      "excess_return": port_ret - cost - bench_ret,
                      "turnover": turnover, "traded_value": turnover * pv,
                      "expected_alpha": exp_alpha, "cost": cost})
        prev_w, prev_ids = w, day.get_column("stock_id").to_list()

    daily_df = pl.DataFrame(daily).sort("date")
    holdings_df = pl.concat(holdings)
    excess = daily_df.get_column("excess_return")
    stats = annualize(excess)
    te = float(excess.std() * np.sqrt(TRADING_DAYS)) if excess.len() > 1 else None
    summary = {
        "ann_excess_return": stats["ann_return"],
        "tracking_error": te,
        "information_ratio": (stats["ann_return"] / te) if te else None,
        "max_drawdown": max_drawdown(excess),
        "avg_turnover": daily_df.get_column("turnover").mean(),
    }
    result = {"holdings": holdings_df, "daily": daily_df, "summary": summary}
    if exposure_df is not None:
        bench = alpha_df.select(["date", "stock_id", "benchmark_weight"])
        result["industry_exposure"] = active_industry_exposure(
            holdings_df.select(["date", "stock_id", "weight"]), bench, exposure_df)
        result["style_exposure"] = style_exposure(
            holdings_df.select(["date", "stock_id", "weight"]), exposure_df)
    logger.info("index-enhanced backtest summary: %s", summary)
    return result


# ---------------------------------------------------------------- 多空策略

def build_long_short_weights(signal_df: pl.DataFrame, top_quantile: float = 0.1,
                             bottom_quantile: float = 0.1,
                             weighting: str = "equal",
                             signal_col: str = "signal_score") -> pl.DataFrame:
    """每日按 signal 排序：做多 Top q、做空 Bottom q；支持等权 / score 加权。

    返回 [date, stock_id, weight]（多头权重和 +1，空头权重和 -1）。
    """
    pct = (
        (pl.col(signal_col).rank("average").over("date") - 1)
        / (pl.col(signal_col).count().over("date") - 1).clip(1, None)
    )
    df = signal_df.drop_nulls([signal_col]).with_columns(pct.alias("_pct"))
    longs = df.filter(pl.col("_pct") >= 1.0 - top_quantile)
    shorts = df.filter(pl.col("_pct") <= bottom_quantile)
    if weighting == "equal":
        longs = longs.with_columns((1.0 / pl.len().over("date")).alias("weight"))
        shorts = shorts.with_columns((-1.0 / pl.len().over("date")).alias("weight"))
    elif weighting == "score":
        lz = (pl.col(signal_col) - pl.col(signal_col).min().over("date") + 1e-12)
        longs = longs.with_columns((lz / lz.sum().over("date")).alias("weight"))
        sz = (pl.col(signal_col).max().over("date") - pl.col(signal_col) + 1e-12)
        shorts = shorts.with_columns((-sz / sz.sum().over("date")).alias("weight"))
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    return pl.concat([longs, shorts]).select(["date", "stock_id", "weight"]).sort(
        ["date", "stock_id"])


def neutralize_long_short_weights(weights: pl.DataFrame, exposure_df: pl.DataFrame,
                                  industry_neutral: bool = False,
                                  mktcap_neutral: bool = False,
                                  beta_neutral: bool = False) -> pl.DataFrame:
    """多空权重中性化（近似法，逐日横截面）：

    - 行业中性：行业内多空权重分别重归一，使每行业净暴露为 0；
    - 市值 / Beta 中性：去除权重与暴露的横截面相关（w -= proj），再重归一。
    """
    keep = ["date", "stock_id"] + [c for c in ("industry", "log_mktcap", "beta")
                                   if c in exposure_df.columns]
    df = weights.join(exposure_df.select(keep), on=["date", "stock_id"], how="left")
    if industry_neutral and "industry" in df.columns:
        long_w = pl.when(pl.col("weight") > 0).then(pl.col("weight")).otherwise(0.0)
        short_w = pl.when(pl.col("weight") < 0).then(-pl.col("weight")).otherwise(0.0)
        df = df.with_columns(
            (long_w / long_w.sum().over(["date", "industry"]).clip(1e-12, None)
             - short_w / short_w.sum().over(["date", "industry"]).clip(1e-12, None))
            .alias("weight")
        ).with_columns(  # 行业等权聚合后整体重归一
            (pl.col("weight")
             / pl.when(pl.col("weight") > 0).then(pl.col("weight")).otherwise(0.0)
             .sum().over("date").clip(1e-12, None)).alias("weight"))
    for flag, col in ((mktcap_neutral, "log_mktcap"), (beta_neutral, "beta")):
        if flag and col in df.columns:
            x = pl.col(col).cast(pl.Float64).fill_null(0.0)
            xc = x - x.mean().over("date")
            proj = ((pl.col("weight") * xc).sum().over("date")
                    / (xc * xc).sum().over("date").clip(1e-12, None))
            df = df.with_columns((pl.col("weight") - proj * xc).alias("weight"))
    return df.select(["date", "stock_id", "weight"])


def calculate_long_short_return(weights: pl.DataFrame,
                                return_df: pl.DataFrame) -> pl.DataFrame:
    """每日多空组合收益（持仓 t 日建立，取 t+1 日收益）。

    返回 [date, long_ret, short_ret, ls_ret]。
    """
    nxt = _next_returns(return_df)
    df = weights.join(nxt.select(["date", "stock_id", "next_return"]),
                      on=["date", "stock_id"], how="left").with_columns(
        pl.col("next_return").fill_null(0.0))
    return (
        df.group_by("date", maintain_order=True)
        .agg(
            (pl.when(pl.col("weight") > 0).then(pl.col("weight") * pl.col("next_return"))
             .otherwise(0.0)).sum().alias("long_ret"),
            (pl.when(pl.col("weight") < 0).then(pl.col("weight") * pl.col("next_return"))
             .otherwise(0.0)).sum().alias("short_ret"),
        )
        .with_columns((pl.col("long_ret") + pl.col("short_ret")).alias("ls_ret"))
        .sort("date")
    )


def calculate_turnover_ts(weights: pl.DataFrame) -> pl.DataFrame:
    """每日双边换手率：sum(|w_t - w_{t-1}|)（按 stock 对齐，缺失视为 0）。"""
    w = weights.sort(["stock_id", "date"]).with_columns(
        pl.col("weight").shift(1).over("stock_id").alias("_prev"))
    return (
        w.group_by("date", maintain_order=True)
        .agg((pl.col("weight").fill_null(0.0) - pl.col("_prev").fill_null(0.0))
             .abs().sum().alias("turnover"))
        .sort("date")
    )


def _resample_rebalance(signal_df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """按调仓频率取调仓日信号并 forward-fill 持仓（weekly=每周一个调仓日等）。"""
    if freq == "daily":
        return signal_df
    dates = signal_df.select("date").unique().sort("date")
    if freq == "weekly":
        dates = dates.with_columns(pl.col("date").dt.week().alias("_p"),
                                   pl.col("date").dt.year().alias("_y"))
    elif freq == "monthly":
        dates = dates.with_columns(pl.col("date").dt.month().alias("_p"),
                                   pl.col("date").dt.year().alias("_y"))
    else:
        raise ValueError(f"unknown rebalance freq: {freq}")
    rebal = dates.group_by(["_y", "_p"]).agg(pl.col("date").min()).get_column("date")
    return signal_df.filter(pl.col("date").is_in(rebal.implode()))


def run_long_short_backtest(signal_df: pl.DataFrame, return_df: pl.DataFrame,
                            cfg: dict,
                            exposure_df: pl.DataFrame | None = None) -> dict:
    """多空策略回测主入口（检验信号纯度）。

    signal_df : date, stock_id, signal_score（每个调仓日只用当日已产生的 signal）。
    cfg : portfolio_config.yaml 中 long_short 段。
    """
    ls_cfg = cfg.get("long_short", cfg)
    rebal = _resample_rebalance(signal_df, ls_cfg.get("rebalance_freq", "daily"))
    weights = build_long_short_weights(
        rebal, ls_cfg.get("top_quantile", 0.1), ls_cfg.get("bottom_quantile", 0.1),
        ls_cfg.get("weighting", "equal"))
    if exposure_df is not None and (ls_cfg.get("industry_neutral") or
                                    ls_cfg.get("mktcap_neutral") or
                                    ls_cfg.get("beta_neutral")):
        weights = neutralize_long_short_weights(
            weights, exposure_df,
            ls_cfg.get("industry_neutral", False),
            ls_cfg.get("mktcap_neutral", False),
            ls_cfg.get("beta_neutral", False))
    rets = calculate_long_short_return(weights, return_df)
    to = calculate_turnover_ts(weights)
    daily = rets.join(to, on="date", how="left").with_columns(
        pl.col("turnover").fill_null(0.0),
        (pl.col("ls_ret") - pl.col("turnover").fill_null(0.0)
         * float(ls_cfg.get("cost_rate", 0.0015))).alias("ls_ret_net"),
    )
    perf = evaluate_long_short_performance(daily)
    result = {"weights": weights, "daily": daily, "summary": perf}
    if exposure_df is not None:
        result["style_exposure"] = style_exposure(weights, exposure_df)
    logger.info("long-short backtest summary: %s", perf)
    return result


def evaluate_long_short_performance(daily: pl.DataFrame) -> dict:
    """多空绩效：年化收益/波动、Sharpe、最大回撤、胜率、月度收益、
    多头/空头收益、换手、成本后收益。"""
    gross = annualize(daily.get_column("ls_ret"))
    net = annualize(daily.get_column("ls_ret_net"))
    monthly = (
        daily.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month").agg(pl.col("ls_ret").sum().alias("monthly_return"))
        .sort("month")
    )
    return {
        "ann_return": gross["ann_return"],
        "ann_vol": gross["ann_vol"],
        "sharpe": gross["sharpe"],
        "max_drawdown": gross["max_drawdown"],
        "win_rate": gross["win_rate"],
        "ann_return_net": net["ann_return"],
        "sharpe_net": net["sharpe"],
        "long_ann_return": daily.get_column("long_ret").mean() * TRADING_DAYS,
        "short_ann_return": daily.get_column("short_ret").mean() * TRADING_DAYS,
        "avg_turnover": daily.get_column("turnover").mean(),
        "monthly_returns": monthly,
    }


def save_backtest_report(result: dict, out_dir: str | Path, name: str) -> None:
    """落盘回测结果表与净值图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily = result["daily"]
    daily.write_parquet(out / f"{name}_daily.parquet")
    ret_col = "excess_return" if "excess_return" in daily.columns else "ls_ret_net"
    nav = (1.0 + daily.get_column(ret_col)).cum_prod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily.get_column("date"), nav, lw=1)
    ax.set_title(f"{name} NAV ({ret_col})")
    fig.tight_layout()
    fig.savefig(out / f"{name}_nav.png", dpi=120)
    plt.close(fig)
