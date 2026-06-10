"""信号评估模块（第十三节）。

输入：
- signal 表：date, stock_id, model_*, label_type, horizon, objective_type,
  raw_score, rank_score, zscore_score, neutral_score
- return 表：date, stock_id, future_return_{h}d, benchmark_return, industry
- risk exposure 表：date, stock_id, industry, log_mktcap, beta, ...

输出（reports 目录）：
model_summary.csv / ic_timeseries.parquet / group_return.parquet /
long_short_return.parquet / exposure_report.parquet + 图表 png。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.utils.logger import get_logger
from src.utils.metrics import (add_group_col, annualize, daily_ic,
                               daily_rank_ic, group_returns, ic_summary,
                               long_short_returns, max_drawdown, top_turnover)

logger = get_logger(__name__)

STYLE_COLS_DEFAULT = ["log_mktcap", "beta", "turnover", "volatility",
                      "liquidity", "value", "growth", "quality"]


def evaluate_ic(df: pl.DataFrame, signal_col: str, return_col: str) -> tuple[pl.DataFrame, dict]:
    """每日横截面 pearson IC 时间序列 + 汇总（mean/std/ICIR/t/正比例）。"""
    ts = daily_ic(df, signal_col, return_col)
    return ts, ic_summary(ts)


def evaluate_rank_ic(df: pl.DataFrame, signal_col: str, return_col: str) -> tuple[pl.DataFrame, dict]:
    """每日横截面 spearman RankIC 时间序列 + 汇总。"""
    ts = daily_rank_ic(df, signal_col, return_col)
    summ = ic_summary(ts, "rank_ic")
    return ts, {f"rank_{k}": v for k, v in summ.items()}


def evaluate_group_return(df: pl.DataFrame, signal_col: str, return_col: str,
                          n_groups: int = 5) -> pl.DataFrame:
    """每日按 signal 分组（0=Bottom..n-1=Top）的平均未来收益。"""
    return group_returns(df, signal_col, return_col, n_groups)


def evaluate_long_short(grp: pl.DataFrame, n_groups: int = 5) -> tuple[pl.DataFrame, dict]:
    """Top - Bottom 多空收益时序 + 年化汇总（含多头组合收益）。"""
    ls = long_short_returns(grp, n_groups)
    stats = annualize(ls.get_column("long_short_ret"))
    long_stats = annualize(ls.get_column("long_ret"))
    return ls, {
        "ls_ann_return": stats["ann_return"],
        "ls_sharpe": stats["sharpe"],
        "ls_max_drawdown": stats["max_drawdown"],
        "long_ann_return": long_stats["ann_return"],
    }


def evaluate_turnover(df: pl.DataFrame, signal_col: str,
                      top_quantile: float = 0.2) -> tuple[pl.DataFrame, dict]:
    """Top 组换手率时序 + 均值。"""
    ts = top_turnover(df, signal_col, top_quantile)
    mean = ts.get_column("turnover").mean() if ts.height else None
    return ts, {"top_turnover_mean": mean}


def evaluate_exposure(df: pl.DataFrame, signal_col: str,
                      industry_col: str = "industry",
                      style_cols: list[str] | None = None,
                      top_quantile: float = 0.2) -> pl.DataFrame:
    """Top 组暴露：行业权重 - universe 行业权重；风格平均暴露（标准化口径）。

    返回长表 [exposure_type, name, exposure]。
    """
    style_cols = [c for c in (style_cols or STYLE_COLS_DEFAULT) if c in df.columns]
    n_groups = max(int(round(1.0 / top_quantile)), 2)
    g = add_group_col(df.drop_nulls([signal_col]), signal_col, n_groups)
    top = g.filter(pl.col("group") == n_groups - 1)
    rows = []
    if industry_col in df.columns:
        uni = (df.group_by(industry_col).agg(pl.len().alias("n"))
               .with_columns((pl.col("n") / pl.col("n").sum()).alias("uni_w")))
        tw = (top.group_by(industry_col).agg(pl.len().alias("n"))
              .with_columns((pl.col("n") / pl.col("n").sum()).alias("top_w")))
        ind = uni.join(tw, on=industry_col, how="full", coalesce=True).fill_null(0.0)
        for r in ind.iter_rows(named=True):
            rows.append({"exposure_type": "industry", "name": str(r[industry_col]),
                         "exposure": r["top_w"] - r["uni_w"]})
    for c in style_cols:
        rows.append({"exposure_type": "style", "name": c,
                     "exposure": top.get_column(c).cast(pl.Float64).mean()})
    return pl.DataFrame(rows, schema={"exposure_type": pl.Utf8, "name": pl.Utf8,
                                      "exposure": pl.Float64})


def evaluate_by_year(df: pl.DataFrame, signal_col: str, return_col: str,
                     n_groups: int = 5) -> pl.DataFrame:
    """分年度表现：每年 IC、RankIC、ICIR、多空收益、最大回撤。"""
    rows = []
    for (year,), part in (
        df.with_columns(pl.col("date").dt.year().alias("_y"))
        .partition_by("_y", as_dict=True).items()
    ):
        ic = ic_summary(daily_ic(part, signal_col, return_col))
        ric = ic_summary(daily_rank_ic(part, signal_col, return_col), "rank_ic")
        grp = group_returns(part, signal_col, return_col, n_groups)
        ls = long_short_returns(grp, n_groups)
        ls_ret = ls.get_column("long_short_ret")
        rows.append({
            "year": year,
            "ic_mean": ic["ic_mean"], "rankic_mean": ric["ic_mean"],
            "icir": ic["icir"],
            "ls_ann_return": ls_ret.mean() * 252 if ls_ret.len() else None,
            "ls_max_drawdown": max_drawdown(ls_ret),
        })
    return pl.DataFrame(rows).sort("year")


def evaluate_one_signal(df: pl.DataFrame, signal_col: str, return_col: str,
                        n_groups: int = 5, top_quantile: float = 0.2,
                        industry_col: str = "industry") -> dict:
    """单个 (signal_col, return_col) 的完整评估，返回各结果表与 summary。"""
    ic_ts, ic_stats = evaluate_ic(df, signal_col, return_col)
    ric_ts, ric_stats = evaluate_rank_ic(df, signal_col, return_col)
    grp = evaluate_group_return(df, signal_col, return_col, n_groups)
    ls, ls_stats = evaluate_long_short(grp, n_groups)
    to_ts, to_stats = evaluate_turnover(df, signal_col, top_quantile)
    exposure = evaluate_exposure(df, signal_col, industry_col,
                                 top_quantile=top_quantile)
    yearly = evaluate_by_year(df, signal_col, return_col, n_groups)
    summary = {**ic_stats, **ric_stats, **ls_stats, **to_stats}
    return {"ic_ts": ic_ts.join(ric_ts, on="date", how="inner"),
            "group_return": grp, "long_short": ls, "turnover": to_ts,
            "exposure": exposure, "yearly": yearly, "summary": summary}


def compare_models(summary_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """分模型比较：family / objective / label_type / horizon / score_col 五个维度。"""
    out = {}
    for dim in ["model_family", "objective_type", "label_type", "horizon", "score_col"]:
        if dim in summary_df.columns:
            out[dim] = (
                summary_df.group_by(dim)
                .agg(pl.col("ic_mean").mean(), pl.col("rank_ic_mean").mean(),
                     pl.col("icir").mean(), pl.col("ls_ann_return").mean())
                .sort(dim)
            )
    return out


def _plot_report(name: str, res: dict, out_dir: Path) -> None:
    """输出图表：IC/RankIC 时序、分组收益柱状图、多空净值曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ic = res["ic_ts"]
    axes[0, 0].plot(ic.get_column("date"), ic.get_column("ic").cum_sum(), lw=1)
    axes[0, 0].set_title(f"{name} cumulative IC")
    axes[0, 1].plot(ic.get_column("date"), ic.get_column("rank_ic").cum_sum(),
                    lw=1, color="darkorange")
    axes[0, 1].set_title("cumulative RankIC")
    grp_mean = (res["group_return"].group_by("group")
                .agg(pl.col("group_return").mean()).sort("group"))
    axes[1, 0].bar(grp_mean.get_column("group").cast(pl.Utf8),
                   grp_mean.get_column("group_return"))
    axes[1, 0].set_title("mean group return (0=Bottom)")
    ls = res["long_short"]
    nav = (1.0 + ls.get_column("long_short_ret")).cum_prod()
    axes[1, 1].plot(ls.get_column("date"), nav, lw=1, color="green")
    axes[1, 1].set_title("long-short NAV")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_report.png", dpi=120)
    plt.close(fig)


def generate_signal_report(signal_df: pl.DataFrame, return_df: pl.DataFrame,
                           config: dict, reports_root: str | Path,
                           exposure_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """对信号库全部 (model_name x score_col) 生成评估报告。

    Parameters
    ----------
    signal_df : processed 信号表（可含多个 model_name）。
    return_df : date, stock_id, future_return_{h}d[, industry] 收益表。
    config : signal_config.yaml 中 evaluation 段。

    Returns
    -------
    model_summary 汇总表（同时落盘全部 parquet / csv / png）。
    """
    out_dir = Path(reports_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_groups = config.get("n_groups", 5)
    top_q = config.get("top_quantile", 0.1)
    score_cols = config.get("score_cols", ["raw_score", "rank_score",
                                           "zscore_score", "neutral_score"])
    join_cols = ["date", "stock_id"]
    if exposure_df is not None:
        keep = [c for c in exposure_df.columns
                if c in join_cols + ["industry"] + STYLE_COLS_DEFAULT]
        signal_df = signal_df.join(exposure_df.select(keep), on=join_cols, how="left")

    summaries, ic_all, grp_all, ls_all, exp_all = [], [], [], [], []
    for (model_name,), part in signal_df.partition_by("model_name", as_dict=True).items():
        h = int(part.get_column("horizon")[0])
        ret_col = f"future_return_{h}d"
        if ret_col not in return_df.columns:
            raise ValueError(f"return table missing {ret_col}")
        df = part.join(return_df.select(join_cols + [ret_col] +
                                        (["industry"] if "industry" in return_df.columns
                                         and "industry" not in part.columns else [])),
                       on=join_cols, how="inner")
        for sc in score_cols:
            if sc not in df.columns:
                continue
            res = evaluate_one_signal(df, sc, ret_col, n_groups, top_q)
            tag = {"model_name": model_name, "score_col": sc,
                   "model_family": part.get_column("model_family")[0],
                   "objective_type": part.get_column("objective_type")[0],
                   "label_type": part.get_column("label_type")[0],
                   "horizon": h}
            summaries.append({**tag, **{
                "ic_mean": res["summary"].get("ic_mean"),
                "icir": res["summary"].get("icir"),
                "rank_ic_mean": res["summary"].get("rank_ic_mean"),
                "rank_icir": res["summary"].get("rank_icir"),
                "ls_ann_return": res["summary"].get("ls_ann_return"),
                "ls_sharpe": res["summary"].get("ls_sharpe"),
                "ls_max_drawdown": res["summary"].get("ls_max_drawdown"),
                "long_ann_return": res["summary"].get("long_ann_return"),
                "top_turnover_mean": res["summary"].get("top_turnover_mean"),
            }})
            lit = [pl.lit(model_name).alias("model_name"), pl.lit(sc).alias("score_col")]
            ic_all.append(res["ic_ts"].with_columns(lit))
            grp_all.append(res["group_return"].with_columns(lit))
            ls_all.append(res["long_short"].with_columns(lit))
            exp_all.append(res["exposure"].with_columns(lit))
            res["yearly"].with_columns(lit).write_parquet(
                out_dir / f"{model_name}_{sc}_yearly.parquet")
            _plot_report(f"{model_name}_{sc}", res, out_dir)

    summary_df = pl.DataFrame(summaries)
    summary_df.write_csv(out_dir / "model_summary.csv")
    pl.concat(ic_all).write_parquet(out_dir / "ic_timeseries.parquet")
    pl.concat(grp_all).write_parquet(out_dir / "group_return.parquet")
    pl.concat(ls_all).write_parquet(out_dir / "long_short_return.parquet")
    pl.concat(exp_all).write_parquet(out_dir / "exposure_report.parquet")
    for dim, table in compare_models(summary_df).items():
        table.write_csv(out_dir / f"compare_by_{dim}.csv")
    logger.info("signal report written to %s\n%s", out_dir, summary_df)
    return summary_df
