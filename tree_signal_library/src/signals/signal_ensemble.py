"""多模型信号融合模块（第十四节）。

融合方式：
1. equal_rank        : 各模型 rank_score 等权平均
2. equal_neutral     : 各模型 neutral_score 等权平均
3. icir_weighted     : 滚动 ICIR 加权（max(ICIR,0) 归一化）
4. rankicir_weighted : 滚动 RankICIR 加权
5. family            : 家族内等权 -> 家族间加权
6. stacking          : 二层模型（Ridge / ElasticNet / LightGBM）滚动训练

时序纪律（硬约束）：
- 日期 t 的动态权重只使用 t 之前已实现的 IC（IC_d 需要 d+h 日收益，
  仅当 d <= t - h - 1 时已实现，故统一右移 horizon+1 天）；
- 权重历史表记录 info_as_of_date，断言 info_as_of_date < apply_date；
- 不允许使用未来 ICIR。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.utils.logger import get_logger
from src.utils.metrics import daily_rank_ic, daily_ic

logger = get_logger(__name__)

METHODS = ("equal_rank", "equal_neutral", "icir_weighted",
           "rankicir_weighted", "family", "stacking")


def _pivot_scores(signals: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """长表 -> 宽表：每个 model_name 一列 score。"""
    return signals.pivot(values=score_col, index=["date", "stock_id"],
                         on="model_name").sort(["date", "stock_id"])


def equal_weight_ensemble(signals: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """等权融合：ensemble_score = mean(score_m)。返回 [date, stock_id, ensemble_score]。"""
    wide = _pivot_scores(signals, score_col)
    model_cols = [c for c in wide.columns if c not in ("date", "stock_id")]
    return wide.with_columns(
        pl.mean_horizontal(model_cols).alias("ensemble_score")
    ).select(["date", "stock_id", "ensemble_score"])


def compute_rolling_icir(signals: pl.DataFrame, return_df: pl.DataFrame,
                         horizon: int, lookback_days: int = 252,
                         score_col: str = "neutral_score",
                         use_rank: bool = True) -> pl.DataFrame:
    """逐模型计算「截至每个日期可用」的滚动 ICIR。

    IC_d 依赖 d+h 收益，需右移 horizon+1 天后再做滚动均值/标准差，
    保证日期 t 的 ICIR 只含已实现信息。返回 [date, model_name, icir,
    info_as_of_date]。
    """
    ret_col = f"future_return_{horizon}d"
    if ret_col not in return_df.columns:
        raise ValueError(f"return table missing {ret_col}")
    frames = []
    for (model_name,), part in signals.partition_by("model_name", as_dict=True).items():
        df = part.join(return_df.select(["date", "stock_id", ret_col]),
                       on=["date", "stock_id"], how="inner")
        ic_fn = daily_rank_ic if use_rank else daily_ic
        ic_ts = ic_fn(df, score_col, ret_col).rename(
            {"rank_ic" if use_rank else "ic": "ic"})
        ic_ts = ic_ts.sort("date").with_columns(
            pl.col("ic").shift(horizon + 1).alias("_lag_ic"),
            pl.col("date").shift(horizon + 1).alias("info_as_of_date"),
        ).with_columns(
            (
                pl.col("_lag_ic").rolling_mean(lookback_days, min_samples=2)
                / pl.col("_lag_ic").rolling_std(lookback_days, min_samples=2)
            ).alias("icir"),
            pl.lit(model_name).alias("model_name"),
        )
        frames.append(ic_ts.select(["date", "model_name", "icir", "info_as_of_date"]))
    out = pl.concat(frames)
    # 时序审计：权重信息日期必须严格早于应用日期
    bad = out.drop_nulls("info_as_of_date").filter(
        pl.col("info_as_of_date") >= pl.col("date"))
    if bad.height:
        raise AssertionError("future information leaked into ensemble weights")
    return out


def icir_weighted_ensemble(signals: pl.DataFrame, return_df: pl.DataFrame,
                           horizon: int, lookback_days: int = 252,
                           min_history_days: int = 60,
                           score_col: str = "neutral_score",
                           use_rank: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """ICIR（或 RankICIR）加权融合：weight_m = max(ICIR_m,0)/sum(max(ICIR_m,0))。

    历史不足 min_history_days 或全部 ICIR<=0 时退化为等权。
    返回 (ensemble 信号, 权重历史表)。
    """
    icir = compute_rolling_icir(signals, return_df, horizon, lookback_days,
                                score_col, use_rank)
    weights = icir.with_columns(
        pl.col("icir").clip(0.0, None).fill_null(0.0).alias("_w")
    ).with_columns(
        pl.when(pl.col("_w").sum().over("date") > 0)
        .then(pl.col("_w") / pl.col("_w").sum().over("date"))
        .otherwise(1.0 / pl.col("_w").count().over("date"))
        .alias("weight")
    )
    # 历史不足时退化为等权
    day_rank = weights.select("date").unique().sort("date").with_row_index("_dayno")
    weights = weights.join(day_rank, on="date").with_columns(
        pl.when(pl.col("_dayno") < min_history_days)
        .then(1.0 / pl.col("weight").count().over("date"))
        .otherwise(pl.col("weight"))
        .alias("weight")
    ).select(["date", "model_name", "icir", "info_as_of_date", "weight"])

    merged = signals.join(weights.select(["date", "model_name", "weight"]),
                          on=["date", "model_name"], how="inner")
    ens = (
        merged.group_by(["date", "stock_id"], maintain_order=True)
        .agg(
            (pl.col(score_col) * pl.col("weight")).sum().alias("_num"),
            pl.col("weight").sum().alias("_den"),
        )
        .with_columns((pl.col("_num") / pl.col("_den")).alias("ensemble_score"))
        .select(["date", "stock_id", "ensemble_score"])
        .sort(["date", "stock_id"])
    )
    return ens, weights


def family_ensemble(signals: pl.DataFrame, score_col: str = "neutral_score",
                    family_weights: dict[str, float] | None = None) -> pl.DataFrame:
    """家族融合：家族内等权 -> 家族间按 family_weights 加权。"""
    fam = (
        signals.group_by(["date", "stock_id", "model_family"], maintain_order=True)
        .agg(pl.col(score_col).mean().alias("fam_score"))
    )
    if family_weights:
        wmap = pl.DataFrame({"model_family": list(family_weights),
                             "fam_w": list(family_weights.values())})
        fam = fam.join(wmap, on="model_family", how="left").with_columns(
            pl.col("fam_w").fill_null(0.0))
    else:
        fam = fam.with_columns(pl.lit(1.0).alias("fam_w"))
    return (
        fam.group_by(["date", "stock_id"], maintain_order=True)
        .agg(((pl.col("fam_score") * pl.col("fam_w")).sum()
              / pl.col("fam_w").sum()).alias("ensemble_score"))
        .sort(["date", "stock_id"])
    )


def stacking_ensemble(signals: pl.DataFrame, return_df: pl.DataFrame,
                      horizon: int, lookback_days: int = 252,
                      refit_freq_days: int = 21, alpha: float = 1.0,
                      score_col: str = "neutral_score") -> pl.DataFrame:
    """Stacking：用过去窗口内各模型 score 为特征、已实现未来收益为标签，
    滚动训练 Ridge 二层模型，对当前日期输出 ensemble_score。

    训练样本只取 date <= t - horizon - 1（标签已实现），每 refit_freq_days
    重新拟合一次；历史不足时退化为等权。
    """
    ret_col = f"future_return_{horizon}d"
    wide = _pivot_scores(signals, score_col).join(
        return_df.select(["date", "stock_id", ret_col]),
        on=["date", "stock_id"], how="left")
    model_cols = [c for c in wide.columns
                  if c not in ("date", "stock_id", ret_col)]
    dates = wide.get_column("date").unique().sort().to_list()
    coef: np.ndarray | None = None
    out_frames = []
    for i, d in enumerate(dates):
        if coef is None or i % refit_freq_days == 0:
            cutoff_idx = i - horizon - 1
            if cutoff_idx > 0:
                lo = max(0, cutoff_idx - lookback_days)
                hist = wide.filter(
                    pl.col("date").is_in(dates[lo:cutoff_idx])
                ).drop_nulls(model_cols + [ret_col])
                if hist.height > 10 * len(model_cols):
                    x = hist.select(model_cols).to_numpy()
                    y = hist.get_column(ret_col).to_numpy()
                    xtx = x.T @ x + alpha * np.eye(x.shape[1])
                    coef = np.linalg.solve(xtx, x.T @ y)
        day = wide.filter(pl.col("date") == d)
        x_day = day.select(model_cols).to_numpy()
        if coef is None:
            score = np.nanmean(x_day, axis=1)  # 历史不足 -> 等权
        else:
            score = np.nan_to_num(x_day) @ coef
        out_frames.append(day.select(["date", "stock_id"]).with_columns(
            pl.Series("ensemble_score", score)))
    return pl.concat(out_frames).sort(["date", "stock_id"])


def run_ensemble(signals: pl.DataFrame, config: dict,
                 return_df: pl.DataFrame | None = None,
                 horizon: int = 5) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """融合主入口：按 config.method 分发，返回 (ensemble 信号, 权重历史或 None)。

    各方式优缺点：
    - equal_rank：最稳健、零参数、对离群 score 不敏感；牺牲模型差异信息。
    - equal_neutral：保留 zscore 幅度信息；对未中性化残余暴露敏感。
    - icir_weighted / rankicir_weighted：利用历史表现自适应；权重噪声大、
      存在过拟合历史的风险，需要足够回看窗口。
    - family：先去家族内冗余再融合，降低同族高相关信号的隐性加权。
    - stacking：表达能力最强，可学习交互；但二层模型易过拟合、
      可解释性最差，必须严格滚动训练。
    推荐路线：第一版 equal_rank -> 第二版 icir_weighted -> 第三版 stacking。
    """
    method = config.get("method", "equal_rank")
    if method not in METHODS:
        raise ValueError(f"unknown ensemble method: {method}; expect {METHODS}")
    lookback = config.get("lookback_days", 252)
    min_hist = config.get("min_history_days", 60)
    logger.info("ensemble method=%s horizon=%d", method, horizon)
    if method == "equal_rank":
        return equal_weight_ensemble(signals, "rank_score"), None
    if method == "equal_neutral":
        return equal_weight_ensemble(signals, "neutral_score"), None
    if method in ("icir_weighted", "rankicir_weighted"):
        if return_df is None:
            raise ValueError(f"{method} requires return table")
        return icir_weighted_ensemble(
            signals, return_df, horizon, lookback, min_hist,
            use_rank=(method == "rankicir_weighted"))
    if method == "family":
        return family_ensemble(signals, "neutral_score",
                               config.get("family_weights")), None
    if return_df is None:
        raise ValueError("stacking requires return table")
    st = config.get("stacking", {})
    return stacking_ensemble(signals, return_df, horizon, lookback,
                             st.get("refit_freq_days", 21),
                             st.get("alpha", 1.0)), None


def save_ensemble(ens: pl.DataFrame, weights: pl.DataFrame | None,
                  output_root: str | Path, method: str) -> None:
    """落库 ensemble 信号与权重历史。"""
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    ens.write_parquet(out / f"ensemble_{method}.parquet")
    if weights is not None:
        weights.write_parquet(out / f"ensemble_{method}_weights.parquet")
    logger.info("ensemble saved -> %s (rows=%d)", out, ens.height)
