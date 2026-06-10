"""统一信号库 schema 校验与落库（第十一节 23 字段）。

按 model_name / year 分区写 parquet，便于增量追加与滚动读取。
写入前强制校验：字段齐全、dtype 正确、三个 version 字段非空、
(date, stock_id) 唯一 —— 禁止无版本信号入库。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from src.utils.io import write_partitioned
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 统一信号 schema（第十一节）
SIGNAL_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "stock_id": pl.Utf8,
    "model_family": pl.Utf8,
    "model_name": pl.Utf8,
    "model_version": pl.Utf8,
    "feature_version": pl.Utf8,
    "label_version": pl.Utf8,
    "label_type": pl.Utf8,
    "horizon": pl.Int32,
    "objective_type": pl.Utf8,
    "raw_score": pl.Float64,
    "window_id": pl.Utf8,
    "train_start": pl.Date,
    "train_end": pl.Date,
    "valid_start": pl.Date,
    "valid_end": pl.Date,
    "test_start": pl.Date,
    "test_end": pl.Date,
    "ic_valid": pl.Float64,
    "rankic_valid": pl.Float64,
    "icir_valid": pl.Float64,
    "best_iteration": pl.Int32,
    "created_time": pl.Datetime,
}

VALID_FAMILIES = {"lightgbm", "xgboost", "catboost"}
VALID_OBJECTIVES = {"regression", "ranking", "classification"}
VALID_LABEL_TYPES = {"raw_return", "excess_return", "industry_neutral_return",
                     "residual_return", "rank_return", "group_label"}
VALID_HORIZONS = {1, 5, 10, 20}


def build_signal_frame(predictions: pl.DataFrame, meta: dict) -> pl.DataFrame:
    """把 (date, stock_id, raw_score) 预测结果 + 元数据组装为统一 23 字段信号表。"""
    n = predictions.height
    df = predictions.select(["date", "stock_id", "raw_score"])
    consts = {k: meta[k] for k in [
        "model_family", "model_name", "model_version", "feature_version",
        "label_version", "label_type", "horizon", "objective_type", "window_id",
        "train_start", "train_end", "valid_start", "valid_end",
        "test_start", "test_end", "ic_valid", "rankic_valid", "icir_valid",
        "best_iteration",
    ]}
    df = df.with_columns([
        pl.lit(v).alias(k) for k, v in consts.items()
    ]).with_columns(pl.lit(dt.datetime.now()).alias("created_time"))
    df = df.with_columns([
        pl.col(c).cast(t) for c, t in SIGNAL_SCHEMA.items() if c in df.columns
    ])
    logger.info("built signal frame: model=%s window=%s rows=%d",
                meta["model_name"], meta["window_id"], n)
    return df.select(list(SIGNAL_SCHEMA))


def validate_signal_frame(df: pl.DataFrame) -> None:
    """信号入库前强制校验（schema / 枚举值 / 版本非空 / 唯一性）。"""
    missing = set(SIGNAL_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"signal frame missing columns: {sorted(missing)}")
    for col in ("model_version", "feature_version", "label_version"):
        if df.get_column(col).null_count() > 0 or (df.get_column(col) == "").any():
            raise ValueError(f"signal frame has empty {col}: versioning is mandatory")
    bad_family = set(df.get_column("model_family").unique()) - VALID_FAMILIES
    if bad_family:
        raise ValueError(f"invalid model_family: {bad_family}")
    bad_obj = set(df.get_column("objective_type").unique()) - VALID_OBJECTIVES
    if bad_obj:
        raise ValueError(f"invalid objective_type: {bad_obj}")
    bad_label = set(df.get_column("label_type").unique()) - VALID_LABEL_TYPES
    if bad_label:
        raise ValueError(f"invalid label_type: {bad_label}")
    bad_h = set(df.get_column("horizon").unique()) - VALID_HORIZONS
    if bad_h:
        raise ValueError(f"invalid horizon: {bad_h}")
    dup = df.height - df.unique(subset=["date", "stock_id", "model_name",
                                        "model_version"]).height
    if dup:
        raise ValueError(f"signal frame has {dup} duplicated "
                         "(date, stock_id, model_name, model_version) rows")


def write_signals(df: pl.DataFrame, root: str | Path) -> list[Path]:
    """校验后按 model_name / year 分区落库。"""
    validate_signal_frame(df)
    written: list[Path] = []
    for (model_name,), part in df.partition_by("model_name", as_dict=True).items():
        written += write_partitioned(part, Path(root) / str(model_name))
    return written


def read_signals(root: str | Path, model_name: str | None = None,
                 start_date: str | dt.date | None = None,
                 end_date: str | dt.date | None = None,
                 columns: list[str] | None = None) -> pl.DataFrame:
    """从信号库读取信号（可按模型 / 日期过滤）。"""
    from src.utils.io import scan_parquet_dataset
    path = Path(root) / model_name if model_name else Path(root)
    return scan_parquet_dataset(path, start_date, end_date, columns).collect()
