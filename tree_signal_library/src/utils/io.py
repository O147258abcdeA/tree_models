"""parquet 读写封装（polars）。

- 强制因子列 float32、stock_id Utf8
- 按 year 分区写出 / 追加
- 惰性读取 + 日期过滤
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)


def to_date(x: str | dt.date) -> dt.date:
    """字符串 'YYYY-MM-DD' / date -> date。"""
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def enforce_dtypes(df: pl.DataFrame, factor_cols: Iterable[str],
                   stock_col: str = "stock_id", date_col: str = "date") -> pl.DataFrame:
    """强制 dtype：因子列 float32、stock_id Utf8、date pl.Date。"""
    casts = [pl.col(c).cast(pl.Float32) for c in factor_cols if c in df.columns]
    if stock_col in df.columns:
        casts.append(pl.col(stock_col).cast(pl.Utf8))
    if date_col in df.columns and df.schema[date_col] != pl.Date:
        casts.append(pl.col(date_col).cast(pl.Date))
    return df.with_columns(casts) if casts else df


def scan_parquet_dataset(path: str | Path,
                         start_date: str | dt.date | None = None,
                         end_date: str | dt.date | None = None,
                         columns: list[str] | None = None,
                         date_col: str = "date") -> pl.LazyFrame:
    """惰性扫描 parquet 数据集（单文件 / 目录 / hive 分区），带日期谓词下推。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    pattern = str(path / "**" / "*.parquet") if path.is_dir() else str(path)
    lf = pl.scan_parquet(pattern)
    if start_date is not None:
        lf = lf.filter(pl.col(date_col) >= to_date(start_date))
    if end_date is not None:
        lf = lf.filter(pl.col(date_col) <= to_date(end_date))
    if columns is not None:
        lf = lf.select(columns)
    return lf


def write_partitioned(df: pl.DataFrame, root: str | Path,
                      date_col: str = "date", overwrite: bool = False) -> list[Path]:
    """按 year 分区写出 parquet。同分区已存在时：overwrite=True 覆盖，否则与旧数据
    concat 后去重重写（追加语义，禁止静默丢数据）。返回写出的文件列表。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    df = df.with_columns(pl.col(date_col).dt.year().alias("_year"))
    for (year,), part in df.partition_by("_year", as_dict=True).items():
        part = part.drop("_year")
        out = root / f"year={year}" / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not overwrite:
            old = pl.read_parquet(out)
            part = pl.concat([old, part], how="diagonal_relaxed")
            key = [date_col, "stock_id"] if "stock_id" in part.columns else [date_col]
            part = part.unique(subset=key, keep="last", maintain_order=True)
        part.sort(date_col).write_parquet(out)
        written.append(out)
        logger.info("wrote %s rows=%d", out, part.height)
    return written


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    """读取单个 parquet 文件。"""
    return pl.read_parquet(path, columns=columns)
