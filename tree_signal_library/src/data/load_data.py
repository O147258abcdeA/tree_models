"""数据读取模块（polars 惰性读取，禁止一次性加载所有年份）。

底层原语为 get_factor(date)：逐日读出单日横截面。
load_factor_data 在其上提供日期区间的惰性扫描（谓词下推到 parquet 分区）。
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import polars as pl

from src.utils.io import scan_parquet_dataset, to_date
from src.utils.logger import get_logger

logger = get_logger(__name__)

FACTOR_PATTERN = re.compile(r"^factor_\d+$")
# 已知的非因子分类列
KNOWN_CATEGORICAL = ["stock_id", "industry"]


def get_factor(date: str | dt.date, path: str | Path,
               columns: list[str] | None = None) -> pl.DataFrame:
    """逐日读取原语：读出单个交易日的横截面因子数据。

    Parameters
    ----------
    date : 交易日。
    path : 因子库根目录（parquet，按 year 或 date 分区）。
    columns : 只读取指定列；None 表示全部列。

    Returns
    -------
    单日横截面 DataFrame；该日无数据时返回空表（不静默吞错，调用方自行判断）。
    """
    d = to_date(date)
    lf = scan_parquet_dataset(path, start_date=d, end_date=d, columns=columns)
    df = lf.collect()
    if df.is_empty():
        logger.warning("no data for date=%s under %s", d, path)
    return df


def load_factor_data(path: str | Path,
                     start_date: str | dt.date | None = None,
                     end_date: str | dt.date | None = None,
                     columns: list[str] | None = None) -> pl.LazyFrame:
    """按日期区间惰性扫描因子库（谓词下推，不会立即读入内存）。

    返回 LazyFrame；调用方应逐窗口 / 逐日 collect，禁止对多年数据整体 collect。
    """
    logger.info("scan factor data: path=%s start=%s end=%s cols=%s",
                path, start_date, end_date,
                len(columns) if columns else "all")
    return scan_parquet_dataset(path, start_date=start_date,
                                end_date=end_date, columns=columns)


def iter_dates(path: str | Path,
               start_date: str | dt.date | None = None,
               end_date: str | dt.date | None = None,
               date_col: str = "date") -> list[dt.date]:
    """列出数据集中 [start, end] 区间内的全部交易日（只扫描 date 列）。"""
    lf = scan_parquet_dataset(path, start_date=start_date, end_date=end_date,
                              columns=[date_col])
    return lf.unique().collect().get_column(date_col).sort().to_list()


def get_factor_cols(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """按 ^factor_\\d+$ 正则识别因子列。"""
    cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    return [c for c in cols if FACTOR_PATTERN.match(c)]


def get_categorical_cols(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """识别分类列：已知分类列 + schema 中的字符串/Categorical 非因子列。"""
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    out = []
    for c, dtype in schema.items():
        if FACTOR_PATTERN.match(c):
            continue
        if c in KNOWN_CATEGORICAL or dtype in (pl.Utf8, pl.Categorical):
            out.append(c)
    return out
