"""ranking group 统一工具（LightGBM / XGBoost / CatBoost 三库共用）。

机制保证：group 全部来自同一函数 build_group_sizes / build_group_ids，
强制 df 按 date 排序，禁止把不同日期的股票混进同一个 ranking group。
"""
from __future__ import annotations

import numpy as np
import polars as pl


def assert_sorted_by_date(df: pl.DataFrame, date_col: str = "date") -> None:
    """断言 df 已按 date 排序（ranking group 构造的前置条件）。"""
    if not df.get_column(date_col).is_sorted():
        raise ValueError("dataframe must be sorted by date for ranking groups")


def build_group_sizes(df: pl.DataFrame, date_col: str = "date") -> np.ndarray:
    """每个 date 一个 query group，返回按 date 顺序的 group sizes。

    LightGBM 的 group= 与 XGBoost DMatrix.set_group 直接使用。
    """
    assert_sorted_by_date(df, date_col)
    sizes = (
        df.group_by(date_col, maintain_order=True)
        .agg(pl.len().alias("n"))
        .get_column("n")
        .to_numpy()
    )
    if sizes.sum() != df.height:
        raise AssertionError("group sizes do not cover all rows")
    return sizes


def build_group_ids(df: pl.DataFrame, date_col: str = "date") -> np.ndarray:
    """每个 date 映射为一个整数 group_id（CatBoost Pool 使用）。"""
    assert_sorted_by_date(df, date_col)
    return df.get_column(date_col).rank("dense").cast(pl.Int32).to_numpy() - 1


def to_feature_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """特征列 -> float32 numpy 矩阵（与树模型库交互的唯一转换点）。"""
    return df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_numpy()


def to_label_array(df: pl.DataFrame, label_col: str) -> np.ndarray:
    """标签列 -> float64 numpy 数组。"""
    return df.get_column(label_col).cast(pl.Float64).to_numpy()


def to_rank_relevance(df: pl.DataFrame, label_col: str) -> np.ndarray:
    """ranking relevance：要求为非负整数（由 y_group_h 分组标签充当）。"""
    y = df.get_column(label_col).cast(pl.Float64).to_numpy()
    if np.isnan(y).any():
        raise ValueError("ranking labels contain nulls; drop them before training")
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError("ranking relevance must be non-negative integers "
                         f"(use y_group_h labels, got values like {y[:5]})")
    return y.astype(np.int32)


def drop_null_labels(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
    """剔除标签为 null 的样本（序列末尾不足 h 天的行）。"""
    return df.filter(pl.col(label_col).is_not_null())
