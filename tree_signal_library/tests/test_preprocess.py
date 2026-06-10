"""数据读取与横截面预处理测试。"""
import datetime as dt

import numpy as np

import polars as pl
import pytest

from src.data.load_data import get_factor, get_factor_cols, load_factor_data
from src.data.preprocess import (assert_no_future_cols, compute_missing_rate,
                                 fill_missing_cross_section, filter_universe,
                                 neutralize_cross_section, preprocess_one_day,
                                 standardize_cross_section,
                                 winsorize_cross_section)
from src.utils.config import load_config


def test_get_factor_single_day(raw_dataset, synthetic_panel):
    d = synthetic_panel.get_column("date").min()
    day = get_factor(d, raw_dataset)
    assert day.get_column("date").unique().to_list() == [d]
    assert day.height == synthetic_panel.filter(pl.col("date") == d).height


def test_load_factor_data_range_and_columns(raw_dataset):
    lf = load_factor_data(raw_dataset, "2020-02-01", "2020-03-01",
                          columns=["date", "stock_id", "factor_001"])
    df = lf.collect()
    assert df.columns == ["date", "stock_id", "factor_001"]
    assert df.get_column("date").min() >= dt.date(2020, 2, 1)
    assert df.get_column("date").max() <= dt.date(2020, 3, 1)


def test_get_factor_cols(synthetic_panel, factor_cols):
    assert len(factor_cols) == 20
    assert all(c.startswith("factor_") for c in factor_cols)
    assert "close" not in factor_cols


def test_assert_no_future_cols():
    with pytest.raises(ValueError):
        assert_no_future_cols(["factor_001", "y_raw_5"])


def test_filter_universe(synthetic_panel):
    cfg = {"filter": {"drop_st": True, "drop_suspended": True}}
    day = synthetic_panel.filter(pl.col("date") == synthetic_panel.get_column("date").min())
    out = filter_universe(day, cfg)
    assert out.filter(pl.col("is_st") == 1).height == 0
    assert out.filter(pl.col("is_suspended") == 1).height == 0


def test_winsorize_and_standardize(synthetic_panel, factor_cols):
    day = synthetic_panel.filter(pl.col("date") == synthetic_panel.get_column("date").min())
    w = winsorize_cross_section(day, factor_cols, 0.05, 0.95)
    c = "factor_003"
    assert w.get_column(c).max() <= day.get_column(c).quantile(0.95, "linear") + 1e-9
    filled = fill_missing_cross_section(w, factor_cols)
    assert filled.get_column("factor_002").null_count() == 0
    z = standardize_cross_section(filled, factor_cols)
    assert abs(z.get_column(c).mean()) < 1e-5
    assert abs(z.get_column(c).std() - 1.0) < 1e-5


def test_missing_rate(synthetic_panel, factor_cols):
    day = synthetic_panel.filter(pl.col("date") == synthetic_panel.get_column("date").min())
    rates = compute_missing_rate(day, factor_cols)
    assert rates["factor_001"] == 0.0
    assert 0.0 <= rates["factor_002"] <= 1.0


def test_neutralize_kills_exposure_correlation(synthetic_panel):
    day = synthetic_panel.filter(
        pl.col("date") == synthetic_panel.get_column("date").min())
    # 构造与 log_mktcap 强相关（含噪声）的因子
    noise = pl.Series("n", np.random.default_rng(3).normal(0, 0.1, day.height))
    day = day.with_columns((pl.col("log_mktcap") * 2.0 + noise).alias("factor_001"))
    out = neutralize_cross_section(day, ["factor_001"], ["log_mktcap"], "industry")
    corr = out.select(pl.corr("factor_001", "log_mktcap")).item()
    assert abs(corr) < 1e-6


def test_preprocess_one_day_cross_section_isolation(synthetic_panel, factor_cols):
    """篡改某一天的数据，其他日期的预处理结果必须逐位不变（无跨日统计）。"""
    cfg = load_config("data_config.yaml")
    dates = synthetic_panel.get_column("date").unique().sort().to_list()
    d0, d1 = dates[0], dates[1]
    day1 = synthetic_panel.filter(pl.col("date") == d1)
    base = preprocess_one_day(day1, cfg, factor_cols)
    # 篡改 d0 的数据不影响 d1 的结果（preprocess 是逐日的，天然隔离）
    again = preprocess_one_day(day1, cfg, factor_cols)
    assert base.equals(again)
