"""标签构造与防未来函数测试。"""
import numpy as np
import polars as pl
import pytest

from src.data.label_builder import (build_all_labels, build_classification_label,
                                    build_excess_return, build_forward_return,
                                    build_group_label, build_ranking_group,
                                    build_rank_label, build_residual_label)
from src.utils.config import load_config


@pytest.fixture(scope="module")
def labels(synthetic_panel):
    cfg = load_config("label_config.yaml")
    cfg["horizons"] = [1, 5]
    out, report = build_all_labels(synthetic_panel, cfg)
    return out, report


def test_forward_return_alignment(synthetic_panel):
    """y_raw_h 必须等于 close_{t+h}/close_t - 1，且行索引在 t。"""
    df = build_forward_return(synthetic_panel, 5)
    s = df.filter(pl.col("stock_id") == "s0000").sort("date")
    close = s.get_column("close").to_numpy()
    y = s.get_column("y_raw_5").to_numpy()
    expect = close[5] / close[0] - 1.0
    assert abs(y[0] - expect) < 1e-5


def test_last_h_days_are_null(labels):
    """末日截断测试：最后 h 个交易日标签必须全 null。"""
    out, _ = labels
    dates = out.get_column("date").unique().sort().to_list()
    tail = out.filter(pl.col("date").is_in(dates[-5:]))
    assert tail.get_column("y_raw_5").null_count() == tail.height


def test_excess_return_centered(synthetic_panel):
    df = build_forward_return(synthetic_panel, 1)
    df = build_excess_return(df, 1)
    # 等权基准下，日内超额收益均值应接近 0（不完全为 0：复合 vs 单期）
    daily_mean = df.drop_nulls("y_excess_1").group_by("date").agg(
        pl.col("y_excess_1").mean()).get_column("y_excess_1")
    assert abs(daily_mean.mean()) < 5e-3


def test_rank_and_group_labels(labels):
    out, _ = labels
    valid = out.drop_nulls("y_rank_5")
    assert valid.get_column("y_rank_5").min() >= 0.0
    assert valid.get_column("y_rank_5").max() <= 1.0
    groups = out.drop_nulls("y_group_5").get_column("y_group_5").unique().sort().to_list()
    assert groups == [0, 1, 2, 3, 4]


def test_classification_label(synthetic_panel):
    df = build_forward_return(synthetic_panel, 1)
    df = build_classification_label(df, 1, 0.2, 0.2, keep_middle=True)
    one_day = df.filter(pl.col("date") == df.get_column("date").min())
    counts = one_day.drop_nulls("y_class_1").group_by("y_class_1").agg(pl.len())
    cmap = dict(counts.iter_rows())
    n = one_day.drop_nulls("y_class_1").height
    assert abs(cmap[1] / n - 0.2) < 0.05
    assert abs(cmap[0] / n - 0.2) < 0.05


def test_residual_label_orthogonal_to_styles(synthetic_panel):
    df = build_forward_return(synthetic_panel, 1)
    df = build_residual_label(df, 1, ["log_mktcap", "beta"], "industry")
    d = df.get_column("date").unique().sort().to_list()[0]
    day = df.filter(pl.col("date") == d).drop_nulls("y_residual_1")
    assert abs(day.select(pl.corr("y_residual_1", "log_mktcap")).item()) < 1e-6


def test_ranking_group_requires_sorted(synthetic_panel):
    df = synthetic_panel.sort(["stock_id", "date"])  # 故意按 stock 排序
    with pytest.raises(ValueError):
        build_ranking_group(df)
    ok = synthetic_panel.sort(["date", "stock_id"])
    sizes = build_ranking_group(ok)
    assert sizes.sum() == ok.height
    assert len(sizes) == ok.get_column("date").n_unique()


def test_label_quality_report(labels):
    _, report = labels
    assert "coverage" in report.columns
    cov = report.filter(pl.col("label") == "y_raw_1").get_column("coverage")[0]
    assert cov > 0.9
