"""滚动窗口切分测试。"""
import datetime as dt

import pytest

from src.data.split_data import generate_rolling_windows


def test_windows_are_time_ordered_and_non_overlapping():
    windows = generate_rolling_windows("2010-01-01", "2020-12-31",
                                       train_years=5, valid_years=1,
                                       test_months=3, step_months=3)
    assert len(windows) > 1
    for w in windows:
        assert w.train_start < w.train_end < w.valid_start < w.valid_end \
            < w.test_start <= w.test_end
    # test 区间互不重叠且首尾相接（每个交易日只出现在一个 test 窗口）
    for a, b in zip(windows, windows[1:]):
        assert a.test_end < b.test_start


def test_no_windows_raises():
    with pytest.raises(ValueError):
        generate_rolling_windows("2020-01-01", "2021-01-01",
                                 train_years=5, valid_years=1)


def test_window_ids_unique():
    windows = generate_rolling_windows("2010-01-01", "2018-12-31")
    ids = [w.window_id for w in windows]
    assert len(ids) == len(set(ids))
