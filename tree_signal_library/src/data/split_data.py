"""滚动窗口切分模块。

- generate_rolling_windows: 生成 train/valid/test 时间窗口序列
- load_window_data: 加载单窗口数据，train/valid、valid/test 之间加 embargo gap
  （= max horizon 个交易日），杜绝标签前视泄漏；内置时间切分断言。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path

import polars as pl

from src.utils.io import scan_parquet_dataset, to_date
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WindowSpec:
    """单个滚动窗口的六个时间边界（闭区间）。"""
    window_id: str
    train_start: dt.date
    train_end: dt.date
    valid_start: dt.date
    valid_end: dt.date
    test_start: dt.date
    test_end: dt.date

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, dt.date) else v
                for k, v in asdict(self).items()}


def _add_months(d: dt.date, months: int) -> dt.date:
    """日历月平移（保持月初对齐用）。"""
    y, m = divmod((d.year * 12 + d.month - 1) + months, 12)
    return dt.date(y, m + 1, 1) if d.day == 1 else dt.date(y, m + 1, min(d.day, 28))


def generate_rolling_windows(start: str | dt.date, end: str | dt.date,
                             train_years: int = 5, valid_years: int = 1,
                             test_months: int = 3, step_months: int = 3,
                             embargo_days: int = 20) -> list[WindowSpec]:
    """生成滚动窗口序列（按日历边界；embargo 在加载时以交易日实施）。

    所有切分严格按时间向前推进，禁止随机打乱。
    test 区间相邻窗口首尾相接、互不重叠，保证每个交易日只出现在一个 test 窗口。
    """
    start, end = to_date(start), to_date(end)
    windows: list[WindowSpec] = []
    i = 0
    train_start = start
    while True:
        train_end = _add_months(train_start, 12 * train_years) - dt.timedelta(days=1)
        valid_start = _add_months(train_start, 12 * train_years)
        valid_end = _add_months(valid_start, 12 * valid_years) - dt.timedelta(days=1)
        test_start = _add_months(valid_start, 12 * valid_years)
        test_end = _add_months(test_start, test_months) - dt.timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_end, end)
        windows.append(WindowSpec(
            window_id=f"w{i:03d}_{test_start:%Y%m%d}",
            train_start=train_start, train_end=train_end,
            valid_start=valid_start, valid_end=valid_end,
            test_start=test_start, test_end=test_end,
        ))
        i += 1
        train_start = _add_months(train_start, step_months)
    if not windows:
        raise ValueError(f"no rolling windows in [{start}, {end}]; "
                         "check train/valid/test window sizes")
    logger.info("generated %d rolling windows, embargo=%d trading days",
                len(windows), embargo_days)
    return windows


def generate_expanding_windows(start: str | dt.date, end: str | dt.date,
                               valid_years: int = 1,
                               test_months: int = 3, step_months: int = 3,
                               embargo_days: int = 20,
                               min_train_years: int = 3) -> list[WindowSpec]:
    """生成扩展窗口序列（Expanding Window）。

    与 rolling 的区别：训练起点固定为 start，训练窗口随时间推移持续扩大，
    每个窗口使用从开始到当前的全部历史数据训练。

    原理：
    - Expanding 方法假设更多历史数据能提供更好的模型泛化能力；
    - 训练集持续累积，适合数据量有限或市场结构相对稳定的场景；
    - 对比 Rolling：Rolling 假设近期数据更相关（regime shift），Expanding
      假设长期模式更稳定。

    Parameters
    ----------
    start : 全部历史数据起始日期（固定训练起点）。
    end : 回溯区间终止日期。
    valid_years : 验证窗口年数。
    test_months : 测试窗口月数。
    step_months : 每次向前推进的月数。
    embargo_days : train/valid、valid/test 之间隔离的交易日数。
    min_train_years : 最小训练年数（第一个窗口至少需要这么多年的训练数据）。

    Returns
    -------
    WindowSpec 列表（按时间顺序），训练起点固定，窗口逐步扩大。
    """
    start, end = to_date(start), to_date(end)
    windows: list[WindowSpec] = []
    i = 0
    # 第一个 valid 起点：至少满足 min_train_years
    first_valid_start = _add_months(start, 12 * min_train_years)
    valid_start = first_valid_start
    while True:
        train_start = start  # 固定起点
        train_end = valid_start - dt.timedelta(days=1)
        valid_end = _add_months(valid_start, 12 * valid_years) - dt.timedelta(days=1)
        test_start = _add_months(valid_start, 12 * valid_years)
        test_end = _add_months(test_start, test_months) - dt.timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_end, end)
        windows.append(WindowSpec(
            window_id=f"e{i:03d}_{test_start:%Y%m%d}",
            train_start=train_start, train_end=train_end,
            valid_start=valid_start, valid_end=valid_end,
            test_start=test_start, test_end=test_end,
        ))
        i += 1
        valid_start = _add_months(valid_start, step_months)
    if not windows:
        raise ValueError(f"no expanding windows in [{start}, {end}]; "
                         "check min_train_years/valid_years/test_months")
    logger.info("generated %d expanding windows, embargo=%d trading days",
                len(windows), embargo_days)
    return windows


def _load_span(feature_path: str | Path, label_path: str | Path,
               start: dt.date, end: dt.date) -> pl.DataFrame:
    """加载 [start, end] 区间内的特征 + 标签（按 date, stock_id join）。"""
    feat = scan_parquet_dataset(feature_path, start, end).collect()
    lab = scan_parquet_dataset(label_path, start, end).collect()
    return feat.join(lab, on=["date", "stock_id"], how="inner").sort(["date", "stock_id"])


def _shrink_end_by_trading_days(df_dates: list[dt.date], end: dt.date,
                                n_days: int) -> dt.date | None:
    """把区间右端点向前收缩 n_days 个交易日（embargo 实施）。"""
    eligible = [d for d in df_dates if d <= end]
    if len(eligible) <= n_days:
        return None
    return eligible[-(n_days + 1)]


def load_window_data(window: WindowSpec, feature_path: str | Path,
                     label_path: str | Path,
                     embargo_days: int = 20) -> dict[str, pl.DataFrame]:
    """加载单窗口 train / valid / test 数据。

    embargo：train_end、valid_end 各向前收缩 embargo_days 个交易日，
    使 train/valid 标签的未来收益窗口不与下一段重叠。
    内置断言：max(train.date) < min(valid.date) <= max(valid.date) < min(test.date)。
    """
    all_dates = (
        scan_parquet_dataset(label_path, window.train_start, window.test_end,
                             columns=["date"])
        .unique().collect().get_column("date").sort().to_list()
    )
    train_end = _shrink_end_by_trading_days(all_dates, window.train_end, embargo_days)
    valid_end = _shrink_end_by_trading_days(all_dates, window.valid_end, embargo_days)
    if train_end is None or valid_end is None:
        raise ValueError(f"window {window.window_id}: not enough trading days for embargo")

    train = _load_span(feature_path, label_path, window.train_start, train_end)
    valid = _load_span(feature_path, label_path, window.valid_start, valid_end)
    test = _load_span(feature_path, label_path, window.test_start, window.test_end)
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        if part.is_empty():
            raise ValueError(f"window {window.window_id}: empty {name} split")

    # 时间切分断言（防泄漏硬约束）
    assert train.get_column("date").max() < valid.get_column("date").min(), \
        "train/valid overlap detected"
    assert valid.get_column("date").max() < test.get_column("date").min(), \
        "valid/test overlap detected"
    logger.info(
        "window %s: train %s..%s (%d rows) | valid %s..%s (%d rows) | test %s..%s (%d rows)",
        window.window_id,
        train.get_column("date").min(), train.get_column("date").max(), train.height,
        valid.get_column("date").min(), valid.get_column("date").max(), valid.height,
        test.get_column("date").min(), test.get_column("date").max(), test.height)
    return {"train": train, "valid": valid, "test": test}
