"""合成小面板数据 fixture：~40 股 x ~260 交易日 x 20 因子。

因子 factor_001 与未来收益弱相关（用于验证 IC 方向），其余为噪声。
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_STOCKS = 40
N_DAYS = 260
N_FACTORS = 20
INDUSTRIES = ["tech", "bank", "energy", "health"]


def _trading_days(start: dt.date, n: int) -> list[dt.date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


@pytest.fixture(scope="session")
def synthetic_panel() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    dates = _trading_days(dt.date(2020, 1, 1), N_DAYS)
    stocks = [f"s{i:04d}" for i in range(N_STOCKS)]
    rows = []
    close = {s: 10.0 + rng.random() * 10 for s in stocks}
    industry = {s: INDUSTRIES[i % len(INDUSTRIES)] for i, s in enumerate(stocks)}
    signal_strength = {s: rng.normal() for s in stocks}
    for d in dates:
        for s in stocks:
            # factor_001 含真实 alpha 信息：驱动次日收益
            f1 = signal_strength[s] + rng.normal(0, 1)
            ret = 0.002 * f1 + rng.normal(0, 0.02)
            close[s] *= (1 + ret)
            row = {
                "date": d, "stock_id": s, "industry": industry[s],
                "market_cap": float(np.exp(rng.normal(22, 1))),
                "log_mktcap": float(rng.normal(22, 1)),
                "turnover": float(abs(rng.normal(0.02, 0.01))),
                "beta": float(rng.normal(1, 0.2)),
                "volatility": float(abs(rng.normal(0.3, 0.05))),
                "benchmark_weight": 1.0 / N_STOCKS,
                "return": ret, "close": close[s],
                "is_st": int(rng.random() < 0.03),
                "is_suspended": int(rng.random() < 0.02),
                "is_limit_up": 0, "is_limit_down": 0,
                "factor_001": f1,
            }
            for k in range(2, N_FACTORS + 1):
                row[f"factor_{k:03d}"] = float(rng.normal())
            # 随机缺失
            if rng.random() < 0.05:
                row["factor_002"] = None
            rows.append(row)
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    return df.sort(["date", "stock_id"])


@pytest.fixture(scope="session")
def raw_dataset(synthetic_panel, tmp_path_factory) -> Path:
    """合成面板写成按 year 分区的 parquet 数据集。"""
    from src.utils.io import write_partitioned
    root = tmp_path_factory.mktemp("raw")
    write_partitioned(synthetic_panel, root)
    return root


@pytest.fixture(scope="session")
def factor_cols(synthetic_panel) -> list[str]:
    from src.data.load_data import get_factor_cols
    return get_factor_cols(synthetic_panel)
