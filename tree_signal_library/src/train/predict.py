"""样本外预测与信号落库。

- predict_one_model_one_window：训练结果 -> 测试期每日每股 raw_score；
- save_prediction_signal：组装 23 字段统一 schema 并校验入库。
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src.models.model_factory import get_predictor
from src.signals.signal_writer import build_signal_frame, write_signals
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_one_model_one_window(model, model_family: str, test_df: pl.DataFrame,
                                 feature_cols: list[str]) -> pl.DataFrame:
    """用已训练模型在测试期生成每日每股预测，返回 [date, stock_id, raw_score]。"""
    predictor = get_predictor(model_family)
    scores = predictor(model, test_df, feature_cols)
    return test_df.select(["date", "stock_id"]).with_columns(
        pl.Series("raw_score", scores).cast(pl.Float64))


def save_prediction_signal(predictions: pl.DataFrame, meta: dict,
                           signals_root: str | Path) -> list[Path]:
    """测试期预测 + 元数据 -> 统一 schema -> 校验 -> 按 model_name/year 落库。"""
    frame = build_signal_frame(predictions, meta)
    paths = write_signals(frame, signals_root)
    logger.info("signal saved: model=%s window=%s rows=%d -> %d files",
                meta["model_name"], meta["window_id"], frame.height, len(paths))
    return paths
