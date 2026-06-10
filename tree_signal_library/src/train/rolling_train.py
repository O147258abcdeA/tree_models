"""滚动训练总管线。

rolling_train_pipeline：
  遍历窗口 x model_specs，单窗口内：
    加载 train/valid/test（含 embargo）-> 训练 -> valid IC/RankIC ->
    test 样本外预测 -> 保存模型 checkpoint -> 保存测试期信号 -> 释放内存。
内存中始终只保留一个窗口的数据。
"""
from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

from src.data.load_data import get_factor_cols
from src.data.split_data import generate_rolling_windows, load_window_data
from src.models.model_factory import get_config_file
from src.train.predict import save_prediction_signal
from src.train.train_one_window import train_one_model_one_window
from src.utils.config import load_config
from src.utils.io import scan_parquet_dataset
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _model_params(spec: dict, override: dict | None = None) -> dict:
    """加载 family 配置中 objective_type 对应的默认超参，可被 spec.params 覆盖。"""
    from src.utils.config import merge_config
    cfg = load_config(get_config_file(spec["model_family"]))
    params = cfg.get(spec["objective_type"])
    if params is None:
        raise ValueError(f"no default params for {spec['model_family']}/"
                         f"{spec['objective_type']}")
    return merge_config(params, override or spec.get("params"))


def rolling_train_pipeline(feature_path: str | Path, label_path: str | Path,
                           train_cfg: dict, versions: dict,
                           start: str, end: str,
                           feature_cols: list[str] | None = None) -> pl.DataFrame:
    """滚动训练主入口。

    Parameters
    ----------
    train_cfg : train_config.yaml 内容（rolling 设置 + model_specs）。
    versions : {feature_version, label_version}。
    start, end : 整个回溯区间（start 为最早 train_start，end 为最晚 test_end）。

    Returns
    -------
    训练日志汇总表（每窗口每模型一行：window_id, model_name, best_iteration,
    ic_valid, rankic_valid, icir_valid）。
    """
    set_seed(train_cfg.get("seed", 42))
    rolling = train_cfg["rolling"]
    windows = generate_rolling_windows(
        start, end,
        train_years=rolling.get("train_years", 5),
        valid_years=rolling.get("valid_years", 1),
        test_months=rolling.get("test_months", 3),
        step_months=rolling.get("step_months", 3),
        embargo_days=rolling.get("embargo_days", 20),
    )
    specs = train_cfg["model_specs"]
    models_root = train_cfg.get("models_path", "data/models")
    signals_root = train_cfg.get("signals_raw_path", "data/signals/raw")

    if feature_cols is None:
        schema_cols = scan_parquet_dataset(feature_path).collect_schema().names()
        feature_cols = get_factor_cols(pl.DataFrame(schema={c: pl.Float32 for c in schema_cols}))
        if not feature_cols:
            raise ValueError("no factor_* columns found in feature dataset")
    logger.info("rolling train: %d windows x %d specs, %d features",
                len(windows), len(specs), len(feature_cols))

    logs: list[dict] = []
    for window in windows:
        data = load_window_data(window, feature_path, label_path,
                                embargo_days=rolling.get("embargo_days", 20))
        for spec in specs:
            params = _model_params(spec)
            result, meta = train_one_model_one_window(
                data, window, spec, feature_cols, params, versions,
                models_root, seed=train_cfg.get("seed", 42))
            save_prediction_signal(result["predictions"], meta, signals_root)
            logs.append({
                "window_id": window.window_id,
                "model_name": meta["model_name"],
                "best_iteration": meta["best_iteration"],
                "ic_valid": meta["ic_valid"],
                "rankic_valid": meta["rankic_valid"],
                "icir_valid": meta["icir_valid"],
            })
            del result
        del data
        gc.collect()

    log_df = pl.DataFrame(logs)
    logger.info("rolling train done:\n%s", log_df)
    return log_df
