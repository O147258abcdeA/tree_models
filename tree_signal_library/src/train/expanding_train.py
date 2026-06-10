"""扩展训练总管线（Expanding Window Training）。

expanding_train_pipeline：
  与 rolling_train_pipeline 接口一致，唯一区别是训练窗口采用
  expanding 策略（训练起点固定，窗口随时间扩大）。

原理：
  - 固定训练起点 start，每个窗口使用从 start 到 valid_start 前的全部数据训练；
  - 随着时间推进，训练集不断增长，模型能看到更长的历史模式；
  - 适用于相信「数据越多越好」或市场结构变化缓慢的假设；
  - 缺点：训练集过大时可能引入 noise / regime 偏移；计算量随窗口递增。
"""
from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

from src.data.load_data import get_factor_cols
from src.data.split_data import generate_expanding_windows, load_window_data
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


def expanding_train_pipeline(feature_path: str | Path, label_path: str | Path,
                             train_cfg: dict, versions: dict,
                             start: str, end: str,
                             feature_cols: list[str] | None = None) -> pl.DataFrame:
    """扩展训练主入口。

    Parameters
    ----------
    feature_path : 特征数据 parquet 路径。
    label_path : 标签数据 parquet 路径。
    train_cfg : train_config.yaml 内容（expanding 设置 + model_specs）。
    versions : {feature_version, label_version}。
    start : 训练起始日期（固定训练起点）。
    end : 回溯区间终止日期。
    feature_cols : 特征列名列表（None 则自动检测 factor_* 列）。

    Returns
    -------
    训练日志汇总表（每窗口每模型一行：window_id, model_name, best_iteration,
    ic_valid, rankic_valid, icir_valid）。
    """
    set_seed(train_cfg.get("seed", 42))
    expanding = train_cfg["expanding"]
    windows = generate_expanding_windows(
        start, end,
        valid_years=expanding.get("valid_years", 1),
        test_months=expanding.get("test_months", 3),
        step_months=expanding.get("step_months", 3),
        embargo_days=expanding.get("embargo_days", 20),
        min_train_years=expanding.get("min_train_years", 3),
    )
    specs = train_cfg["model_specs"]
    models_root = train_cfg.get("models_path", "data/models")
    signals_root = train_cfg.get("signals_raw_path", "data/signals/raw")

    if feature_cols is None:
        schema_cols = scan_parquet_dataset(feature_path).collect_schema().names()
        feature_cols = get_factor_cols(pl.DataFrame(schema={c: pl.Float32 for c in schema_cols}))
        if not feature_cols:
            raise ValueError("no factor_* columns found in feature dataset")
    logger.info("expanding train: %d windows x %d specs, %d features",
                len(windows), len(specs), len(feature_cols))

    logs: list[dict] = []
    for window in windows:
        data = load_window_data(window, feature_path, label_path,
                                embargo_days=expanding.get("embargo_days", 20))
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
    logger.info("expanding train done:\n%s", log_df)
    return log_df
