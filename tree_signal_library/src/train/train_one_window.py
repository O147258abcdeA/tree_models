"""单窗口训练模块。

train_one_model_one_window：
- 取模型 runner（model_factory）训练 + valid early stopping；
- 计算 valid 每日横截面 IC / RankIC / ICIR；
- 保存 checkpoint：model 文件 + meta.json（参数、best_iteration、窗口边界、版本）；
- 返回带元数据的结果 dict。
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.data.label_builder import label_col
from src.data.split_data import WindowSpec
from src.models.model_factory import (MODEL_FILE_EXT, build_model_name,
                                      get_runner, get_saver)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ranking 用整数分组标签训练；IC 评估统一用对应 horizon 的连续超额收益
RANKING_TRAIN_LABEL = "group_label"


def resolve_label_cols(objective_type: str, label_type: str,
                       horizon: int) -> tuple[str, str]:
    """返回 (训练标签列, IC评估标签列)。

    regression：训练与评估都用 label_type 对应的连续收益；
    ranking：训练用 y_group_h（非负整数 relevance），评估用 label_type 连续收益。
    """
    eval_col = label_col(label_type if label_type not in
                         ("rank_return", "group_label") else "excess_return", horizon)
    if objective_type == "ranking":
        return label_col(RANKING_TRAIN_LABEL, horizon), eval_col
    return label_col(label_type, horizon), eval_col


def save_model_checkpoint(result: dict, meta: dict, models_root: str | Path) -> Path:
    """保存模型 checkpoint：{model_name}/{window_id}/model.* + meta.json + 特征重要性。"""
    out_dir = Path(models_root) / meta["model_name"] / meta["window_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    saver = get_saver(meta["model_family"])
    saver(result["model"], out_dir / MODEL_FILE_EXT[meta["model_family"]])
    meta_out = {k: v for k, v in meta.items()}
    meta_out.update({
        "best_iteration": result["best_iteration"],
        "ic_valid": result["ic_valid"],
        "rankic_valid": result["rankic_valid"],
        "icir_valid": result["icir_valid"],
        "rankicir_valid": result["rankicir_valid"],
    })
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2, default=str)
    result["feature_importance"].write_parquet(out_dir / "feature_importance.parquet")
    logger.info("checkpoint saved: %s", out_dir)
    return out_dir


def train_one_model_one_window(data: dict[str, pl.DataFrame],
                               window: WindowSpec,
                               spec: dict,
                               feature_cols: list[str],
                               params: dict,
                               versions: dict,
                               models_root: str | Path,
                               seed: int = 42) -> tuple[dict, dict]:
    """训练单个模型单个窗口并保存 checkpoint。

    Parameters
    ----------
    data : {"train": df, "valid": df, "test": df}（load_window_data 输出）。
    spec : {model_family, objective_type, label_type, horizon, model_version}。
    versions : {feature_version, label_version}。

    Returns
    -------
    (result, meta)：result 为 runner 输出（含 predictions），meta 为信号元数据。
    """
    family = spec["model_family"]
    objective_type = spec["objective_type"]
    label_type = spec["label_type"]
    horizon = int(spec["horizon"])
    model_name = build_model_name(family, objective_type, label_type, horizon,
                                  spec["model_version"])
    train_label, eval_label = resolve_label_cols(objective_type, label_type, horizon)
    logger.info("train %s | window=%s | label=%s eval=%s | features=%d",
                model_name, window.window_id, train_label, eval_label,
                len(feature_cols))

    runner = get_runner(family, objective_type)
    result = runner(data["train"], data["valid"], data["test"], feature_cols,
                    train_label, params, objective_type=objective_type,
                    eval_label_col=eval_label, seed=seed)

    meta = {
        "model_family": family,
        "model_name": model_name,
        "model_version": spec["model_version"],
        "feature_version": versions["feature_version"],
        "label_version": versions["label_version"],
        "label_type": label_type,
        "horizon": horizon,
        "objective_type": objective_type,
        "window_id": window.window_id,
        "train_start": window.train_start,
        "train_end": window.train_end,
        "valid_start": window.valid_start,
        "valid_end": window.valid_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "ic_valid": result["ic_valid"],
        "rankic_valid": result["rankic_valid"],
        "icir_valid": result["icir_valid"],
        "best_iteration": result["best_iteration"],
        "params": params,
    }
    save_model_checkpoint(result, meta, models_root)
    return result, meta
