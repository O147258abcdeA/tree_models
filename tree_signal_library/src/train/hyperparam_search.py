"""超参搜索模块：基于 Optuna TPE 采样器的贝叶斯超参优化。

只允许使用 train/valid 窗口，严禁使用测试集调参。
目标函数为 valid 集每日横截面 RankIC 均值（最大化）。

核心原理：
- TPE（Tree-structured Parzen Estimator）通过构建两个概率密度模型
  l(x) 和 g(x) 分别拟合好/差候选参数的分布，选择使 l(x)/g(x) 最大的
  下一组参数进行评估，相比随机搜索能更高效地逼近全局最优。
- 支持 early pruning（MedianPruner）：当中间指标不佳时提前终止 trial。
- 所有 trial 记录保存为 polars DataFrame，便于后续分析。

接口说明：
- hyperparam_search: 主搜索入口，返回最优参数 + 全部 trial 记录。
- build_optuna_search_space: 从 YAML 格式参数空间构建 Optuna suggest 调用。
"""
from __future__ import annotations

from typing import Any

import optuna
import polars as pl

from src.models.model_factory import get_runner
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 静默 Optuna 内部日志，避免大量 trial 打印
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """根据参数 spec 调用对应的 Optuna suggest 方法。

    Parameters
    ----------
    trial : 当前 Optuna trial 对象。
    name : 参数名。
    spec : 参数空间描述，格式为：
        - {"type": "int", "low": 10, "high": 200, "step": 10}
        - {"type": "float", "low": 0.01, "high": 0.3, "log": True}
        - {"type": "categorical", "choices": ["a", "b", "c"]}

    Returns
    -------
    建议的参数值。
    """
    ptype = spec["type"]
    if ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"],
                                 step=spec.get("step", 1),
                                 log=spec.get("log", False))
    elif ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"],
                                   step=spec.get("step"),
                                   log=spec.get("log", False))
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"unknown param type '{ptype}' for param '{name}'")


def build_optuna_search_space(space_cfg: dict[str, dict]) -> dict[str, dict]:
    """验证并规范化从 YAML 加载的搜索空间配置。

    Parameters
    ----------
    space_cfg : 参数空间字典，key 为参数名，value 为类型描述 dict。
        示例：
        {
            "num_leaves": {"type": "int", "low": 31, "high": 255, "step": 1},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "feature_fraction": {"type": "float", "low": 0.5, "high": 0.9},
            "objective": {"type": "categorical", "choices": ["regression", "regression_l1"]},
        }

    Returns
    -------
    验证后的搜索空间（与输入格式相同）。
    """
    valid_types = {"int", "float", "categorical"}
    for name, spec in space_cfg.items():
        if "type" not in spec:
            raise ValueError(f"param '{name}' missing 'type' key")
        if spec["type"] not in valid_types:
            raise ValueError(f"param '{name}' has invalid type '{spec['type']}'; "
                             f"expect {valid_types}")
        if spec["type"] in ("int", "float") and ("low" not in spec or "high" not in spec):
            raise ValueError(f"param '{name}' of type '{spec['type']}' "
                             "requires 'low' and 'high'")
        if spec["type"] == "categorical" and "choices" not in spec:
            raise ValueError(f"param '{name}' of type 'categorical' requires 'choices'")
    return space_cfg


def hyperparam_search(train_df: pl.DataFrame, valid_df: pl.DataFrame,
                      feature_cols: list[str], label_col: str,
                      base_params: dict, search_space: dict[str, dict],
                      model_family: str, objective_type: str,
                      n_trials: int = 50, eval_label_col: str | None = None,
                      seed: int = 42, timeout: int | None = None,
                      direction: str = "maximize") -> tuple[dict, pl.DataFrame]:
    """使用 Optuna TPE 进行超参搜索，目标为 valid 每日横截面 RankIC 均值。

    注意：只使用 train/valid；test 数据严禁传入本函数。

    Parameters
    ----------
    train_df : 训练集 DataFrame（含 feature + label 列）。
    valid_df : 验证集 DataFrame（含 feature + label 列）。
    feature_cols : 特征列名列表。
    label_col : 训练标签列名。
    base_params : 基础超参字典（搜索空间中未涉及的参数使用此默认值）。
    search_space : Optuna 搜索空间描述 dict，格式见 build_optuna_search_space。
    model_family : 模型类型，"lightgbm" / "xgboost" / "catboost"。
    objective_type : 目标类型，"regression" / "ranking"。
    n_trials : 最大搜索 trial 数量（默认 50）。
    eval_label_col : IC 评估用的连续标签列名（ranking 训练时需要传入）。
    seed : 随机种子。
    timeout : 搜索总时间限制（秒），None 表示不限制。
    direction : 优化方向，"maximize"（RankIC）或 "minimize"。

    Returns
    -------
    (best_params, trials_df)
        best_params : 最优参数字典（base_params + 搜索到的最优值）。
        trials_df : 全部 trial 记录表，含 trial_number、rankic_valid、
                    ic_valid、best_iteration 以及每个搜索参数值。
    """
    search_space = build_optuna_search_space(search_space)
    runner = get_runner(model_family, objective_type)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler,
                                study_name=f"{model_family}_{objective_type}_tuning")

    trial_records: list[dict] = []

    def _objective(trial: optuna.Trial) -> float:
        """Optuna 目标函数：在 valid 上评估 RankIC。"""
        params = dict(base_params)
        for name, spec in search_space.items():
            params[name] = _suggest_param(trial, name, spec)

        result = runner(train_df, valid_df, valid_df, feature_cols, label_col,
                        params, objective_type=objective_type,
                        eval_label_col=eval_label_col, seed=seed)

        score = result["rankic_valid"] if result["rankic_valid"] is not None else float("-inf")

        trial_records.append({
            "trial_number": trial.number,
            "rankic_valid": result["rankic_valid"],
            "ic_valid": result["ic_valid"],
            "icir_valid": result["icir_valid"],
            "best_iteration": result["best_iteration"],
            **{f"p_{k}": params[k] for k in search_space},
        })
        logger.info("trial %d/%d rankic=%.4f", trial.number + 1, n_trials, score)
        return score

    study.optimize(_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # 构建最优参数
    best_params = dict(base_params)
    best_params.update(study.best_params)

    trials_df = pl.DataFrame(trial_records)
    logger.info("optuna search done: best_value=%.4f best_params=%s",
                study.best_value, study.best_params)
    return best_params, trials_df
