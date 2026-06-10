"""tree_signal_library 管线 CLI 入口。

用法：
  python run_pipeline.py --stage preprocess --start 2015-01-01 --end 2024-12-31
  python run_pipeline.py --stage label     --start 2015-01-01 --end 2024-12-31
  python run_pipeline.py --stage train     --start 2015-01-01 --end 2024-12-31
  python run_pipeline.py --stage signal
  python run_pipeline.py --stage evaluate
  python run_pipeline.py --stage ensemble  --horizon 5
  python run_pipeline.py --stage backtest  --signal-file data/signals/ensemble/ensemble_equal_rank.parquet

各 stage 的路径与参数均来自 config/*.yaml。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from src.utils.config import load_config, get_version  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("run_pipeline", ROOT / "data" / "reports" / "pipeline.log")


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def stage_preprocess(args) -> None:
    from src.data.preprocess import preprocess_panel
    cfg = load_config("data_config.yaml")
    preprocess_panel(_abs(cfg["raw_path"]), _abs(cfg["processed_path"]), cfg,
                     args.start, args.end)


def stage_label(args) -> None:
    from src.data.label_builder import build_labels_to_parquet
    dcfg = load_config("data_config.yaml")
    lcfg = load_config("label_config.yaml")
    build_labels_to_parquet(_abs(dcfg["raw_path"]), _abs(lcfg["labels_path"]),
                            lcfg, args.start, args.end)


def stage_train(args) -> None:
    from src.train.rolling_train import rolling_train_pipeline
    dcfg = load_config("data_config.yaml")
    lcfg = load_config("label_config.yaml")
    tcfg = load_config("train_config.yaml")
    versions = {"feature_version": get_version(dcfg, "feature_version", "f"),
                "label_version": get_version(lcfg, "label_version", "l")}
    tcfg["models_path"] = str(_abs(tcfg["models_path"]))
    tcfg["signals_raw_path"] = str(_abs(tcfg["signals_raw_path"]))
    log = rolling_train_pipeline(_abs(dcfg["processed_path"]),
                                 _abs(lcfg["labels_path"]),
                                 tcfg, versions, args.start, args.end)
    log.write_csv(_abs("data/reports/train_log.csv"))


def _load_exposure(scfg: dict) -> pl.DataFrame | None:
    from src.utils.io import scan_parquet_dataset
    path = _abs(scfg["risk_exposure_path"])
    if not path.exists():
        logger.warning("risk exposure path missing: %s", path)
        return None
    cols = ["date", "stock_id", scfg.get("industry_col", "industry"),
            *scfg.get("style_cols", [])]
    lf = scan_parquet_dataset(path)
    avail = [c for c in cols if c in lf.collect_schema().names()]
    return lf.select(avail).collect()


def stage_signal(args) -> None:
    from src.signals.signal_processor import batch_process_all_signals
    scfg = load_config("signal_config.yaml")
    batch_process_all_signals(_abs(scfg["signals_raw_path"]),
                              _abs(scfg["signals_processed_path"]),
                              _load_exposure(scfg), scfg)


def _load_returns(horizons: list[int]) -> pl.DataFrame:
    """从原始面板构造评估用收益表：future_return_{h}d（仅评估用，非特征）。"""
    from src.utils.io import scan_parquet_dataset
    dcfg = load_config("data_config.yaml")
    cols = ["date", "stock_id", "close", "industry"]
    df = scan_parquet_dataset(_abs(dcfg["raw_path"])).select(cols).collect() \
        .sort(["stock_id", "date"])
    return df.with_columns([
        (pl.col("close").shift(-h).over("stock_id") / pl.col("close") - 1.0)
        .alias(f"future_return_{h}d") for h in horizons
    ]).drop("close")


def stage_evaluate(args) -> None:
    from src.signals.signal_evaluator import generate_signal_report
    from src.utils.io import scan_parquet_dataset
    scfg = load_config("signal_config.yaml")
    ecfg = scfg["evaluation"]
    signals = scan_parquet_dataset(_abs(scfg["signals_processed_path"])).collect()
    returns = _load_returns(ecfg.get("horizons", [1, 5, 10, 20]))
    generate_signal_report(signals, returns, ecfg, _abs(ecfg["reports_path"]),
                           _load_exposure(scfg))


def stage_ensemble(args) -> None:
    from src.signals.signal_ensemble import run_ensemble, save_ensemble
    from src.utils.io import scan_parquet_dataset
    ecfg = load_config("ensemble_config.yaml")
    signals = scan_parquet_dataset(_abs(ecfg["signals_processed_path"])).collect()
    returns = _load_returns([args.horizon])
    ens, weights = run_ensemble(signals, ecfg, returns, args.horizon)
    save_ensemble(ens, weights, _abs(ecfg["ensemble_output_path"]), ecfg["method"])


def stage_backtest(args) -> None:
    from src.portfolio.backtest import run_long_short_backtest, save_backtest_report
    from src.utils.io import scan_parquet_dataset
    pcfg = load_config("portfolio_config.yaml")
    dcfg = load_config("data_config.yaml")
    signal = pl.read_parquet(_abs(args.signal_file)).rename(
        {"ensemble_score": "signal_score"}, strict=False)
    returns = scan_parquet_dataset(_abs(dcfg["raw_path"])).select(
        ["date", "stock_id", "return"]).collect()
    result = run_long_short_backtest(signal, returns, pcfg)
    save_backtest_report(result, _abs("data/reports"), "long_short")


STAGES = {"preprocess": stage_preprocess, "label": stage_label,
          "train": stage_train, "signal": stage_signal,
          "evaluate": stage_evaluate, "ensemble": stage_ensemble,
          "backtest": stage_backtest}


def main() -> None:
    parser = argparse.ArgumentParser(description="tree signal library pipeline")
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--signal-file", default=None)
    args = parser.parse_args()
    logger.info("=== stage %s start ===", args.stage)
    STAGES[args.stage](args)
    logger.info("=== stage %s done ===", args.stage)


if __name__ == "__main__":
    main()
