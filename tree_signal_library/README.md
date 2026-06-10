# tree_signal_library — 树模型多模型 Alpha 信号库

A股日频面板（~5000 股 x 1000–2000 因子）的工程化树模型信号生产线：
LightGBM / XGBoost / CatBoost 滚动/扩展训练 -> Optuna 超参优化 ->
样本外预测 -> 统一信号库 -> 标准化/中性化 -> 评估 -> 多模型融合 ->
中证1000指数增强 / 多空回测。

核心约定：所有表操作使用 **polars**（numpy 仅用于模型矩阵与横截面回归）；
模型不是最终资产，标准化后的每日每股 alpha signal 才是。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 滚动训练（默认）
python run_pipeline.py --stage train --start 2015-01-01 --end 2024-12-31

# 扩展训练
python run_pipeline.py --stage train --start 2015-01-01 --end 2024-12-31 --train-mode expanding

# 其他阶段
python run_pipeline.py --stage preprocess --start 2015-01-01 --end 2024-12-31
python run_pipeline.py --stage label     --start 2015-01-01 --end 2024-12-31
python run_pipeline.py --stage signal
python run_pipeline.py --stage evaluate
python run_pipeline.py --stage ensemble  --horizon 5
python run_pipeline.py --stage backtest  --signal-file data/signals/ensemble/ensemble_equal_rank.parquet
```

启用 Optuna 超参调参：在 `config/train_config.yaml` 中设置 `optuna.enabled: true`。
详细 API 文档见 [docs/API_REFERENCE.md](docs/API_REFERENCE.md)。

## 测试

```bash
python -m pytest tests -q   # 合成面板数据，覆盖对齐/无泄漏/全模块闭环
```

## 目录结构

```
config/      # 全部 yaml 配置（数据/标签/三模型超参/滚动训练/信号/融合/组合）
data/        # raw / processed / labels / signals / models / reports
src/data     # load_data, preprocess, label_builder, split_data
src/models   # lgbm_model, xgb_model, cat_model, rank_utils, model_factory
src/train    # train_one_window, rolling_train, hyperparam_search, predict
src/signals  # signal_writer, signal_processor, signal_neutralizer,
             # signal_evaluator, signal_ensemble
src/portfolio# optimizer(cvxpy), backtest, exposure, risk_model
src/utils    # logger, io, metrics, seed, config
notebooks/   # 01 数据检查 / 02 单模型评估 / 03 信号融合 / 04 组合回测
run_pipeline.py
```

## 防泄漏硬约束（代码中强制）

- 横截面操作全部 `group_by("date")`，禁止跨日统计；
- 特征处理对 `y_*` / `future_*` 列静态断言隔离；
- 滚动切分按时间 + embargo（= max horizon 交易日），train/valid/test 边界 assert；
- ranking group 统一由 rank_utils 按 date 构造，强制排序断言；
- 标签末尾 h 天自动置 null；
- 融合动态权重只用 t-h-1 之前已实现的 IC，权重历史含 `info_as_of_date` 审计；
- 信号入库强制 23 字段 schema + model/feature/label_version 非空 + 唯一性校验。

## 信号命名

`{family}_{obj}_{label}_h{horizon}_v{ver}`，如第一阶段两个信号：
`lgbm_reg_excess_h5_v01`、`lgbm_reg_residual_h5_v01`（见 config/train_config.yaml）。
