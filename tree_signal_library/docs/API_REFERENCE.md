# tree_signal_library API 参考文档与原理说明

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 训练模式](#2-训练模式)
  - [2.1 Rolling（滚动窗口）训练](#21-rolling滚动窗口训练)
  - [2.2 Expanding（扩展窗口）训练](#22-expanding扩展窗口训练)
  - [2.3 两种模式对比](#23-两种模式对比)
- [3. Optuna 超参优化](#3-optuna-超参优化)
  - [3.1 原理](#31-原理)
  - [3.2 搜索空间配置](#32-搜索空间配置)
  - [3.3 使用方式](#33-使用方式)
- [4. 模块 API 接口说明](#4-模块-api-接口说明)
  - [4.1 数据模块 (src/data)](#41-数据模块-srcdata)
  - [4.2 模型模块 (src/models)](#42-模型模块-srcmodels)
  - [4.3 训练模块 (src/train)](#43-训练模块-srctrain)
  - [4.4 信号模块 (src/signals)](#44-信号模块-srcsignals)
  - [4.5 组合模块 (src/portfolio)](#45-组合模块-srcportfolio)
  - [4.6 工具模块 (src/utils)](#46-工具模块-srcutils)
- [5. 配置文件说明](#5-配置文件说明)
- [6. CLI 管线入口](#6-cli-管线入口)

---

## 1. 整体架构

```
raw data ─→ preprocess ─→ label ─→ train (rolling/expanding + optuna) ─→ signal
  ─→ evaluate ─→ ensemble ─→ backtest
```

全部表操作使用 **polars**（numpy 仅用于模型矩阵与横截面回归）。模型不是最终资产，标准化后的每日每股 alpha signal 才是。

核心防泄漏约束：
- 横截面操作全部 `group_by("date")`，禁止跨日统计
- 滚动/扩展切分按时间 + embargo（= max horizon 交易日）
- train/valid/test 边界 assert 硬约束
- 信号入库强制 schema + version 非空 + 唯一性校验

---

## 2. 训练模式

### 2.1 Rolling（滚动窗口）训练

**原理：**
Rolling Window 假设市场具有 regime shift 特征，近期数据比远期数据更具预测力。每个训练窗口使用固定长度的历史数据（如最近5年），窗口随时间向前滚动。

**窗口切分逻辑：**
```
Window 0: [train: 2015-2019] [valid: 2020] [test: 2021-Q1]
Window 1: [train: 2015.Q2-2019.Q2] [valid: 2020.Q2] [test: 2021-Q2]
  ...
```

**配置项（config/train_config.yaml → rolling）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_years` | int | 5 | 训练窗口长度（年） |
| `valid_years` | int | 1 | 验证窗口长度（年） |
| `test_months` | int | 3 | 测试窗口长度（月） |
| `step_months` | int | 3 | 窗口滚动步长（月） |
| `embargo_days` | int | 20 | 隔离交易日数 |

**接口：**

```python
from src.train.rolling_train import rolling_train_pipeline

log_df = rolling_train_pipeline(
    feature_path,   # 特征 parquet 路径
    label_path,     # 标签 parquet 路径
    train_cfg,      # train_config.yaml 内容
    versions,       # {feature_version, label_version}
    start,          # 回溯区间起始日期 "YYYY-MM-DD"
    end,            # 回溯区间终止日期 "YYYY-MM-DD"
    feature_cols=None  # 特征列名列表（None=自动检测 factor_*）
)
# 返回：pl.DataFrame，每窗口每模型一行训练日志
```

### 2.2 Expanding（扩展窗口）训练

**原理：**
Expanding Window 假设历史数据量越大模型泛化能力越强。训练起点固定不变，随时间推移训练集持续累积。

**窗口切分逻辑：**
```
Window 0: [train: 2015-2017] [valid: 2018] [test: 2019-Q1]
Window 1: [train: 2015-2018.Q2] [valid: 2018.Q2-2019.Q2] [test: 2019-Q2]
  ...（训练起点始终是 2015，训练集越来越大）
```

**配置项（config/train_config.yaml → expanding）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_train_years` | int | 3 | 最小训练年数 |
| `valid_years` | int | 1 | 验证窗口长度（年） |
| `test_months` | int | 3 | 测试窗口长度（月） |
| `step_months` | int | 3 | 窗口推进步长（月） |
| `embargo_days` | int | 20 | 隔离交易日数 |

**接口：**

```python
from src.train.expanding_train import expanding_train_pipeline

log_df = expanding_train_pipeline(
    feature_path,   # 特征 parquet 路径
    label_path,     # 标签 parquet 路径
    train_cfg,      # train_config.yaml 内容
    versions,       # {feature_version, label_version}
    start,          # 固定训练起始日期 "YYYY-MM-DD"
    end,            # 回溯区间终止日期 "YYYY-MM-DD"
    feature_cols=None
)
```

### 2.3 两种模式对比

| 维度 | Rolling | Expanding |
|------|---------|-----------|
| 训练集大小 | 固定（如5年） | 持续增长 |
| 假设 | 近期数据更相关 | 数据越多越好 |
| 适用场景 | 市场结构频繁变化 | 结构相对稳定 |
| 计算量 | 恒定 | 随窗口递增 |
| 过拟合风险 | 训练集小可能欠拟合 | 训练集大可能引入噪声 |
| 窗口 ID 前缀 | `w` (如 w000_20210101) | `e` (如 e000_20210101) |

**选择建议：**
- A股由于政策驱动和市场结构变化较快，通常 Rolling 表现更稳健
- 如果因子逻辑基于长期价值或财务基本面，Expanding 可能更优
- 实践中建议两种都跑一遍，在 ensemble 阶段进行融合对比

---

## 3. Optuna 超参优化

### 3.1 原理

**TPE（Tree-structured Parzen Estimator）算法：**

传统网格搜索和随机搜索不利用历史评估信息。TPE 是一种序贯模型优化（SMBO）方法：

1. **建模阶段**：将历史 trial 按目标值分为"好"和"差"两组（以分位数 γ 为阈值）
2. **两个密度**：分别用 Parzen 估计器（核密度估计）拟合好组 l(x) 和差组 g(x) 的参数分布
3. **采样策略**：选择使 Expected Improvement (EI) 最大的参数组合，即 l(x)/g(x) 最大的点
4. **迭代优化**：每次 trial 后更新密度模型，逐步聚焦到高质量区域

**相比随机搜索的优势：**
- 相同 trial 数量下通常能找到显著更优的参数
- 自动平衡 exploration（探索新区域）和 exploitation（精细调整）
- 支持条件参数和 early pruning

**目标函数：** Valid 集每日横截面 RankIC 均值（最大化）

**搜索策略：** 在第一个窗口上执行 Optuna 搜索，找到最优超参后在后续所有窗口中复用。
理由：每个窗口都做完整搜索计算量过大，且超参在窗口间通常变化不大。

### 3.2 搜索空间配置

在 `config/train_config.yaml` 的 `optuna.search_space` 下按模型 family 配置：

```yaml
optuna:
  enabled: true
  n_trials: 50
  timeout: null   # 秒，null=不限时间
  search_space:
    lightgbm:
      num_leaves:
        type: int
        low: 31
        high: 255
        step: 1
      learning_rate:
        type: float
        low: 0.005
        high: 0.1
        log: true   # 对数空间采样（适合跨数量级的参数）
      feature_fraction:
        type: float
        low: 0.4
        high: 0.9
```

**参数类型说明：**

| type | 必填字段 | 可选字段 | 说明 |
|------|----------|----------|------|
| `int` | `low`, `high` | `step`, `log` | 整数范围 |
| `float` | `low`, `high` | `step`, `log` | 浮点数范围 |
| `categorical` | `choices` | - | 离散选项 |

- `log: true`：在对数空间均匀采样，适合学习率、正则系数等跨数量级参数
- `step`：步长约束，如 `step: 10` 则只从 10 的倍数中选

### 3.3 使用方式

**方式1：通过 CLI 启用**
```bash
# 先在 train_config.yaml 中设置 optuna.enabled: true，然后
python run_pipeline.py --stage train --start 2015-01-01 --end 2024-12-31
```

**方式2：程序化调用**
```python
from src.train.hyperparam_search import hyperparam_search

best_params, trials_df = hyperparam_search(
    train_df=train_data,
    valid_df=valid_data,
    feature_cols=features,
    label_col="y_excess_return_5d",
    base_params=default_params,
    search_space={
        "num_leaves": {"type": "int", "low": 31, "high": 255},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
    },
    model_family="lightgbm",
    objective_type="regression",
    n_trials=50,
    seed=42,
)
```

---

## 4. 模块 API 接口说明

### 4.1 数据模块 (src/data)

#### `split_data.generate_rolling_windows`

```python
def generate_rolling_windows(
    start: str | date, end: str | date,
    train_years: int = 5, valid_years: int = 1,
    test_months: int = 3, step_months: int = 3,
    embargo_days: int = 20
) -> list[WindowSpec]
```

生成滚动窗口序列。所有切分严格按时间向前推进，禁止随机打乱。test 区间相邻窗口首尾相接、互不重叠。

#### `split_data.generate_expanding_windows`

```python
def generate_expanding_windows(
    start: str | date, end: str | date,
    valid_years: int = 1, test_months: int = 3,
    step_months: int = 3, embargo_days: int = 20,
    min_train_years: int = 3
) -> list[WindowSpec]
```

生成扩展窗口序列。训练起点固定为 start，窗口持续扩大。

#### `split_data.load_window_data`

```python
def load_window_data(
    window: WindowSpec, feature_path: str | Path,
    label_path: str | Path, embargo_days: int = 20
) -> dict[str, pl.DataFrame]
```

加载单窗口 train/valid/test 数据，自动实施 embargo。返回 `{"train": df, "valid": df, "test": df}`。

#### `split_data.WindowSpec`

```python
@dataclass(frozen=True)
class WindowSpec:
    window_id: str      # 窗口标识，如 "w000_20210101" 或 "e000_20210101"
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date
```

#### `preprocess.preprocess_panel`

原始数据预处理：缺失值填充、极端值处理、标准化。所有操作按 date 分组（横截面），禁止跨日。

#### `label_builder.build_labels_to_parquet`

构建标签：未来 h 日收益（超额/残差/行业中性/分组），末尾 h 天自动置 null。

### 4.2 模型模块 (src/models)

#### `model_factory.get_runner`

```python
def get_runner(model_family: str, objective_type: str) -> Callable
```

返回单窗口运行器函数。`model_family` ∈ {"lightgbm", "xgboost", "catboost"}，`objective_type` ∈ {"regression", "ranking"}。

**Runner 统一接口：**
```python
def run_xxx_one_window(
    train_df: pl.DataFrame, valid_df: pl.DataFrame, test_df: pl.DataFrame,
    feature_cols: list[str], label_col: str, params: dict,
    objective_type: str = "regression",
    eval_label_col: str | None = None, seed: int = 42
) -> dict
```

返回：
```python
{
    "model": 训练好的模型对象,
    "best_iteration": int,
    "ic_valid": float | None,        # valid IC 均值
    "rankic_valid": float | None,    # valid RankIC 均值
    "icir_valid": float | None,      # valid ICIR
    "rankicir_valid": float | None,  # valid RankICIR
    "predictions": pl.DataFrame,     # [date, stock_id, raw_score]
    "feature_importance": pl.DataFrame  # [feature, importance]
}
```

#### `model_factory.get_saver` / `get_loader` / `get_predictor`

```python
def get_saver(model_family: str) -> Callable      # save_model(model, path)
def get_loader(model_family: str) -> Callable      # load_model(path) -> model
def get_predictor(model_family: str) -> Callable   # predict(model, df, features) -> ndarray
```

#### `model_factory.build_model_name`

```python
def build_model_name(model_family, objective_type, label_type, horizon, model_version) -> str
```

模型命名规范：`{family}_{obj}_{label}_h{h}_v{ver}`，如 `lgbm_reg_excess_h5_v01`。

### 4.3 训练模块 (src/train)

#### `rolling_train.rolling_train_pipeline`

```python
def rolling_train_pipeline(
    feature_path: str | Path, label_path: str | Path,
    train_cfg: dict, versions: dict,
    start: str, end: str,
    feature_cols: list[str] | None = None
) -> pl.DataFrame
```

滚动训练主入口。若 `train_cfg["optuna"]["enabled"] = True`，在首窗口执行 Optuna 调参。

#### `expanding_train.expanding_train_pipeline`

```python
def expanding_train_pipeline(
    feature_path: str | Path, label_path: str | Path,
    train_cfg: dict, versions: dict,
    start: str, end: str,
    feature_cols: list[str] | None = None
) -> pl.DataFrame
```

扩展训练主入口。接口与 rolling 完全一致，仅窗口切分策略不同。

#### `train_one_window.train_one_model_one_window`

```python
def train_one_model_one_window(
    data: dict[str, pl.DataFrame],  # {"train", "valid", "test"}
    window: WindowSpec, spec: dict,
    feature_cols: list[str], params: dict,
    versions: dict, models_root: str | Path, seed: int = 42
) -> tuple[dict, dict]
```

训练单个模型单个窗口并保存 checkpoint。返回 `(result, meta)`。

#### `hyperparam_search.hyperparam_search`

```python
def hyperparam_search(
    train_df: pl.DataFrame, valid_df: pl.DataFrame,
    feature_cols: list[str], label_col: str,
    base_params: dict, search_space: dict[str, dict],
    model_family: str, objective_type: str,
    n_trials: int = 50, eval_label_col: str | None = None,
    seed: int = 42, timeout: int | None = None,
    direction: str = "maximize"
) -> tuple[dict, pl.DataFrame]
```

使用 Optuna TPE 搜索超参。返回 `(best_params, trials_df)`。

**search_space 格式：**
```python
{
    "param_name": {"type": "int|float|categorical", "low": ..., "high": ..., ...},
    ...
}
```

#### `predict.save_prediction_signal`

```python
def save_prediction_signal(
    predictions: pl.DataFrame,  # [date, stock_id, raw_score]
    meta: dict, signals_root: str | Path
) -> list[Path]
```

测试期预测 → 统一 schema → 校验 → 落库。

### 4.4 信号模块 (src/signals)

#### `signal_processor.batch_process_all_signals`

对所有原始信号执行：去极值 → 标准化 → 行业/风格中性化 → 落库。

#### `signal_evaluator.generate_signal_report`

生成信号评估报告：IC/RankIC 时序、分组收益、多空收益、换手率。

#### `signal_ensemble.run_ensemble`

多模型信号融合（等权 / IC 加权 / 优化加权），返回融合信号和权重。

### 4.5 组合模块 (src/portfolio)

#### `backtest.run_long_short_backtest`

多空回测：按信号打分分组，多头 Top 组空头 Bottom 组，计算每日收益和绩效指标。

### 4.6 工具模块 (src/utils)

#### `metrics.daily_ic` / `daily_rank_ic`

```python
def daily_ic(df, signal_col, return_col, date_col="date") -> pl.DataFrame
def daily_rank_ic(df, signal_col, return_col, date_col="date") -> pl.DataFrame
```

每日横截面 Pearson IC / Spearman RankIC。返回 `[date, ic]` 或 `[date, rank_ic]`。

#### `metrics.ic_summary`

```python
def ic_summary(ic_df, ic_col="ic") -> dict
```

返回：`{ic_mean, ic_std, icir, ic_tstat, ic_positive_ratio, n_days}`

#### `metrics.annualize`

```python
def annualize(returns: pl.Series, periods_per_year=252) -> dict
```

返回：`{ann_return, ann_vol, sharpe, max_drawdown, win_rate}`

#### `config.load_config` / `merge_config`

```python
def load_config(path: str | Path) -> dict     # 加载 YAML 配置
def merge_config(base: dict, override: dict | None) -> dict  # 深度合并
```

#### `seed.set_seed` / `seed_params`

```python
def set_seed(seed: int = 42) -> None
def seed_params(model_family: str, seed: int = 42) -> dict
```

统一随机种子管理。`seed_params` 返回各模型库的种子参数字典。

---

## 5. 配置文件说明

| 文件 | 说明 |
|------|------|
| `data_config.yaml` | 原始数据路径、处理参数 |
| `label_config.yaml` | 标签构建参数（horizon、类型） |
| `train_config.yaml` | 训练模式、滚动/扩展参数、Optuna 设置、模型清单 |
| `lgbm_config.yaml` | LightGBM 默认超参（regression + ranking） |
| `xgb_config.yaml` | XGBoost 默认超参 |
| `cat_config.yaml` | CatBoost 默认超参 |
| `signal_config.yaml` | 信号处理与评估配置 |
| `ensemble_config.yaml` | 信号融合配置 |
| `portfolio_config.yaml` | 组合回测配置 |

### train_config.yaml 完整结构

```yaml
train_mode: rolling | expanding  # 训练模式

rolling:
  train_years: 5
  valid_years: 1
  test_months: 3
  step_months: 3
  embargo_days: 20

expanding:
  min_train_years: 3
  valid_years: 1
  test_months: 3
  step_months: 3
  embargo_days: 20

optuna:
  enabled: true | false
  n_trials: 50
  timeout: null
  search_space:
    lightgbm: { ... }
    xgboost: { ... }
    catboost: { ... }

model_specs:
  - model_family: lightgbm
    objective_type: regression
    label_type: excess_return
    horizon: 5
    model_version: v01

seed: 42
```

---

## 6. CLI 管线入口

```bash
# 滚动训练（默认）
python run_pipeline.py --stage train --start 2015-01-01 --end 2024-12-31

# 扩展训练
python run_pipeline.py --stage train --start 2015-01-01 --end 2024-12-31 --train-mode expanding

# 启用 Optuna 调参（需先在 train_config.yaml 设 optuna.enabled: true）
python run_pipeline.py --stage train --start 2015-01-01 --end 2024-12-31

# 其他阶段
python run_pipeline.py --stage preprocess --start 2015-01-01 --end 2024-12-31
python run_pipeline.py --stage label     --start 2015-01-01 --end 2024-12-31
python run_pipeline.py --stage signal
python run_pipeline.py --stage evaluate
python run_pipeline.py --stage ensemble  --horizon 5
python run_pipeline.py --stage backtest  --signal-file data/signals/ensemble/ensemble_equal_rank.parquet
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--stage` | 必选，运行阶段 |
| `--start` | 起始日期 YYYY-MM-DD |
| `--end` | 终止日期 YYYY-MM-DD |
| `--train-mode` | 训练模式 rolling/expanding（覆盖 yaml 配置） |
| `--horizon` | 预测 horizon（ensemble 阶段用） |
| `--signal-file` | 信号文件路径（backtest 阶段用） |
