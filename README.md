# tree_models

A Python package that combines multiple tree-based machine learning models into a
single **combo** estimator.  Predictions are produced by majority vote
(classification) or weighted averaging (regression), giving you an easy ensemble
without writing boilerplate.

## Installation

```bash
pip install -e .
```

## Quick start

### Classification

```python
from tree_models import TreeModelsCombo

combo = TreeModelsCombo(task="classification")
combo.fit(X_train, y_train)
print(combo.score(X_test, y_test))
```

### Regression

```python
from tree_models import TreeModelsCombo

combo = TreeModelsCombo(task="regression")
combo.fit(X_train, y_train)
print(combo.score(X_test, y_test))
```

### Custom estimators

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tree_models import TreeModelsCombo

combo = TreeModelsCombo(
    estimators=[
        RandomForestClassifier(n_estimators=200, random_state=0),
        GradientBoostingClassifier(n_estimators=200, random_state=0),
    ],
    weights=[2.0, 1.0],   # give Random Forest twice the vote weight
)
combo.fit(X_train, y_train)
preds = combo.predict(X_test)
proba = combo.predict_proba(X_test)
```

## API

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Fit all estimators |
| `predict(X)` | Majority vote (clf) / weighted mean (reg) |
| `predict_proba(X)` | Weighted-average class probabilities (clf only) |
| `score(X, y)` | Accuracy (clf) / R² (reg) |

## Default estimators

When no `estimators` list is provided, `TreeModelsCombo` uses:

* `DecisionTreeClassifier` / `DecisionTreeRegressor`
* `RandomForestClassifier` / `RandomForestRegressor`
* `ExtraTreesClassifier` / `ExtraTreesRegressor`
* `GradientBoostingClassifier` / `GradientBoostingRegressor`

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```