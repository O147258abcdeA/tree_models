"""Tests for TreeModelsCombo."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from tree_models import TreeModelsCombo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return X, y


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def test_default_estimators_fit_predict(self, clf_data):
        X, y = clf_data
        combo = TreeModelsCombo(task="classification")
        combo.fit(X, y)
        preds = combo.predict(X)
        assert preds.shape == (len(y),)
        assert set(preds).issubset(set(y))

    def test_custom_estimators(self, clf_data):
        X, y = clf_data
        estimators = [
            DecisionTreeClassifier(random_state=0),
            RandomForestClassifier(n_estimators=10, random_state=0),
        ]
        combo = TreeModelsCombo(estimators=estimators, task="classification")
        combo.fit(X, y)
        preds = combo.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba(self, clf_data):
        X, y = clf_data
        combo = TreeModelsCombo(task="classification")
        combo.fit(X, y)
        proba = combo.predict_proba(X)
        assert proba.shape == (len(y), len(np.unique(y)))
        # Probabilities should sum to 1 for each sample
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)), atol=1e-6)

    def test_weighted_vote(self, clf_data):
        X, y = clf_data
        estimators = [
            DecisionTreeClassifier(random_state=0),
            DecisionTreeClassifier(random_state=1),
        ]
        combo = TreeModelsCombo(
            estimators=estimators, task="classification", weights=[3.0, 1.0]
        )
        combo.fit(X, y)
        preds = combo.predict(X)
        assert preds.shape == (len(y),)

    def test_score(self, clf_data):
        X, y = clf_data
        combo = TreeModelsCombo(task="classification")
        combo.fit(X, y)
        score = combo.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_classes_attribute(self, clf_data):
        X, y = clf_data
        combo = TreeModelsCombo(task="classification")
        combo.fit(X, y)
        assert hasattr(combo, "classes_")
        np.testing.assert_array_equal(combo.classes_, np.unique(y))

    def test_auto_task_inferred_from_classifier(self, clf_data):
        X, y = clf_data
        estimators = [DecisionTreeClassifier(random_state=0)]
        combo = TreeModelsCombo(estimators=estimators)  # task='auto'
        combo.fit(X, y)
        assert combo.task_ == "classification"


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestRegression:
    def test_default_estimators_fit_predict(self, reg_data):
        X, y = reg_data
        combo = TreeModelsCombo(task="regression")
        combo.fit(X, y)
        preds = combo.predict(X)
        assert preds.shape == (len(y),)

    def test_custom_estimators(self, reg_data):
        X, y = reg_data
        estimators = [
            DecisionTreeRegressor(random_state=0),
            DecisionTreeRegressor(max_depth=3, random_state=1),
        ]
        combo = TreeModelsCombo(estimators=estimators, task="regression")
        combo.fit(X, y)
        preds = combo.predict(X)
        assert preds.shape == (len(y),)

    def test_weighted_average(self, reg_data):
        X, y = reg_data
        estimators = [
            DecisionTreeRegressor(random_state=0),
            DecisionTreeRegressor(random_state=1),
        ]
        combo = TreeModelsCombo(
            estimators=estimators, task="regression", weights=[2.0, 1.0]
        )
        combo.fit(X, y)
        preds = combo.predict(X)
        assert preds.shape == (len(y),)

    def test_score(self, reg_data):
        X, y = reg_data
        combo = TreeModelsCombo(task="regression")
        combo.fit(X, y)
        # On training data, R² should be near 1.0 for tree models
        score = combo.score(X, y)
        assert score > 0.9

    def test_auto_task_inferred_from_regressor(self, reg_data):
        X, y = reg_data
        estimators = [DecisionTreeRegressor(random_state=0)]
        combo = TreeModelsCombo(estimators=estimators)  # task='auto'
        combo.fit(X, y)
        assert combo.task_ == "regression"

    def test_predict_proba_raises_for_regression(self, reg_data):
        X, y = reg_data
        combo = TreeModelsCombo(task="regression")
        combo.fit(X, y)
        with pytest.raises(ValueError, match="predict_proba"):
            combo.predict_proba(X)


# ---------------------------------------------------------------------------
# Edge-case / error tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_not_fitted_raises(self, clf_data):
        X, _ = clf_data
        combo = TreeModelsCombo()
        with pytest.raises(Exception):
            combo.predict(X)

    def test_invalid_task_raises(self, clf_data):
        X, y = clf_data
        combo = TreeModelsCombo(task="unknown")
        with pytest.raises(ValueError, match="Unknown task"):
            combo.fit(X, y)

    def test_default_task_is_classification_when_no_estimators(self, clf_data):
        X, y = clf_data
        combo = TreeModelsCombo()  # no estimators, task='auto'
        combo.fit(X, y)
        assert combo.task_ == "classification"
