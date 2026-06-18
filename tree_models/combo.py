"""TreeModelsCombo: combine multiple tree-based estimators into a single model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted


def _default_classifiers() -> List[BaseEstimator]:
    """Return the default list of tree classifiers."""
    return [
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(n_estimators=100, random_state=0),
        ExtraTreesClassifier(n_estimators=100, random_state=0),
        GradientBoostingClassifier(n_estimators=100, random_state=0),
    ]


def _default_regressors() -> List[BaseEstimator]:
    """Return the default list of tree regressors."""
    return [
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=100, random_state=0),
        ExtraTreesRegressor(n_estimators=100, random_state=0),
        GradientBoostingRegressor(n_estimators=100, random_state=0),
    ]


class TreeModelsCombo(BaseEstimator):
    """Combine multiple tree-based estimators via voting (classification) or
    averaging (regression).

    Parameters
    ----------
    estimators : list of estimators, optional
        The tree-based models to combine.  When *None* a sensible default set
        of scikit-learn tree models is used automatically – Decision Tree,
        Random Forest, Extra Trees and Gradient Boosting.
    task : {'auto', 'classification', 'regression'}, default='auto'
        Whether to treat the problem as classification or regression.  When
        ``'auto'`` the task is inferred from the type of the first estimator
        supplied (or defaults to ``'classification'`` when no estimators are
        given).
    weights : array-like of shape (n_estimators,), optional
        Weights applied to each estimator's predictions.  ``None`` means equal
        weights.

    Attributes
    ----------
    estimators_ : list of fitted estimators
        The fitted copies of all estimators.
    task_ : str
        Resolved task type (``'classification'`` or ``'regression'``).
    classes_ : ndarray of shape (n_classes,)
        The class labels (only present for classification tasks).
    """

    def __init__(
        self,
        estimators: Optional[List[BaseEstimator]] = None,
        task: str = "auto",
        weights: Optional[Union[List[float], np.ndarray]] = None,
    ) -> None:
        self.estimators = estimators
        self.task = task
        self.weights = weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_task(self, first_estimator: Optional[BaseEstimator]) -> str:
        if self.task in ("classification", "regression"):
            return self.task
        if self.task == "auto":
            if first_estimator is not None:
                if isinstance(first_estimator, RegressorMixin):
                    return "regression"
                return "classification"
            return "classification"
        raise ValueError(
            f"Unknown task '{self.task}'. Choose 'auto', 'classification', or 'regression'."
        )

    def _get_estimators(self, task: str) -> List[BaseEstimator]:
        if self.estimators is not None:
            return [clone(est) for est in self.estimators]
        if task == "regression":
            return _default_regressors()
        return _default_classifiers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit all estimators on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        first = (self.estimators or [None])[0]
        self.task_ = self._resolve_task(first)
        estimators = self._get_estimators(self.task_)

        self.estimators_ = []
        for est in estimators:
            self.estimators_.append(est.fit(X, y))

        if self.task_ == "classification":
            self.classes_ = self.estimators_[0].classes_

        return self

    def predict(self, X):
        """Predict class labels (classification) or target values (regression).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "estimators_")
        if self.task_ == "classification":
            return self._predict_classification(X)
        return self._predict_regression(X)

    def predict_proba(self, X):
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, "estimators_")
        if self.task_ != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")

        probas = [est.predict_proba(X) for est in self.estimators_]
        weights = self._normalized_weights()
        weighted = np.average(np.stack(probas, axis=0), axis=0, weights=weights)
        return weighted

    def score(self, X, y):
        """Return the mean accuracy (classification) or R² (regression).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
        """
        check_is_fitted(self, "estimators_")
        if self.task_ == "classification":
            return ClassifierMixin.score(self, X, y)
        return RegressorMixin.score(self, X, y)

    # ------------------------------------------------------------------
    # Private prediction helpers
    # ------------------------------------------------------------------

    def _normalized_weights(self) -> Optional[np.ndarray]:
        if self.weights is None:
            return None
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    def _predict_classification(self, X) -> np.ndarray:
        """Majority vote across all estimators."""
        predictions = np.array([est.predict(X) for est in self.estimators_])
        weights = self._normalized_weights()

        n_samples = predictions.shape[1]
        result = np.empty(n_samples, dtype=self.classes_.dtype)

        for i in range(n_samples):
            votes = predictions[:, i]
            if weights is None:
                # Unweighted majority vote
                unique, counts = np.unique(votes, return_counts=True)
                result[i] = unique[np.argmax(counts)]
            else:
                # Weighted vote: accumulate weights per class
                vote_weights: Dict[Any, float] = {}
                for vote, w in zip(votes, weights):
                    vote_weights[vote] = vote_weights.get(vote, 0.0) + w
                result[i] = max(vote_weights, key=vote_weights.__getitem__)

        return result

    def _predict_regression(self, X) -> np.ndarray:
        """Weighted average across all estimators."""
        predictions = np.array([est.predict(X) for est in self.estimators_])
        weights = self._normalized_weights()
        return np.average(predictions, axis=0, weights=weights)
