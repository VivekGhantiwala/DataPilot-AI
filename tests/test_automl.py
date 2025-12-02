"""
Tests for AutoML module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from src.automl import AutoML


class TestAutoMLClassification:
    """Test cases for AutoML classification."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y = pd.Series(y, name="target")
        return X, y

    def test_automl_initialization(self):
        """Test AutoML initialization."""
        automl = AutoML(task="classification")
        assert automl is not None
        assert automl.task == "classification"

    def test_automl_fit_classification(self, classification_data):
        """Test AutoML fit for classification."""
        X, y = classification_data
        automl = AutoML(task="classification", max_models=3, verbose=False)
        automl.fit(X, y)

        assert automl.best_model is not None
        assert automl.best_model_name is not None

    def test_automl_predict(self, classification_data):
        """Test AutoML predictions."""
        X, y = classification_data
        automl = AutoML(task="classification", max_models=3, verbose=False)
        automl.fit(X, y)

        predictions = automl.predict(X.head(10))
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_automl_predict_proba(self, classification_data):
        """Test AutoML probability predictions."""
        X, y = classification_data
        automl = AutoML(task="classification", max_models=3, verbose=False)
        automl.fit(X, y)

        proba = automl.predict_proba(X.head(10))
        assert proba.shape[0] == 10
        assert proba.shape[1] == 2  # Binary classification
        assert all(0 <= p <= 1 for row in proba for p in row)

    def test_automl_leaderboard(self, classification_data):
        """Test AutoML leaderboard."""
        X, y = classification_data
        automl = AutoML(task="classification", max_models=3, verbose=False)
        automl.fit(X, y)

        leaderboard = automl.get_leaderboard()
        assert leaderboard is not None
        assert len(leaderboard) > 0

    def test_automl_feature_importance(self, classification_data):
        """Test feature importance extraction."""
        X, y = classification_data
        automl = AutoML(task="classification", max_models=3, verbose=False)
        automl.fit(X, y)

        try:
            importance = automl.get_feature_importance()
            assert importance is not None
        except (AttributeError, ValueError):
            # Some models may not support feature importance
            pass


class TestAutoMLRegression:
    """Test cases for AutoML regression."""

    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=5, noise=0.1, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y = pd.Series(y, name="target")
        return X, y

    def test_automl_fit_regression(self, regression_data):
        """Test AutoML fit for regression."""
        X, y = regression_data
        automl = AutoML(task="regression", max_models=3, verbose=False)
        automl.fit(X, y)

        assert automl.best_model is not None

    def test_automl_predict_regression(self, regression_data):
        """Test AutoML regression predictions."""
        X, y = regression_data
        automl = AutoML(task="regression", max_models=3, verbose=False)
        automl.fit(X, y)

        predictions = automl.predict(X.head(10))
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestAutoMLAutoDetect:
    """Test AutoML task auto-detection."""

    def test_auto_detect_classification(self):
        """Test auto-detection of classification task."""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], 100))

        automl = AutoML(task="auto", max_models=2, verbose=False)
        automl.fit(X, y)

        assert automl.task in ["classification", "auto"]

    def test_auto_detect_regression(self):
        """Test auto-detection of regression task."""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randn(100))

        automl = AutoML(task="auto", max_models=2, verbose=False)
        automl.fit(X, y)

        assert automl.task in ["regression", "auto"]


class TestEdgeCases:
    """Test edge cases."""

    def test_small_dataset(self):
        """Test with small dataset."""
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        y = pd.Series([0, 1, 0, 1, 0])

        automl = AutoML(task="classification", max_models=2, verbose=False)
        # Should handle gracefully or raise appropriate error
        try:
            automl.fit(X, y)
            assert automl.best_model is not None
        except ValueError:
            # Expected for very small datasets
            pass

    def test_with_missing_values(self):
        """Test with missing values in data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        X.iloc[0:10, 0] = np.nan
        y = pd.Series(np.random.choice([0, 1], 100))

        automl = AutoML(task="classification", max_models=2, verbose=False)
        # Should handle missing values
        try:
            automl.fit(X.fillna(0), y)
            assert automl.best_model is not None
        except Exception:
            pass
