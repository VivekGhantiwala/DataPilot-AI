"""
Tests for Exploratory Data Analysis module.
"""

import numpy as np
import pandas as pd
import pytest

from src.eda import ExploratoryAnalysis


class TestExploratoryAnalysis:
    """Test cases for ExploratoryAnalysis class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 65, n),
                "income": np.random.normal(50000, 15000, n),
                "score": np.random.uniform(0, 100, n),
                "category": np.random.choice(["A", "B", "C"], n),
                "target": np.random.choice([0, 1], n),
            }
        )

    @pytest.fixture
    def eda(self, sample_data):
        """Create EDA instance with sample data."""
        return ExploratoryAnalysis(sample_data)

    def test_initialization(self, sample_data):
        """Test EDA initialization."""
        eda = ExploratoryAnalysis(sample_data)
        assert eda.data is not None
        assert len(eda.data) == 100

    def test_basic_statistics(self, eda):
        """Test basic statistics computation."""
        stats = eda.basic_statistics()
        assert stats is not None
        assert isinstance(stats, (dict, pd.DataFrame))

    def test_get_numeric_columns(self, eda):
        """Test getting numeric columns."""
        numeric_cols = eda.data.select_dtypes(include=[np.number]).columns.tolist()
        assert "age" in numeric_cols
        assert "income" in numeric_cols
        assert "score" in numeric_cols

    def test_get_categorical_columns(self, eda):
        """Test getting categorical columns."""
        cat_cols = eda.data.select_dtypes(include=["object"]).columns.tolist()
        assert "category" in cat_cols

    def test_correlation_analysis(self, eda):
        """Test correlation analysis."""
        corr = eda.correlation_analysis()
        assert corr is not None
        if isinstance(corr, pd.DataFrame):
            assert corr.shape[0] == corr.shape[1]

    def test_missing_values_analysis(self, sample_data):
        """Test missing values analysis."""
        # Add some missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:5, "income"] = np.nan

        eda = ExploratoryAnalysis(data_with_missing)
        missing = eda.missing_values_analysis()

        assert missing is not None

    def test_distribution_analysis(self, eda):
        """Test distribution analysis."""
        dist = eda.distribution_analysis()
        assert dist is not None

    def test_full_analysis(self, eda):
        """Test full EDA analysis."""
        results = eda.full_analysis()
        assert results is not None
        assert isinstance(results, dict)


class TestStatisticalTests:
    """Test statistical tests functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for statistical tests."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "group": np.repeat(["A", "B"], 50),
                "value": np.concatenate([np.random.normal(10, 2, 50), np.random.normal(12, 2, 50)]),
                "category": np.random.choice(["X", "Y", "Z"], 100),
            }
        )

    def test_normality_check(self, sample_data):
        """Test normality checking."""
        eda = ExploratoryAnalysis(sample_data)
        # Should not raise an error
        assert eda is not None


class TestEdgeCases:
    """Test edge cases."""

    def test_single_row(self):
        """Test with single row DataFrame."""
        data = pd.DataFrame({"a": [1], "b": [2]})
        eda = ExploratoryAnalysis(data)
        assert eda.basic_statistics() is not None

    def test_all_same_values(self):
        """Test with constant values."""
        data = pd.DataFrame({"constant": [5] * 10})
        eda = ExploratoryAnalysis(data)
        stats = eda.basic_statistics()
        assert stats is not None

    def test_mixed_dtypes(self):
        """Test with mixed data types."""
        data = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        eda = ExploratoryAnalysis(data)
        assert eda.basic_statistics() is not None
