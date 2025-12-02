"""
Tests for Data Preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "numeric_1": [1, 2, np.nan, 4, 5, 100],  # Has missing and outlier
                "numeric_2": [10, 20, 30, 40, 50, 60],
                "category": ["A", "B", "A", "C", "B", "A"],
                "text": ["hello", "world", "test", "data", "hello", "world"],
            }
        )

    @pytest.fixture
    def preprocessor(self, sample_data):
        """Create preprocessor with sample data."""
        prep = DataPreprocessor()
        prep.load_dataframe(sample_data)
        return prep

    def test_load_dataframe(self, sample_data):
        """Test loading DataFrame into preprocessor."""
        prep = DataPreprocessor()
        prep.load_dataframe(sample_data)
        assert prep.data is not None
        assert len(prep.data) == 6
        assert len(prep.data.columns) == 4

    def test_get_data_info(self, preprocessor):
        """Test getting data information."""
        info = preprocessor.get_data_info()
        assert "shape" in info
        assert "columns" in info
        assert "dtypes" in info
        assert info["shape"] == (6, 4)

    def test_handle_missing_values_mean(self, preprocessor):
        """Test handling missing values with mean strategy."""
        preprocessor.handle_missing_values(strategy="auto", numerical_strategy="mean")
        data = preprocessor.get_clean_data()
        # Check no missing values remain in numeric columns
        assert data["numeric_1"].isnull().sum() == 0

    def test_handle_missing_values_median(self, preprocessor):
        """Test handling missing values with median strategy."""
        preprocessor.handle_missing_values(strategy="auto", numerical_strategy="median")
        data = preprocessor.get_clean_data()
        assert data["numeric_1"].isnull().sum() == 0

    def test_remove_duplicates(self, sample_data):
        """Test duplicate removal."""
        # Add duplicate rows
        data_with_dups = pd.concat([sample_data, sample_data.iloc[[0]]])
        prep = DataPreprocessor()
        prep.load_dataframe(data_with_dups)

        original_len = len(prep.data)
        prep.remove_duplicates()

        assert len(prep.get_clean_data()) < original_len

    def test_handle_outliers_clip(self, preprocessor):
        """Test outlier handling with clip method."""
        preprocessor.handle_outliers(method="clip")
        data = preprocessor.get_clean_data()
        # The extreme value should be clipped
        assert data["numeric_1"].max() < 100

    def test_encode_categorical_label(self, preprocessor):
        """Test label encoding of categorical columns."""
        preprocessor.encode_categorical(method="label")
        data = preprocessor.get_clean_data()
        # Categorical columns should be numeric
        assert data["category"].dtype in [np.int64, np.int32, np.float64]

    def test_scale_features_standard(self, preprocessor):
        """Test standard scaling."""
        preprocessor.handle_missing_values(strategy="auto")
        preprocessor.scale_features(method="standard")
        data = preprocessor.get_clean_data()
        # Scaled data should have mean close to 0
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(data[col].mean()) < 1

    def test_scale_features_minmax(self, preprocessor):
        """Test min-max scaling."""
        preprocessor.handle_missing_values(strategy="auto")
        preprocessor.scale_features(method="minmax")
        data = preprocessor.get_clean_data()
        # Data should be between 0 and 1
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert data[col].min() >= -0.01
            assert data[col].max() <= 1.01

    def test_preprocess_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        prep = DataPreprocessor()
        prep.load_dataframe(sample_data)

        result = prep.preprocess_pipeline(
            handle_missing=True, remove_dups=True, handle_outliers_flag=True, scale=False, encode=False
        )

        assert result is not None
        assert result.isnull().sum().sum() == 0

    def test_get_preprocessing_summary(self, preprocessor):
        """Test getting preprocessing summary."""
        preprocessor.handle_missing_values(strategy="auto")
        summary = preprocessor.get_preprocessing_summary()

        assert isinstance(summary, dict)
        assert "original_shape" in summary or "shape" in summary or len(summary) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        prep = DataPreprocessor()
        empty_df = pd.DataFrame()
        prep.load_dataframe(empty_df)
        assert prep.data is not None

    def test_all_missing_column(self):
        """Test handling column with all missing values."""
        prep = DataPreprocessor()
        data = pd.DataFrame({"all_missing": [np.nan, np.nan, np.nan], "valid": [1, 2, 3]})
        prep.load_dataframe(data)
        prep.handle_missing_values(strategy="auto")
        # Should handle gracefully
        assert prep.get_clean_data() is not None

    def test_single_column(self):
        """Test with single column DataFrame."""
        prep = DataPreprocessor()
        data = pd.DataFrame({"single": [1, 2, 3, 4, 5]})
        prep.load_dataframe(data)
        prep.scale_features(method="standard")
        assert len(prep.get_clean_data().columns) == 1
