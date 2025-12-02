"""
Exploratory Data Analysis (EDA) Module
Provides comprehensive data exploration and statistical analysis.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class ExploratoryAnalysis:
    """
    A comprehensive class for Exploratory Data Analysis.

    Attributes:
        data (pd.DataFrame): The dataset to analyze
        numerical_columns (List[str]): List of numerical column names
        categorical_columns (List[str]): List of categorical column names
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the ExploratoryAnalysis with a dataset.

        Args:
            data (pd.DataFrame): The dataset to analyze
        """
        self.data = data.copy()
        self._identify_column_types()

    def _identify_column_types(self):
        """Automatically identify numerical and categorical columns."""
        self.numerical_columns = self.data.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        self.categorical_columns = self.data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Returns:
            Dict containing basic dataset information
        """
        info = {
            "rows": self.data.shape[0],
            "columns": self.data.shape[1],
            "total_cells": self.data.shape[0] * self.data.shape[1],
            "missing_cells": self.data.isnull().sum().sum(),
            "missing_percentage": (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100),
            "duplicate_rows": self.data.duplicated().sum(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "numerical_columns": len(self.numerical_columns),
            "categorical_columns": len(self.categorical_columns),
        }
        return info

    def generate_summary(self, include_all: bool = True) -> pd.DataFrame:
        """
        Generate a comprehensive statistical summary of the dataset.

        Args:
            include_all (bool): Whether to include all statistics

        Returns:
            pd.DataFrame: Statistical summary
        """
        summary_stats = []

        for col in self.data.columns:
            col_stats = {
                "column": col,
                "dtype": str(self.data[col].dtype),
                "non_null_count": self.data[col].count(),
                "null_count": self.data[col].isnull().sum(),
                "null_percentage": self.data[col].isnull().sum() / len(self.data) * 100,
                "unique_count": self.data[col].nunique(),
                "unique_percentage": self.data[col].nunique() / len(self.data) * 100,
            }

            if col in self.numerical_columns:
                col_stats.update(
                    {
                        "mean": self.data[col].mean(),
                        "std": self.data[col].std(),
                        "min": self.data[col].min(),
                        "q1": self.data[col].quantile(0.25),
                        "median": self.data[col].median(),
                        "q3": self.data[col].quantile(0.75),
                        "max": self.data[col].max(),
                        "iqr": self.data[col].quantile(0.75) - self.data[col].quantile(0.25),
                        "skewness": self.data[col].skew(),
                        "kurtosis": self.data[col].kurtosis(),
                    }
                )
            else:
                col_stats.update(
                    {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "q1": None,
                        "median": None,
                        "q3": None,
                        "max": None,
                        "iqr": None,
                        "skewness": None,
                        "kurtosis": None,
                        "mode": self.data[col].mode().iloc[0] if len(self.data[col].mode()) > 0 else None,
                        "top_value_count": (
                            self.data[col].value_counts().iloc[0] if len(self.data[col].value_counts()) > 0 else None
                        ),
                    }
                )

            summary_stats.append(col_stats)

        return pd.DataFrame(summary_stats)

    def numerical_summary(self) -> pd.DataFrame:
        """
        Generate detailed summary for numerical columns.

        Returns:
            pd.DataFrame: Numerical summary statistics
        """
        if not self.numerical_columns:
            print("No numerical columns found.")
            return pd.DataFrame()

        summary = self.data[self.numerical_columns].describe().T
        summary["skewness"] = self.data[self.numerical_columns].skew()
        summary["kurtosis"] = self.data[self.numerical_columns].kurtosis()
        summary["null_count"] = self.data[self.numerical_columns].isnull().sum()
        summary["null_percentage"] = self.data[self.numerical_columns].isnull().sum() / len(self.data) * 100

        return summary

    def categorical_summary(self) -> pd.DataFrame:
        """
        Generate detailed summary for categorical columns.

        Returns:
            pd.DataFrame: Categorical summary statistics
        """
        if not self.categorical_columns:
            print("No categorical columns found.")
            return pd.DataFrame()

        summary_data = []
        for col in self.categorical_columns:
            value_counts = self.data[col].value_counts()
            summary_data.append(
                {
                    "column": col,
                    "unique_values": self.data[col].nunique(),
                    "null_count": self.data[col].isnull().sum(),
                    "null_percentage": self.data[col].isnull().sum() / len(self.data) * 100,
                    "mode": self.data[col].mode().iloc[0] if len(self.data[col].mode()) > 0 else None,
                    "mode_frequency": value_counts.iloc[0] if len(value_counts) > 0 else None,
                    "mode_percentage": (value_counts.iloc[0] / len(self.data) * 100) if len(value_counts) > 0 else None,
                }
            )

        return pd.DataFrame(summary_data)

    def correlation_analysis(self, method: str = "pearson", min_correlation: float = 0.0) -> pd.DataFrame:
        """
        Perform correlation analysis on numerical columns.

        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            min_correlation (float): Minimum correlation threshold to display

        Returns:
            pd.DataFrame: Correlation matrix
        """
        if not self.numerical_columns:
            print("No numerical columns found for correlation analysis.")
            return pd.DataFrame()

        corr_matrix = self.data[self.numerical_columns].corr(method=method)

        if min_correlation > 0:
            # Filter correlations below threshold
            mask = np.abs(corr_matrix) >= min_correlation
            corr_matrix = corr_matrix.where(mask, other=np.nan)

        return corr_matrix

    def get_high_correlations(self, threshold: float = 0.7, method: str = "pearson") -> pd.DataFrame:
        """
        Get pairs of highly correlated features.

        Args:
            threshold (float): Minimum correlation threshold
            method (str): Correlation method

        Returns:
            pd.DataFrame: DataFrame with highly correlated pairs
        """
        if not self.numerical_columns:
            print("No numerical columns found.")
            return pd.DataFrame()

        corr_matrix = self.data[self.numerical_columns].corr(method=method)

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find pairs with correlation above threshold
        high_corr_pairs = []
        for col in upper.columns:
            for idx in upper.index:
                if abs(upper.loc[idx, col]) >= threshold:
                    high_corr_pairs.append({"feature_1": idx, "feature_2": col, "correlation": upper.loc[idx, col]})

        result = pd.DataFrame(high_corr_pairs)
        if not result.empty:
            result = result.sort_values("correlation", key=abs, ascending=False)

        return result

    def distribution_analysis(self, column: str) -> Dict[str, Any]:
        """
        Analyze the distribution of a numerical column.

        Args:
            column (str): Column name to analyze

        Returns:
            Dict containing distribution statistics
        """
        if column not in self.numerical_columns:
            raise ValueError(f"Column '{column}' is not numerical.")

        data = self.data[column].dropna()

        # Normality tests
        if len(data) >= 8:  # Minimum sample size for these tests
            shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for large datasets
            _, ks_p = stats.kstest(data, "norm", args=(data.mean(), data.std()))
        else:
            shapiro_stat, shapiro_p = None, None
            ks_p = None

        analysis = {
            "column": column,
            "count": len(data),
            "mean": data.mean(),
            "median": data.median(),
            "mode": data.mode().iloc[0] if len(data.mode()) > 0 else None,
            "std": data.std(),
            "variance": data.var(),
            "min": data.min(),
            "max": data.max(),
            "range": data.max() - data.min(),
            "q1": data.quantile(0.25),
            "q3": data.quantile(0.75),
            "iqr": data.quantile(0.75) - data.quantile(0.25),
            "skewness": data.skew(),
            "kurtosis": data.kurtosis(),
            "shapiro_statistic": shapiro_stat,
            "shapiro_p_value": shapiro_p,
            "ks_p_value": ks_p,
            "is_normal": shapiro_p > 0.05 if shapiro_p else None,
        }

        return analysis

    def value_counts_analysis(self, column: str, top_n: int = 10, include_percentage: bool = True) -> pd.DataFrame:
        """
        Analyze value counts for a column.

        Args:
            column (str): Column name to analyze
            top_n (int): Number of top values to return
            include_percentage (bool): Whether to include percentage

        Returns:
            pd.DataFrame: Value counts analysis
        """
        value_counts = self.data[column].value_counts().head(top_n)

        result = pd.DataFrame({"value": value_counts.index, "count": value_counts.values})

        if include_percentage:
            result["percentage"] = result["count"] / len(self.data) * 100
            result["cumulative_percentage"] = result["percentage"].cumsum()

        return result

    def missing_value_analysis(self) -> pd.DataFrame:
        """
        Detailed analysis of missing values.

        Returns:
            pd.DataFrame: Missing value analysis
        """
        missing_data = []

        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                missing_data.append(
                    {
                        "column": col,
                        "missing_count": missing_count,
                        "missing_percentage": missing_count / len(self.data) * 100,
                        "dtype": str(self.data[col].dtype),
                    }
                )

        result = pd.DataFrame(missing_data)
        if not result.empty:
            result = result.sort_values("missing_count", ascending=False)

        return result

    def detect_outliers_summary(self, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """
        Summarize outliers for all numerical columns.

        Args:
            method (str): Method for outlier detection ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection

        Returns:
            pd.DataFrame: Outlier summary
        """
        outlier_summary = []

        for col in self.numerical_columns:
            data = self.data[col].dropna()

            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            else:  # zscore
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold]

            outlier_summary.append(
                {
                    "column": col,
                    "outlier_count": len(outliers),
                    "outlier_percentage": len(outliers) / len(data) * 100,
                    "lower_bound": lower_bound if method == "iqr" else None,
                    "upper_bound": upper_bound if method == "iqr" else None,
                }
            )

        return pd.DataFrame(outlier_summary)

    def group_analysis(
        self,
        group_column: str,
        agg_columns: Optional[List[str]] = None,
        agg_functions: List[str] = ["mean", "std", "count"],
    ) -> pd.DataFrame:
        """
        Perform group-by analysis.

        Args:
            group_column (str): Column to group by
            agg_columns (List[str]): Columns to aggregate
            agg_functions (List[str]): Aggregation functions to apply

        Returns:
            pd.DataFrame: Grouped analysis results
        """
        if agg_columns is None:
            agg_columns = self.numerical_columns

        agg_dict = {col: agg_functions for col in agg_columns if col != group_column}

        grouped = self.data.groupby(group_column).agg(agg_dict)
        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

        return grouped.reset_index()

    def statistical_tests(
        self, column1: str, column2: Optional[str] = None, test_type: str = "ttest"
    ) -> Dict[str, Any]:
        """
        Perform statistical tests.

        Args:
            column1 (str): First column for testing
            column2 (str): Second column for testing (optional)
            test_type (str): Type of test ('ttest', 'anova', 'chi2', 'mannwhitney')

        Returns:
            Dict containing test results
        """
        results = {"test_type": test_type}

        if test_type == "ttest" and column2:
            stat, p_value = stats.ttest_ind(self.data[column1].dropna(), self.data[column2].dropna())
            results.update({"statistic": stat, "p_value": p_value, "significant": p_value < 0.05})

        elif test_type == "anova":
            groups = [group[column1].dropna().values for name, group in self.data.groupby(column2)]
            stat, p_value = stats.f_oneway(*groups)
            results.update({"statistic": stat, "p_value": p_value, "significant": p_value < 0.05})

        elif test_type == "chi2" and column2:
            contingency = pd.crosstab(self.data[column1], self.data[column2])
            stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            results.update(
                {"statistic": stat, "p_value": p_value, "degrees_of_freedom": dof, "significant": p_value < 0.05}
            )

        elif test_type == "mannwhitney" and column2:
            stat, p_value = stats.mannwhitneyu(self.data[column1].dropna(), self.data[column2].dropna())
            results.update({"statistic": stat, "p_value": p_value, "significant": p_value < 0.05})

        return results

    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of the EDA.

        Returns:
            str: EDA report
        """
        report = []
        report.append("=" * 60)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 60)

        # Basic Info
        info = self.basic_info()
        report.append("\n📊 BASIC INFORMATION")
        report.append("-" * 40)
        report.append(f"Rows: {info['rows']:,}")
        report.append(f"Columns: {info['columns']}")
        report.append(f"Total Cells: {info['total_cells']:,}")
        report.append(f"Missing Cells: {info['missing_cells']:,} ({info['missing_percentage']:.2f}%)")
        report.append(f"Duplicate Rows: {info['duplicate_rows']:,}")
        report.append(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        report.append(f"Numerical Columns: {info['numerical_columns']}")
        report.append(f"Categorical Columns: {info['categorical_columns']}")

        # Numerical Summary
        if self.numerical_columns:
            report.append("\n📈 NUMERICAL COLUMNS SUMMARY")
            report.append("-" * 40)
            num_summary = self.numerical_summary()
            for col in num_summary.index:
                report.append(f"\n{col}:")
                report.append(f"  Mean: {num_summary.loc[col, 'mean']:.4f}")
                report.append(f"  Std: {num_summary.loc[col, 'std']:.4f}")
                report.append(f"  Min: {num_summary.loc[col, 'min']:.4f}")
                report.append(f"  Max: {num_summary.loc[col, 'max']:.4f}")
                report.append(f"  Skewness: {num_summary.loc[col, 'skewness']:.4f}")

        # Categorical Summary
        if self.categorical_columns:
            report.append("\n📋 CATEGORICAL COLUMNS SUMMARY")
            report.append("-" * 40)
            cat_summary = self.categorical_summary()
            for _, row in cat_summary.iterrows():
                report.append(f"\n{row['column']}:")
                report.append(f"  Unique Values: {row['unique_values']}")
                report.append(f"  Mode: {row['mode']}")
                report.append(f"  Mode Frequency: {row['mode_frequency']}")

        # Missing Values
        missing = self.missing_value_analysis()
        if not missing.empty:
            report.append("\n⚠️ MISSING VALUES")
            report.append("-" * 40)
            for _, row in missing.iterrows():
                report.append(f"{row['column']}: {row['missing_count']} ({row['missing_percentage']:.2f}%)")

        # High Correlations
        high_corr = self.get_high_correlations(threshold=0.7)
        if not high_corr.empty:
            report.append("\n🔗 HIGH CORRELATIONS (|r| >= 0.7)")
            report.append("-" * 40)
            for _, row in high_corr.iterrows():
                report.append(f"{row['feature_1']} <-> {row['feature_2']}: {row['correlation']:.4f}")

        report.append("\n" + "=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)

        return "\n".join(report)

    def print_report(self):
        """Print the EDA report to console."""
        print(self.generate_report())


if __name__ == "__main__":
    # Example usage
    print("Exploratory Data Analysis Module")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, 200),
            "income": np.random.normal(50000, 15000, 200),
            "spending_score": np.random.randint(1, 100, 200),
            "credit_score": np.random.normal(650, 100, 200),
            "category": np.random.choice(["A", "B", "C", "D"], 200),
            "gender": np.random.choice(["M", "F"], 200),
        }
    )

    # Add correlation
    sample_data["bonus"] = sample_data["income"] * 0.1 + np.random.normal(0, 1000, 200)

    # Add missing values
    sample_data.loc[5:15, "age"] = np.nan
    sample_data.loc[20:30, "income"] = np.nan

    # Initialize EDA
    eda = ExploratoryAnalysis(sample_data)

    # Print report
    eda.print_report()

    # Get specific analyses
    print("\n\nCorrelation Matrix:")
    print(eda.correlation_analysis().round(3))

    print("\n\nHigh Correlations:")
    print(eda.get_high_correlations())
