"""
AI-Powered Insights Module
Provides automated insights, explanations, and recommendations for data analysis.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class AIInsights:
    """
    AI-powered insights generator for automated data analysis and recommendations.

    This class provides intelligent insights, pattern detection, and actionable
    recommendations based on data analysis.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the AI Insights generator.

        Args:
            data (pd.DataFrame): The dataset to analyze
        """
        self.data = data.copy()
        self.insights = []
        self.recommendations = []
        self._identify_column_types()

    def _identify_column_types(self):
        """Automatically identify column types."""
        self.numerical_columns = self.data.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        self.categorical_columns = self.data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def detect_data_quality_issues(self) -> Dict[str, Any]:
        """
        Detect data quality issues and provide recommendations.

        Returns:
            Dict containing detected issues and recommendations
        """
        issues = {"missing_values": {}, "outliers": {}, "inconsistencies": [], "data_types": [], "duplicates": 0}

        # Missing values analysis
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]

        for col in high_missing.index:
            issues["missing_values"][col] = {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col]),
                "severity": "critical" if missing_pct[col] > 80 else "high",
                "recommendation": "Consider removing this column or using advanced imputation",
            }

        # Outlier detection
        for col in self.numerical_columns[:10]:  # Limit to first 10
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.data[col] < lower) | (self.data[col] > upper)).sum()
            outlier_pct = outliers / len(self.data) * 100

            if outlier_pct > 10:
                issues["outliers"][col] = {
                    "count": int(outliers),
                    "percentage": float(outlier_pct),
                    "recommendation": "Investigate these outliers - they may be errors or important edge cases",
                }

        # Duplicate rows
        issues["duplicates"] = int(self.data.duplicated().sum())

        return issues

    def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect interesting patterns in the data.

        Returns:
            Dict containing detected patterns
        """
        patterns = {
            "strong_correlations": [],
            "unusual_distributions": [],
            "skewed_features": [],
            "multimodal_features": [],
            "categorical_patterns": [],
        }

        # Strong correlations
        if len(self.numerical_columns) >= 2:
            corr_matrix = self.data[self.numerical_columns].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        patterns["strong_correlations"].append(
                            {
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": float(corr_val),
                                "insight": f'Strong {"positive" if corr_val > 0 else "negative"} correlation detected',
                            }
                        )

        # Skewed features
        for col in self.numerical_columns[:10]:
            skewness = self.data[col].skew()
            if abs(skewness) > 2:
                patterns["skewed_features"].append(
                    {
                        "feature": col,
                        "skewness": float(skewness),
                        "insight": "Highly skewed distribution - consider transformation",
                        "recommendation": "Apply log or Box-Cox transformation",
                    }
                )

        # Categorical patterns
        for col in self.categorical_columns[:5]:
            value_counts = self.data[col].value_counts()
            if len(value_counts) > 0:
                dominant_pct = value_counts.iloc[0] / len(self.data) * 100
                if dominant_pct > 80:
                    patterns["categorical_patterns"].append(
                        {
                            "feature": col,
                            "dominant_value": str(value_counts.index[0]),
                            "percentage": float(dominant_pct),
                            "insight": "Highly imbalanced category distribution",
                            "recommendation": "Consider balancing techniques if using for ML",
                        }
                    )

        return patterns

    def generate_feature_insights(self, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate insights about features and their relationships.

        Args:
            target_column (str): Optional target column for supervised insights

        Returns:
            Dict containing feature insights
        """
        insights = {"feature_statistics": {}, "target_relationships": {}, "feature_importance_hints": []}

        # Feature statistics
        for col in self.numerical_columns[:15]:
            stats = {
                "mean": float(self.data[col].mean()),
                "median": float(self.data[col].median()),
                "std": float(self.data[col].std()),
                "range": float(self.data[col].max() - self.data[col].min()),
                "cv": float(self.data[col].std() / self.data[col].mean()) if self.data[col].mean() != 0 else 0,
            }

            # Generate insight
            if stats["cv"] > 1:
                stats["insight"] = "High variability - may be important for modeling"
            elif stats["cv"] < 0.1:
                stats["insight"] = "Low variability - may not add much information"
            else:
                stats["insight"] = "Moderate variability"

            insights["feature_statistics"][col] = stats

        # Target relationships
        if target_column and target_column in self.data.columns:
            if target_column in self.numerical_columns:
                # Correlation with target
                for col in self.numerical_columns:
                    if col != target_column:
                        corr = self.data[col].corr(self.data[target_column])
                        if abs(corr) > 0.3:
                            insights["target_relationships"][col] = {
                                "correlation": float(corr),
                                "strength": "strong" if abs(corr) > 0.7 else "moderate",
                                "direction": "positive" if corr > 0 else "negative",
                            }
            else:
                # Categorical target - group differences
                for col in self.numerical_columns[:10]:
                    group_means = self.data.groupby(target_column)[col].mean()
                    if group_means.std() > self.data[col].std() * 0.3:
                        insights["target_relationships"][col] = {
                            "group_differences": group_means.to_dict(),
                            "insight": "Significant differences across target categories",
                        }

        return insights

    def generate_recommendations(self, task_type: str = "auto") -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on data analysis.

        Args:
            task_type (str): Type of task ('classification', 'regression', 'auto')

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Data quality recommendations
        issues = self.detect_data_quality_issues()

        if issues["missing_values"]:
            recommendations.append(
                {
                    "category": "Data Quality",
                    "priority": "high",
                    "issue": f"{len(issues['missing_values'])} columns with >50% missing values",
                    "recommendation": "Review and apply appropriate imputation strategies or remove high-missing columns",
                    "action": "Use handle_missing_values() with appropriate strategy",
                }
            )

        if issues["duplicates"] > 0:
            recommendations.append(
                {
                    "category": "Data Quality",
                    "priority": "medium",
                    "issue": f"{issues['duplicates']} duplicate rows found",
                    "recommendation": "Remove duplicate rows to avoid data leakage",
                    "action": "Use remove_duplicates() method",
                }
            )

        # Preprocessing recommendations
        patterns = self.detect_patterns()

        if patterns["skewed_features"]:
            recommendations.append(
                {
                    "category": "Feature Engineering",
                    "priority": "medium",
                    "issue": f"{len(patterns['skewed_features'])} highly skewed features detected",
                    "recommendation": "Apply log or Box-Cox transformations to improve model performance",
                    "action": "Consider feature transformations before modeling",
                }
            )

        if patterns["strong_correlations"]:
            recommendations.append(
                {
                    "category": "Feature Engineering",
                    "priority": "low",
                    "issue": f"{len(patterns['strong_correlations'])} pairs of highly correlated features",
                    "recommendation": "Consider removing one feature from each highly correlated pair to reduce multicollinearity",
                    "action": "Review correlation matrix and remove redundant features",
                }
            )

        # Modeling recommendations
        if task_type == "classification":
            recommendations.append(
                {
                    "category": "Modeling",
                    "priority": "high",
                    "issue": "Classification task detected",
                    "recommendation": "Start with Random Forest or Gradient Boosting, then try ensemble methods",
                    "action": "Use train_all_models() to compare multiple algorithms",
                }
            )
        elif task_type == "regression":
            recommendations.append(
                {
                    "category": "Modeling",
                    "priority": "high",
                    "issue": "Regression task detected",
                    "recommendation": "Start with Random Forest Regressor or Gradient Boosting, evaluate R² and RMSE",
                    "action": "Use train_all_models() and focus on regression metrics",
                }
            )

        # Scale recommendations
        if len(self.numerical_columns) > 0:
            scales = {}
            for col in self.numerical_columns[:10]:
                col_range = self.data[col].max() - self.data[col].min()
                if col_range > 1000:
                    scales[col] = col_range

            if scales:
                recommendations.append(
                    {
                        "category": "Preprocessing",
                        "priority": "high",
                        "issue": f"{len(scales)} features with large value ranges",
                        "recommendation": "Scale features using StandardScaler or MinMaxScaler before training",
                        "action": "Use scale_features() method before modeling",
                    }
                )

        return recommendations

    def generate_automated_report(self, target_column: Optional[str] = None) -> str:
        """
        Generate a comprehensive automated insights report.

        Args:
            target_column (str): Optional target column name

        Returns:
            str: Formatted insights report
        """
        report = []
        report.append("=" * 70)
        report.append("🤖 AI-POWERED DATA INSIGHTS REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Dataset Overview
        report.append("📊 DATASET OVERVIEW")
        report.append("-" * 70)
        report.append(f"Rows: {self.data.shape[0]:,}")
        report.append(f"Columns: {self.data.shape[1]}")
        report.append(f"Numerical Features: {len(self.numerical_columns)}")
        report.append(f"Categorical Features: {len(self.categorical_columns)}")
        report.append(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")

        # Data Quality Issues
        report.append("⚠️ DATA QUALITY ASSESSMENT")
        report.append("-" * 70)
        issues = self.detect_data_quality_issues()

        if issues["missing_values"]:
            report.append(f"Critical Missing Values: {len(issues['missing_values'])} columns")
            for col, info in list(issues["missing_values"].items())[:5]:
                report.append(f"  • {col}: {info['percentage']:.1f}% missing ({info['severity']})")
        else:
            report.append("✅ No critical missing value issues detected")

        if issues["duplicates"] > 0:
            report.append(f"⚠️ {issues['duplicates']} duplicate rows found")
        else:
            report.append("✅ No duplicate rows detected")

        report.append("")

        # Patterns
        report.append("🔍 DETECTED PATTERNS")
        report.append("-" * 70)
        patterns = self.detect_patterns()

        if patterns["strong_correlations"]:
            report.append(f"Strong Correlations ({len(patterns['strong_correlations'])} pairs):")
            for pattern in patterns["strong_correlations"][:5]:
                report.append(f"  • {pattern['feature1']} ↔ {pattern['feature2']}: " f"{pattern['correlation']:.3f}")
        else:
            report.append("No extremely strong correlations detected")

        if patterns["skewed_features"]:
            report.append(f"\nSkewed Features ({len(patterns['skewed_features'])}):")
            for feature in patterns["skewed_features"][:5]:
                report.append(f"  • {feature['feature']}: skewness = {feature['skewness']:.2f}")

        report.append("")

        # Recommendations
        report.append("💡 ACTIONABLE RECOMMENDATIONS")
        report.append("-" * 70)
        task_type = "auto"
        if target_column:
            unique_vals = self.data[target_column].nunique()
            task_type = "classification" if unique_vals <= 10 else "regression"

        recommendations = self.generate_recommendations(task_type)

        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

        for i, rec in enumerate(recommendations[:10], 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            emoji = priority_emoji.get(rec["priority"], "⚪")
            report.append(f"{i}. {emoji} [{rec['priority'].upper()}] {rec['category']}")
            report.append(f"   Issue: {rec['issue']}")
            report.append(f"   → {rec['recommendation']}")
            report.append(f"   Action: {rec['action']}")
            report.append("")

        # Feature Insights
        if target_column:
            report.append("🎯 TARGET RELATIONSHIPS")
            report.append("-" * 70)
            feature_insights = self.generate_feature_insights(target_column)

            if feature_insights["target_relationships"]:
                for feature, info in list(feature_insights["target_relationships"].items())[:10]:
                    if "correlation" in info:
                        report.append(
                            f"  • {feature}: {info['strength']} "
                            f"{info['direction']} correlation ({info['correlation']:.3f})"
                        )

        report.append("")
        report.append("=" * 70)
        report.append("End of Report")
        report.append("=" * 70)

        return "\n".join(report)

    def get_quick_insights(self) -> Dict[str, Any]:
        """
        Get quick summary insights for the dataset.

        Returns:
            Dict containing quick insights
        """
        insights = {
            "data_shape": self.data.shape,
            "missing_percentage": (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100),
            "duplicate_percentage": (self.data.duplicated().sum() / len(self.data) * 100),
            "numerical_features": len(self.numerical_columns),
            "categorical_features": len(self.categorical_columns),
            "has_missing": self.data.isnull().any().any(),
            "has_duplicates": self.data.duplicated().any(),
        }

        # Quick pattern detection
        if len(self.numerical_columns) >= 2:
            corr_matrix = self.data[self.numerical_columns].corr().abs()
            high_corr_count = ((corr_matrix > 0.8) & (corr_matrix < 1.0)).sum().sum() // 2
            insights["high_correlation_pairs"] = int(high_corr_count)

        return insights


if __name__ == "__main__":
    # Example usage
    print("AI Insights Module")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "credit_score": np.random.normal(650, 100, 1000),
            "purchase_amount": np.random.lognormal(5, 1, 1000),
            "category": np.random.choice(["A", "B", "C", "D"], 1000, p=[0.5, 0.3, 0.15, 0.05]),
            "is_premium": np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        }
    )

    # Add some missing values
    sample_data.loc[100:150, "income"] = np.nan
    sample_data.loc[200:250, "credit_score"] = np.nan

    # Initialize AI Insights
    ai = AIInsights(sample_data)

    # Generate report
    print("\nGenerating automated insights report...")
    report = ai.generate_automated_report(target_column="is_premium")
    print(report)

    # Get quick insights
    print("\nQuick Insights:")
    quick = ai.get_quick_insights()
    for key, value in quick.items():
        print(f"  {key}: {value}")
