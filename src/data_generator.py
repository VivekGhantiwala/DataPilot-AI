"""
Sample Data Generator Module
Generate synthetic datasets for testing and demonstration purposes.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataGenerator:
    """
    Generate synthetic datasets for testing and demonstration.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_classification_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        n_informative: int = 5,
        noise: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate synthetic classification dataset.

        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            n_classes (int): Number of classes
            n_informative (int): Number of informative features
            noise (float): Noise level (0-1)

        Returns:
            pd.DataFrame: Generated dataset
        """
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=self.random_state,
            noise=noise,
        )

        # Create feature names
        feature_names = [f"feature_{i+1}" for i in range(n_features)]

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        return df

    def generate_regression_data(
        self, n_samples: int = 1000, n_features: int = 10, n_informative: int = 5, noise: float = 10.0
    ) -> pd.DataFrame:
        """
        Generate synthetic regression dataset.

        Args:
            n_samples (int): Number of samples
            n_features (int): Number of features
            n_informative (int): Number of informative features
            noise (float): Noise level

        Returns:
            pd.DataFrame: Generated dataset
        """
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=self.random_state,
        )

        # Create feature names
        feature_names = [f"feature_{i+1}" for i in range(n_features)]

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        return df

    def generate_customer_data(self, n_customers: int = 1000) -> pd.DataFrame:
        """
        Generate realistic customer dataset.

        Args:
            n_customers (int): Number of customers

        Returns:
            pd.DataFrame: Customer dataset
        """
        np.random.seed(self.random_state)

        # Generate realistic customer data
        data = {
            "customer_id": range(1, n_customers + 1),
            "age": np.random.randint(18, 75, n_customers),
            "gender": np.random.choice(["Male", "Female", "Other"], n_customers, p=[0.49, 0.49, 0.02]),
            "annual_income": np.random.lognormal(10.5, 0.5, n_customers).round(2),
            "credit_score": np.random.normal(650, 100, n_customers).clip(300, 850).astype(int),
            "years_as_customer": np.random.exponential(3, n_customers).clip(0, 20).astype(int),
            "total_purchases": np.random.poisson(25, n_customers),
            "avg_transaction_value": np.random.lognormal(4, 1, n_customers).round(2),
            "category_preference": np.random.choice(
                ["Electronics", "Clothing", "Home", "Sports", "Books"], n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1]
            ),
            "membership_type": np.random.choice(["Basic", "Premium", "Gold"], n_customers, p=[0.6, 0.3, 0.1]),
            "churn_risk": np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        }

        df = pd.DataFrame(data)

        # Add some correlations
        df["lifetime_value"] = (
            df["total_purchases"] * df["avg_transaction_value"] * (1 + df["years_as_customer"] / 10)
            + np.random.normal(0, 100, n_customers)
        ).round(2)

        # Add some missing values randomly
        missing_indices = np.random.choice(n_customers, size=int(n_customers * 0.05), replace=False)
        df.loc[missing_indices, "credit_score"] = np.nan

        return df

    def generate_sales_data(
        self, n_transactions: int = 10000, start_date: str = "2023-01-01", end_date: str = "2023-12-31"
    ) -> pd.DataFrame:
        """
        Generate realistic sales transaction dataset.

        Args:
            n_transactions (int): Number of transactions
            start_date (str): Start date
            end_date (str): End date

        Returns:
            pd.DataFrame: Sales dataset
        """
        np.random.seed(self.random_state)

        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        transaction_dates = np.random.choice(dates, n_transactions)

        # Generate product categories
        categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys"]

        # Generate data
        data = {
            "transaction_id": range(1, n_transactions + 1),
            "date": transaction_dates,
            "product_category": np.random.choice(categories, n_transactions),
            "quantity": np.random.poisson(2, n_transactions).clip(1, 10),
            "unit_price": np.random.lognormal(3.5, 1, n_transactions).round(2),
            "discount_percent": np.random.choice(
                [0, 5, 10, 15, 20, 25], n_transactions, p=[0.5, 0.2, 0.15, 0.1, 0.04, 0.01]
            ),
            "customer_rating": np.random.choice([1, 2, 3, 4, 5], n_transactions, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
            "region": np.random.choice(["North", "South", "East", "West"], n_transactions),
            "sales_rep_id": np.random.randint(1, 21, n_transactions),
        }

        df = pd.DataFrame(data)

        # Calculate total amount
        df["total_amount"] = (df["quantity"] * df["unit_price"] * (1 - df["discount_percent"] / 100)).round(2)

        # Add day of week, month
        df["day_of_week"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.month_name()
        df["quarter"] = df["date"].dt.quarter

        return df

    def add_noise(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Add noise to numerical columns.

        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to add noise to (None for all numerical)
            noise_level (float): Noise level (0-1)

        Returns:
            pd.DataFrame: DataFrame with added noise
        """
        df = df.copy()
        numerical_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numerical_cols:
            if col in df.columns:
                std = df[col].std()
                noise = np.random.normal(0, std * noise_level, len(df))
                df[col] = df[col] + noise

        return df

    def add_outliers(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, outlier_percentage: float = 0.05
    ) -> pd.DataFrame:
        """
        Add outliers to numerical columns.

        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to add outliers to
            outlier_percentage (float): Percentage of outliers to add

        Returns:
            pd.DataFrame: DataFrame with added outliers
        """
        df = df.copy()
        numerical_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        n_outliers = int(len(df) * outlier_percentage)

        for col in numerical_cols:
            if col in df.columns:
                outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
                # Make outliers significantly different
                multiplier = np.random.choice([-3, 3], n_outliers)
                df.loc[outlier_indices, col] = df[col].mean() + multiplier * df[col].std() * 2

        return df


if __name__ == "__main__":
    # Example usage
    print("Data Generator Module")
    print("=" * 50)

    generator = DataGenerator(random_state=42)

    # Generate classification data
    print("\n1. Generating classification dataset...")
    class_data = generator.generate_classification_data(n_samples=500, n_features=8)
    print(f"   Shape: {class_data.shape}")
    print(f"   Target distribution:\n{class_data['target'].value_counts()}")

    # Generate customer data
    print("\n2. Generating customer dataset...")
    customer_data = generator.generate_customer_data(n_customers=1000)
    print(f"   Shape: {customer_data.shape}")
    print(f"   Columns: {list(customer_data.columns)}")

    # Generate sales data
    print("\n3. Generating sales dataset...")
    sales_data = generator.generate_sales_data(n_transactions=5000)
    print(f"   Shape: {sales_data.shape}")
    print(f"   Date range: {sales_data['date'].min()} to {sales_data['date'].max()}")

    print("\n✅ All datasets generated successfully!")
