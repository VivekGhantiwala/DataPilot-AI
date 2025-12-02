"""
Data Preprocessing Module
Handles data cleaning, transformation, and feature engineering.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for cleaning and transforming data.

    Attributes:
        data (pd.DataFrame): The dataset to preprocess
        numerical_columns (List[str]): List of numerical column names
        categorical_columns (List[str]): List of categorical column names
        scalers (dict): Dictionary storing fitted scalers
        encoders (dict): Dictionary storing fitted encoders
    """

    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.data = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            pd.DataFrame: Loaded dataset
        """
        self.data = pd.read_csv(filepath, **kwargs)
        self._identify_column_types()
        print(f"✅ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load data from an existing DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: The loaded dataset
        """
        self.data = df.copy()
        self._identify_column_types()
        print(f"✅ DataFrame loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def _identify_column_types(self):
        """Automatically identify numerical and categorical columns."""
        self.numerical_columns = self.data.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        self.categorical_columns = self.data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the dataset.

        Returns:
            dict: Dictionary containing data information
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        info = {
            "shape": self.data.shape,
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "missing_percentage": (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "memory_usage": self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        return info

    def handle_missing_values(
        self,
        strategy: str = "auto",
        numerical_strategy: str = "mean",
        categorical_strategy: str = "mode",
        fill_value: Optional[Union[str, int, float]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            strategy (str): Strategy for handling missing values ('auto', 'drop', 'fill')
            numerical_strategy (str): Strategy for numerical columns ('mean', 'median', 'most_frequent')
            categorical_strategy (str): Strategy for categorical columns ('mode', 'most_frequent')
            fill_value: Value to fill missing values with (used when strategy='fill')
            columns (List[str]): Specific columns to handle (None for all)

        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        df = self.data.copy()
        cols_to_process = columns if columns else df.columns.tolist()

        if strategy == "drop":
            df = df.dropna(subset=cols_to_process)

        elif strategy == "fill" and fill_value is not None:
            df[cols_to_process] = df[cols_to_process].fillna(fill_value)

        elif strategy == "auto":
            # Handle numerical columns
            num_cols = [c for c in cols_to_process if c in self.numerical_columns]
            if num_cols:
                imputer = SimpleImputer(strategy=numerical_strategy)
                df[num_cols] = imputer.fit_transform(df[num_cols])
                self.imputers["numerical"] = imputer

            # Handle categorical columns
            cat_cols = [c for c in cols_to_process if c in self.categorical_columns]
            if cat_cols:
                for col in cat_cols:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col] = df[col].fillna(mode_value[0])

        self.data = df
        print(f"✅ Missing values handled using '{strategy}' strategy")
        return self.data

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Args:
            subset (List[str]): Columns to consider for identifying duplicates
            keep (str): Which duplicates to keep ('first', 'last', False)

        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(self.data)

        print(f"✅ Removed {removed_rows} duplicate rows")
        return self.data

    def detect_outliers(
        self, method: str = "iqr", columns: Optional[List[str]] = None, threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers in numerical columns.

        Args:
            method (str): Method for outlier detection ('iqr', 'zscore')
            columns (List[str]): Columns to check for outliers
            threshold (float): Threshold for outlier detection

        Returns:
            pd.DataFrame: DataFrame indicating outlier positions
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        cols = columns if columns else self.numerical_columns
        outlier_mask = pd.DataFrame(False, index=self.data.index, columns=cols)

        for col in cols:
            if method == "iqr":
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[col] = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)

            elif method == "zscore":
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outlier_mask[col] = z_scores > threshold

        outlier_counts = outlier_mask.sum()
        print(f"📊 Outliers detected:\n{outlier_counts[outlier_counts > 0]}")
        return outlier_mask

    def handle_outliers(
        self, method: str = "clip", columns: Optional[List[str]] = None, threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.

        Args:
            method (str): Method for handling outliers ('clip', 'remove', 'replace_mean')
            columns (List[str]): Columns to handle outliers in
            threshold (float): Threshold for outlier detection

        Returns:
            pd.DataFrame: Dataset with handled outliers
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        cols = columns if columns else self.numerical_columns
        df = self.data.copy()

        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            if method == "clip":
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "remove":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == "replace_mean":
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df.loc[mask, col] = df[col].mean()

        self.data = df
        print(f"✅ Outliers handled using '{method}' method")
        return self.data

    def scale_features(self, method: str = "standard", columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            method (str): Scaling method ('standard', 'minmax')
            columns (List[str]): Columns to scale

        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        cols = columns if columns else self.numerical_columns

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.data[cols] = scaler.fit_transform(self.data[cols])
        self.scalers[method] = scaler

        print(f"✅ Features scaled using {method} scaling")
        return self.data

    def encode_categorical(
        self, method: str = "onehot", columns: Optional[List[str]] = None, drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            method (str): Encoding method ('onehot', 'label')
            columns (List[str]): Columns to encode
            drop_first (bool): Whether to drop first category in one-hot encoding

        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        cols = columns if columns else self.categorical_columns

        if method == "onehot":
            self.data = pd.get_dummies(self.data, columns=cols, drop_first=drop_first)
        elif method == "label":
            for col in cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.encoders[col] = le
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        self._identify_column_types()
        print(f"✅ Categorical variables encoded using {method} encoding")
        return self.data

    def create_features(self, operations: List[dict]) -> pd.DataFrame:
        """
        Create new features based on specified operations.

        Args:
            operations (List[dict]): List of feature engineering operations
                Each dict should have:
                - 'name': Name of the new feature
                - 'operation': Type of operation ('ratio', 'product', 'sum', 'diff', 'log', 'sqrt', 'square')
                - 'columns': List of columns involved

        Returns:
            pd.DataFrame: Dataset with new features
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        for op in operations:
            name = op["name"]
            operation = op["operation"]
            cols = op["columns"]

            if operation == "ratio" and len(cols) == 2:
                self.data[name] = self.data[cols[0]] / (self.data[cols[1]] + 1e-8)
            elif operation == "product":
                self.data[name] = self.data[cols].prod(axis=1)
            elif operation == "sum":
                self.data[name] = self.data[cols].sum(axis=1)
            elif operation == "diff" and len(cols) == 2:
                self.data[name] = self.data[cols[0]] - self.data[cols[1]]
            elif operation == "log" and len(cols) == 1:
                self.data[name] = np.log1p(self.data[cols[0]])
            elif operation == "sqrt" and len(cols) == 1:
                self.data[name] = np.sqrt(np.abs(self.data[cols[0]]))
            elif operation == "square" and len(cols) == 1:
                self.data[name] = self.data[cols[0]] ** 2

        self._identify_column_types()
        print(f"✅ Created {len(operations)} new features")
        return self.data

    def get_clean_data(self) -> pd.DataFrame:
        """
        Get the preprocessed data.

        Returns:
            pd.DataFrame: The preprocessed dataset
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        return self.data.copy()

    def preprocess_pipeline(
        self,
        handle_missing: bool = True,
        remove_dups: bool = True,
        handle_outliers_flag: bool = True,
        scale: bool = True,
        encode: bool = True,
    ) -> pd.DataFrame:
        """
        Run a complete preprocessing pipeline.

        Args:
            handle_missing (bool): Whether to handle missing values
            remove_dups (bool): Whether to remove duplicates
            handle_outliers_flag (bool): Whether to handle outliers
            scale (bool): Whether to scale features
            encode (bool): Whether to encode categorical variables

        Returns:
            pd.DataFrame: Fully preprocessed dataset
        """
        print("🔄 Starting preprocessing pipeline...")

        if handle_missing:
            self.handle_missing_values(strategy="auto")

        if remove_dups:
            self.remove_duplicates()

        if handle_outliers_flag:
            self.handle_outliers(method="clip")

        if encode:
            self.encode_categorical(method="label")

        if scale:
            self.scale_features(method="standard")

        print("✅ Preprocessing pipeline completed!")
        return self.data


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 50)

    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, 100),
            "income": np.random.normal(50000, 15000, 100),
            "spending_score": np.random.randint(1, 100, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "gender": np.random.choice(["M", "F"], 100),
        }
    )

    # Add some missing values
    sample_data.loc[5:10, "age"] = np.nan
    sample_data.loc[15:20, "income"] = np.nan

    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_dataframe(sample_data)

    # Get data info
    info = preprocessor.get_data_info()
    print(f"\nData Shape: {info['shape']}")
    print(f"Missing Values: {info['missing_values']}")

    # Run preprocessing pipeline
    clean_data = preprocessor.preprocess_pipeline()
    print(f"\nFinal Shape: {clean_data.shape}")
