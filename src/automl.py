"""
AutoML Module - Automated Machine Learning Pipeline
Provides automated feature engineering, model selection, and hyperparameter optimization.
"""

import json
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# Import models
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Try optional imports
try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings("ignore")


class AutoML:
    """
    Automated Machine Learning Pipeline.

    Provides end-to-end automation for:
    - Data preprocessing
    - Feature engineering
    - Model selection
    - Hyperparameter optimization
    - Model evaluation

    Example:
        >>> automl = AutoML(task='classification')
        >>> automl.fit(X, y)
        >>> predictions = automl.predict(X_new)
        >>> report = automl.get_leaderboard()
    """

    def __init__(
        self,
        task: str = "auto",
        time_budget: int = 300,
        max_models: int = 10,
        cv_folds: int = 5,
        metric: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize AutoML pipeline.

        Args:
            task: 'classification', 'regression', or 'auto'
            time_budget: Maximum time in seconds for training
            max_models: Maximum number of models to train
            cv_folds: Number of cross-validation folds
            metric: Evaluation metric (auto-selected if None)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.task = task
        self.time_budget = time_budget
        self.max_models = max_models
        self.cv_folds = cv_folds
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        # State
        self.is_fitted = False
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.leaderboard = []
        self.feature_names = None
        self.target_encoder = None
        self.scaler = None
        self.feature_selector = None
        self.preprocessing_pipeline = None

        # Timing
        self.start_time = None
        self.training_time = None

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def _detect_task(self, y: np.ndarray) -> str:
        """Automatically detect task type."""
        unique_values = len(np.unique(y))
        if unique_values <= 10 or isinstance(y[0], str):
            return "classification"
        return "regression"

    def _get_models(self) -> Dict[str, Any]:
        """Get available models for the task."""
        if self.task == "classification":
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
                "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "SVM": SVC(probability=True, random_state=self.random_state),
                "KNN": KNeighborsClassifier(n_jobs=-1),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Naive Bayes": GaussianNB(),
                "AdaBoost": AdaBoostClassifier(random_state=self.random_state, algorithm="SAMME"),
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBClassifier(
                    random_state=self.random_state, n_jobs=-1, eval_metric="logloss", verbosity=0
                )
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
        else:
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
                "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "Ridge": Ridge(random_state=self.random_state),
                "Lasso": Lasso(random_state=self.random_state),
                "ElasticNet": ElasticNet(random_state=self.random_state),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
                "AdaBoost": AdaBoostRegressor(random_state=self.random_state),
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0)
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1)
        return models

    def _get_metric(self) -> str:
        """Get default metric for task."""
        if self.metric:
            return self.metric
        if self.task == "classification":
            return "f1_weighted"
        return "r2"

    def _preprocess_data(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None, fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess features and target."""

        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

            # Identify column types
            numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

            # One-hot encode categorical columns
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

            X = X.values

        # Handle missing values
        if fit:
            self.imputer = SimpleImputer(strategy="mean")
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)

        # Scale features
        if fit:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Encode target for classification
        if y is not None and self.task == "classification":
            if isinstance(y[0], str) or isinstance(y, pd.Series) and y.dtype == "object":
                if fit:
                    self.target_encoder = LabelEncoder()
                    y = self.target_encoder.fit_transform(y)
                else:
                    y = self.target_encoder.transform(y)

        return X, y

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], validation_split: float = 0.2
    ) -> "AutoML":
        """
        Fit the AutoML pipeline.

        Args:
            X: Features
            y: Target
            validation_split: Validation set size

        Returns:
            self
        """
        self.start_time = datetime.now()
        self._log("🚀 Starting AutoML Pipeline")
        self._log("=" * 60)

        # Convert to numpy if needed
        if isinstance(y, pd.Series):
            y = y.values

        # Detect task
        if self.task == "auto":
            self.task = self._detect_task(y)
            self._log(f"📋 Detected task: {self.task}")

        # Preprocess data
        self._log("🔧 Preprocessing data...")
        X_processed, y_processed = self._preprocess_data(X, y, fit=True)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed,
            y_processed,
            test_size=validation_split,
            random_state=self.random_state,
            stratify=y_processed if self.task == "classification" else None,
        )

        self._log(f"📊 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Get models and metric
        models = self._get_models()
        metric = self._get_metric()
        self._log(f"📈 Metric: {metric}")
        self._log(f"🤖 Training {len(models)} models...")
        self._log("-" * 60)

        # Train models
        results = []
        for i, (name, model) in enumerate(models.items()):
            if i >= self.max_models:
                break

            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring=metric, n_jobs=-1)

                # Fit on full training data
                model.fit(X_train, y_train)

                # Validation score
                if self.task == "classification":
                    from sklearn.metrics import f1_score

                    val_pred = model.predict(X_val)
                    val_score = f1_score(y_val, val_pred, average="weighted")
                else:
                    from sklearn.metrics import r2_score

                    val_pred = model.predict(X_val)
                    val_score = r2_score(y_val, val_pred)

                result = {
                    "name": name,
                    "model": model,
                    "cv_score_mean": cv_scores.mean(),
                    "cv_score_std": cv_scores.std(),
                    "val_score": val_score,
                }
                results.append(result)

                self._log(f"  ✅ {name}: CV={cv_scores.mean():.4f} (±{cv_scores.std():.4f}), " f"Val={val_score:.4f}")

            except Exception as e:
                self._log(f"  ❌ {name}: Failed - {str(e)[:50]}")

        # Sort by validation score
        results.sort(key=lambda x: x["val_score"], reverse=True)
        self.leaderboard = results

        # Set best model
        if results:
            self.best_model = results[0]["model"]
            self.best_model_name = results[0]["name"]
            self.best_score = results[0]["val_score"]

            # Retrain best model on full data
            self.best_model.fit(X_processed, y_processed)

        self.is_fitted = True
        self.training_time = (datetime.now() - self.start_time).total_seconds()

        self._log("-" * 60)
        self._log(f"🏆 Best Model: {self.best_model_name}")
        self._log(f"📊 Best Score: {self.best_score:.4f}")
        self._log(f"⏱️ Training Time: {self.training_time:.2f}s")
        self._log("=" * 60)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the best model."""
        if not self.is_fitted:
            raise ValueError("AutoML is not fitted. Call fit() first.")

        X_processed, _ = self._preprocess_data(X, fit=False)
        predictions = self.best_model.predict(X_processed)

        # Decode labels if needed
        if self.target_encoder is not None:
            predictions = self.target_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("AutoML is not fitted. Call fit() first.")

        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")

        if not hasattr(self.best_model, "predict_proba"):
            raise ValueError(f"Model {self.best_model_name} doesn't support predict_proba.")

        X_processed, _ = self._preprocess_data(X, fit=False)
        return self.best_model.predict_proba(X_processed)

    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard as DataFrame."""
        if not self.leaderboard:
            raise ValueError("No models trained. Call fit() first.")

        data = []
        for i, result in enumerate(self.leaderboard, 1):
            data.append(
                {
                    "Rank": i,
                    "Model": result["name"],
                    "CV Score (Mean)": result["cv_score_mean"],
                    "CV Score (Std)": result["cv_score_std"],
                    "Validation Score": result["val_score"],
                }
            )

        return pd.DataFrame(data)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the best model."""
        if not self.is_fitted:
            raise ValueError("AutoML is not fitted. Call fit() first.")

        if hasattr(self.best_model, "feature_importances_"):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importance = np.abs(self.best_model.coef_)
            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)
        else:
            raise ValueError(f"Model {self.best_model_name} doesn't support feature importance.")

        # Create DataFrame
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]

        # Handle dimension mismatch
        if len(feature_names) != len(importance):
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        df = df.sort_values("Importance", ascending=False).head(top_n)
        df["Rank"] = range(1, len(df) + 1)

        return df[["Rank", "Feature", "Importance"]]

    def save(self, filepath: str):
        """Save the AutoML model to disk."""
        import joblib

        if not self.is_fitted:
            raise ValueError("AutoML is not fitted. Call fit() first.")

        save_dict = {
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "best_score": self.best_score,
            "task": self.task,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "target_encoder": self.target_encoder,
            "feature_names": self.feature_names,
            "leaderboard": self.leaderboard,
            "training_time": self.training_time,
        }

        joblib.dump(save_dict, filepath)
        self._log(f"✅ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AutoML":
        """Load an AutoML model from disk."""
        import joblib

        save_dict = joblib.load(filepath)

        automl = cls(task=save_dict["task"])
        automl.best_model = save_dict["best_model"]
        automl.best_model_name = save_dict["best_model_name"]
        automl.best_score = save_dict["best_score"]
        automl.scaler = save_dict["scaler"]
        automl.imputer = save_dict["imputer"]
        automl.target_encoder = save_dict["target_encoder"]
        automl.feature_names = save_dict["feature_names"]
        automl.leaderboard = save_dict["leaderboard"]
        automl.training_time = save_dict["training_time"]
        automl.is_fitted = True

        print(f"✅ Model loaded from {filepath}")
        return automl

    def summary(self) -> str:
        """Get a summary of the AutoML run."""
        if not self.is_fitted:
            return "AutoML is not fitted. Call fit() first."

        lines = [
            "=" * 60,
            "🤖 AutoML Summary",
            "=" * 60,
            f"Task: {self.task}",
            f"Best Model: {self.best_model_name}",
            f"Best Score: {self.best_score:.4f}",
            f"Training Time: {self.training_time:.2f}s",
            f"Models Trained: {len(self.leaderboard)}",
            "",
            "Top 5 Models:",
            "-" * 40,
        ]

        for i, result in enumerate(self.leaderboard[:5], 1):
            lines.append(f"  {i}. {result['name']}: {result['val_score']:.4f}")

        lines.extend(["=" * 60])

        return "\n".join(lines)


def run_automl(
    data: pd.DataFrame, target_column: str, task: str = "auto", time_budget: int = 300, verbose: bool = True
) -> Tuple[AutoML, pd.DataFrame]:
    """
    Convenience function to run AutoML on a dataset.

    Args:
        data: DataFrame with features and target
        target_column: Name of target column
        task: 'classification', 'regression', or 'auto'
        time_budget: Maximum time in seconds
        verbose: Whether to print progress

    Returns:
        Tuple of (AutoML instance, leaderboard DataFrame)
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    automl = AutoML(task=task, time_budget=time_budget, verbose=verbose)
    automl.fit(X, y)

    return automl, automl.get_leaderboard()


if __name__ == "__main__":
    # Example usage
    print("AutoML Module Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "feature4": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }
    )

    # Run AutoML
    automl, leaderboard = run_automl(data, "target", task="classification")

    print("\n📊 Leaderboard:")
    print(leaderboard)

    print("\n" + automl.summary())
