"""
Machine Learning Models Module
Provides comprehensive ML model training, evaluation, and comparison.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)

# Regression models
# Classification models
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (  # Classification metrics; Regression metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Sklearn imports
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM (optional)
try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings("ignore")


class MLModels:
    """
    A comprehensive class for machine learning model training and evaluation.

    Attributes:
        data (pd.DataFrame): The dataset
        target_column (str): Name of the target column
        task_type (str): 'classification' or 'regression'
        X_train, X_test, y_train, y_test: Train-test split data
        models (dict): Trained models
        results (dict): Model evaluation results
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str = "auto",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the MLModels class.

        Args:
            data (pd.DataFrame): The dataset
            target_column (str): Name of the target column
            task_type (str): 'classification', 'regression', or 'auto'
            test_size (float): Test set proportion
            random_state (int): Random seed for reproducibility
        """
        self.data = data.copy()
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size

        # Determine task type
        if task_type == "auto":
            unique_values = self.data[target_column].nunique()
            if unique_values <= 10 or self.data[target_column].dtype == "object":
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        else:
            self.task_type = task_type

        # Initialize storage
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = None

        # Prepare data
        self._prepare_data()

        print(f"✅ MLModels initialized for {self.task_type}")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Features: {self.X_train.shape[1]}")

    def _prepare_data(self):
        """Prepare data for training."""
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Handle categorical target for classification
        if self.task_type == "classification" and y.dtype == "object":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task_type == "classification" else None,
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get dictionary of available models for the task type.

        Returns:
            Dict containing model names and classes
        """
        if self.task_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
                "SVM": SVC(probability=True, random_state=self.random_state),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Naive Bayes": GaussianNB(),
                "AdaBoost": AdaBoostClassifier(random_state=self.random_state, algorithm="SAMME"),
            }

            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBClassifier(
                    random_state=self.random_state, eval_metric="logloss", use_label_encoder=False
                )
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = LGBMClassifier(random_state=self.random_state, verbose=-1)

        else:  # regression
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=self.random_state),
                "Lasso": Lasso(random_state=self.random_state),
                "ElasticNet": ElasticNet(random_state=self.random_state),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
                "AdaBoost": AdaBoostRegressor(random_state=self.random_state),
            }

            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBRegressor(random_state=self.random_state)
            if LIGHTGBM_AVAILABLE:
                models["LightGBM"] = LGBMRegressor(random_state=self.random_state, verbose=-1)

        return models

    def train_model(self, model_name: str, model: Optional[Any] = None, use_scaled: bool = True) -> Dict[str, Any]:
        """
        Train a single model.

        Args:
            model_name (str): Name of the model
            model: Model instance (optional, uses default if not provided)
            use_scaled (bool): Whether to use scaled features

        Returns:
            Dict containing training results
        """
        if model is None:
            available_models = self.get_available_models()
            if model_name not in available_models:
                raise ValueError(f"Model '{model_name}' not available")
            model = available_models[model_name]

        # Select features
        X_train = self.X_train_scaled if use_scaled else self.X_train
        X_test = self.X_test_scaled if use_scaled else self.X_test

        # Train model
        model.fit(X_train, self.y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Store model
        self.models[model_name] = model

        # Evaluate
        if self.task_type == "classification":
            results = self._evaluate_classification(model, model_name, y_pred, y_train_pred, X_test)
        else:
            results = self._evaluate_regression(model, model_name, y_pred, y_train_pred)

        self.results[model_name] = results

        return results

    def train_all_models(self, use_scaled: bool = True, models_to_train: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Train all available models.

        Args:
            use_scaled (bool): Whether to use scaled features
            models_to_train (List[str]): Specific models to train

        Returns:
            pd.DataFrame: Comparison of all models
        """
        available_models = self.get_available_models()

        if models_to_train:
            models_to_train = {k: v for k, v in available_models.items() if k in models_to_train}
        else:
            models_to_train = available_models

        print(f"🚀 Training {len(models_to_train)} models...")
        print("-" * 50)

        for name, model in models_to_train.items():
            try:
                print(f"Training {name}...", end=" ")
                self.train_model(name, model, use_scaled)
                print("✅")
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")

        print("-" * 50)
        print("✅ Training complete!")

        return self.compare_models()

    def _evaluate_classification(
        self, model, model_name: str, y_pred: np.ndarray, y_train_pred: np.ndarray, X_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate classification model."""
        results = {
            "model_name": model_name,
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(self.y_test, y_pred, average="weighted", zero_division=0),
            "train_accuracy": accuracy_score(self.y_train, y_train_pred),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
        }

        # ROC AUC for binary classification
        n_classes = len(np.unique(self.y_train))
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                results["roc_auc"] = roc_auc_score(self.y_test, y_proba)
            except Exception:
                results["roc_auc"] = None
        else:
            results["roc_auc"] = None

        # Overfitting check
        results["overfit_ratio"] = results["train_accuracy"] / results["accuracy"]

        return results

    def _evaluate_regression(
        self, model, model_name: str, y_pred: np.ndarray, y_train_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression model."""
        results = {
            "model_name": model_name,
            "r2_score": r2_score(self.y_test, y_pred),
            "mse": mean_squared_error(self.y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_pred)),
            "mae": mean_absolute_error(self.y_test, y_pred),
            "mape": mean_absolute_percentage_error(self.y_test, y_pred) * 100,
            "train_r2": r2_score(self.y_train, y_train_pred),
            "train_rmse": np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
        }

        # Overfitting check
        results["overfit_ratio"] = results["train_r2"] / (results["r2_score"] + 1e-8)

        return results

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.

        Returns:
            pd.DataFrame: Model comparison table
        """
        if not self.results:
            raise ValueError("No models trained yet. Use train_all_models() first.")

        if self.task_type == "classification":
            comparison_data = []
            for name, res in self.results.items():
                comparison_data.append(
                    {
                        "Model": name,
                        "Accuracy": res["accuracy"],
                        "Precision": res["precision"],
                        "Recall": res["recall"],
                        "F1 Score": res["f1_score"],
                        "ROC AUC": res.get("roc_auc"),
                        "Train Accuracy": res["train_accuracy"],
                        "Overfit Ratio": res["overfit_ratio"],
                    }
                )
            df = pd.DataFrame(comparison_data)
            df = df.sort_values("F1 Score", ascending=False)

            # Update best model
            self.best_model_name = df.iloc[0]["Model"]
            self.best_model = self.models[self.best_model_name]

        else:  # regression
            comparison_data = []
            for name, res in self.results.items():
                comparison_data.append(
                    {
                        "Model": name,
                        "R² Score": res["r2_score"],
                        "RMSE": res["rmse"],
                        "MAE": res["mae"],
                        "MAPE (%)": res["mape"],
                        "Train R²": res["train_r2"],
                        "Train RMSE": res["train_rmse"],
                        "Overfit Ratio": res["overfit_ratio"],
                    }
                )
            df = pd.DataFrame(comparison_data)
            df = df.sort_values("R² Score", ascending=False)

            # Update best model
            self.best_model_name = df.iloc[0]["Model"]
            self.best_model = self.models[self.best_model_name]

        print(f"\n🏆 Best Model: {self.best_model_name}")
        return df

    def cross_validate(
        self, model_name: Optional[str] = None, cv: int = 5, scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Args:
            model_name (str): Name of the model to evaluate
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric

        Returns:
            Dict containing cross-validation results
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]

        if scoring is None:
            scoring = "f1_weighted" if self.task_type == "classification" else "r2"

        # Use stratified K-fold for classification
        if self.task_type == "classification":
            kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=kfold, scoring=scoring)

        results = {
            "model_name": model_name,
            "cv_folds": cv,
            "scoring": scoring,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
            "confidence_interval": (scores.mean() - 1.96 * scores.std(), scores.mean() + 1.96 * scores.std()),
        }

        print(f"\n📊 Cross-Validation Results for {model_name}")
        print(f"   Mean Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        print(f"   95% CI: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")

        return results

    def hyperparameter_tuning(
        self, model_name: str, param_grid: Dict[str, List], search_type: str = "grid", cv: int = 5, n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.

        Args:
            model_name (str): Name of the model
            param_grid (Dict): Parameter grid for search
            search_type (str): 'grid' or 'random'
            cv (int): Number of cross-validation folds

            n_iter (int): Number of iterations for random search

        Returns:
            Dict containing best parameters and results
        """
        available_models = self.get_available_models()
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not available")

        model = available_models[model_name]

        # Set up scoring
        if self.task_type == "classification":
            scoring = "f1_weighted"
        else:
            scoring = "r2"

        # Perform search
        if search_type == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )

        print(f"\n🔧 Tuning hyperparameters for {model_name}...")
        search.fit(self.X_train_scaled, self.y_train)

        # Get results
        results = {
            "model_name": model_name,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }

        # Train with best parameters
        best_model = search.best_estimator_
        self.models[f"{model_name}_tuned"] = best_model

        # Evaluate tuned model
        y_pred = best_model.predict(self.X_test_scaled)
        y_train_pred = best_model.predict(self.X_train_scaled)

        if self.task_type == "classification":
            tuned_results = self._evaluate_classification(
                best_model, f"{model_name}_tuned", y_pred, y_train_pred, self.X_test_scaled
            )
        else:
            tuned_results = self._evaluate_regression(best_model, f"{model_name}_tuned", y_pred, y_train_pred)

        self.results[f"{model_name}_tuned"] = tuned_results
        results["evaluation"] = tuned_results

        print(f"\n✅ Best Parameters: {search.best_params_}")
        print(f"   Best CV Score: {search.best_score_:.4f}")

        return results

    def get_feature_importance(self, model_name: Optional[str] = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a trained model.

        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance table
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]

        # Get feature importance based on model type
        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Linear models
            importance = np.abs(model.coef_)
            if len(importance.shape) > 1:
                importance = importance.mean(axis=0)
        else:
            raise ValueError(f"Model '{model_name}' doesn't support feature importance")

        # Create DataFrame
        importance_df = pd.DataFrame({"Feature": self.feature_names, "Importance": importance})
        importance_df = importance_df.sort_values("Importance", ascending=False).head(top_n)
        importance_df["Rank"] = range(1, len(importance_df) + 1)

        print(f"\n📊 Top {top_n} Features for {model_name}:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['Rank']}. {row['Feature']}: {row['Importance']:.4f}")

        return importance_df

    def get_classification_report(self, model_name: Optional[str] = None) -> str:
        """
        Get detailed classification report.

        Args:
            model_name (str): Name of the model

        Returns:
            str: Classification report
        """
        if self.task_type != "classification":
            raise ValueError("Classification report only available for classification tasks")

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]
        y_pred = model.predict(self.X_test_scaled)

        # Get class names
        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_
        else:
            target_names = [str(c) for c in np.unique(self.y_test)]

        report = classification_report(self.y_test, y_pred, target_names=target_names)

        print(f"\n📋 Classification Report for {model_name}")
        print("=" * 60)
        print(report)

        return report

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], model_name: Optional[str] = None, return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            X: Features to predict
            model_name (str): Name of the model to use
            return_proba (bool): Return probabilities for classification

        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]

        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            # Ensure same columns as training
            X = pd.get_dummies(X, drop_first=True)
            # Align columns with training data
            missing_cols = set(self.feature_names) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        if return_proba and hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X_scaled)
        else:
            predictions = model.predict(X_scaled)

            # Decode labels if needed
            if self.label_encoder is not None and not return_proba:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))

        return predictions

    def save_model(self, filepath: str, model_name: Optional[str] = None, include_scaler: bool = True) -> None:
        """
        Save a trained model to disk.

        Args:
            filepath (str): Path to save the model
            model_name (str): Name of the model to save
            include_scaler (bool): Whether to include the scaler
        """
        import joblib

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        save_dict = {
            "model": self.models[model_name],
            "model_name": model_name,
            "feature_names": self.feature_names,
            "task_type": self.task_type,
            "label_encoder": self.label_encoder,
        }

        if include_scaler:
            save_dict["scaler"] = self.scaler

        joblib.dump(save_dict, filepath)
        print(f"✅ Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """
        Load a saved model from disk.

        Args:
            filepath (str): Path to the saved model

        Returns:
            Dict containing the model and metadata
        """
        import joblib

        model_dict = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model_dict

    def plot_confusion_matrix(
        self, model_name: Optional[str] = None, figsize: Tuple[int, int] = (8, 6), cmap: str = "Blues"
    ):
        """
        Plot confusion matrix for a classification model.

        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size
            cmap (str): Colormap

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.task_type != "classification":
            raise ValueError("Confusion matrix only available for classification")

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        cm = self.results[model_name]["confusion_matrix"]

        # Get class names
        if self.label_encoder is not None:
            labels = self.label_encoder.classes_
        else:
            labels = [str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        return fig

    def plot_roc_curve(self, model_names: Optional[List[str]] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot ROC curves for classification models.

        Args:
            model_names (List[str]): Models to include
            figsize (tuple): Figure size

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        if self.task_type != "classification":
            raise ValueError("ROC curve only available for binary classification")

        n_classes = len(np.unique(self.y_train))
        if n_classes != 2:
            raise ValueError("ROC curve only available for binary classification")

        if model_names is None:
            model_names = list(self.models.keys())

        fig, ax = plt.subplots(figsize=figsize)

        for name in model_names:
            if name not in self.models:
                continue

            model = self.models[name]
            if not hasattr(model, "predict_proba"):
                continue

            y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)

            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_feature_importance(
        self, model_name: Optional[str] = None, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot feature importance.

        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to plot
            figsize (tuple): Figure size

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        importance_df = self.get_feature_importance(model_name, top_n)

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title(f"Feature Importance - {model_name or self.best_model_name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()

        return fig

    def plot_predictions_vs_actual(self, model_name: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot predictions vs actual values for regression.

        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        if self.task_type != "regression":
            raise ValueError("This plot is only available for regression tasks")

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]
        y_pred = model.predict(self.X_test_scaled)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.y_test, y_pred, alpha=0.5)

        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predictions vs Actual - {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_residuals(self, model_name: Optional[str] = None, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot residual analysis for regression.

        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        if self.task_type != "regression":
            raise ValueError("Residual plot only available for regression tasks")

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]
        y_pred = model.predict(self.X_test_scaled)
        residuals = self.y_test - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predicted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residual Distribution")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Residual Analysis - {model_name}")
        plt.tight_layout()

        return fig

    def create_ensemble(
        self, model_names: List[str], ensemble_name: str = "Ensemble", voting: str = "soft"
    ) -> Dict[str, Any]:
        """
        Create an ensemble from trained models.

        Args:
            model_names (List[str]): Names of models to ensemble
            ensemble_name (str): Name for the ensemble
            voting (str): 'soft' or 'hard' voting (classification only)

        Returns:
            Dict containing ensemble results
        """
        if self.task_type != "classification":
            raise ValueError("Ensemble voting currently only supports classification")

        # Get trained models
        estimators = []
        for name in model_names:
            if name not in self.models:
                raise ValueError(f"Model '{name}' not trained yet.")
            estimators.append((name, self.models[name]))

        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting=voting)

        # Train and evaluate
        print(f"\n🎯 Creating ensemble from: {', '.join(model_names)}")
        results = self.train_model(ensemble_name, ensemble, use_scaled=True)

        return results

    def get_learning_curves(
        self,
        model_name: Optional[str] = None,
        train_sizes: Optional[np.ndarray] = None,
        cv: int = 5,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Generate and plot learning curves.

        Args:
            model_name (str): Name of the model
            train_sizes: Array of training set sizes
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size

        Returns:
            matplotlib figure and learning curve data
        """
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            # Get fresh model instance
            available = self.get_available_models()
            if model_name not in available:
                raise ValueError(f"Model '{model_name}' not available")
            model = available[model_name]
        else:
            # Clone the trained model type
            model = self.get_available_models().get(model_name)
            if model is None:
                raise ValueError(f"Cannot get fresh instance of '{model_name}'")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Set scoring
        scoring = "f1_weighted" if self.task_type == "classification" else "r2"

        print(f"\n📈 Generating learning curves for {model_name}...")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            self.X_train_scaled,
            self.y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state,
        )

        # Calculate mean and std
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
        ax.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Training Score")
        ax.plot(train_sizes_abs, val_mean, "o-", color="orange", label="Validation Score")

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Score")
        ax.set_title(f"Learning Curves - {model_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        results = {
            "train_sizes": train_sizes_abs,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "train_mean": train_mean,
            "val_mean": val_mean,
        }

        return fig, results

    def get_model_summary(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a trained model.

        Args:
            model_name (str): Name of the model

        Returns:
            Dict containing model summary
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet.")

        model = self.models[model_name]
        results = self.results[model_name]

        summary = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "task_type": self.task_type,
            "n_features": len(self.feature_names),
            "training_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "metrics": {k: v for k, v in results.items() if k not in ["confusion_matrix", "model_name"]},
        }

        # Add model parameters
        if hasattr(model, "get_params"):
            summary["parameters"] = model.get_params()

        print(f"\n📊 Model Summary: {model_name}")
        print("=" * 50)
        print(f"   Model Type: {summary['model_type']}")
        print(f"   Task Type: {summary['task_type']}")
        print(f"   Features: {summary['n_features']}")
        print(f"   Training Samples: {summary['training_samples']}")
        print(f"   Test Samples: {summary['test_samples']}")
        print("\n   Metrics:")
        for metric, value in summary["metrics"].items():
            if isinstance(value, (int, float)):
                print(f"      {metric}: {value:.4f}")

        return summary

    def get_default_param_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get default hyperparameter grids for all models.

        Returns:
            Dict containing param grids for each model
        """
        if self.task_type == "classification":
            param_grids = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0],
                },
                "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto", 0.1, 1]},
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],
                },
            }
        else:
            param_grids = {
                "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
                "Lasso": {"alpha": [0.01, 0.1, 1, 10, 100]},
                "ElasticNet": {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0],
                },
                "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto"]},
                "KNN": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
                "Decision Tree": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            }

        return param_grids

    def auto_tune_best_models(
        self, top_n: int = 3, search_type: str = "random", cv: int = 5, n_iter: int = 20
    ) -> pd.DataFrame:
        """
        Automatically tune the top N performing models.

        Args:
            top_n (int): Number of top models to tune
            search_type (str): 'grid' or 'random'
            cv (int): Number of cross-validation folds
            n_iter (int): Iterations for random search

        Returns:
            pd.DataFrame: Comparison of tuned models
        """
        if not self.results:
            raise ValueError("No models trained yet. Use train_all_models() first.")

        # Get top models
        comparison = self.compare_models()
        top_models = comparison["Model"].head(top_n).tolist()

        param_grids = self.get_default_param_grids()

        print(f"\n🔧 Auto-tuning top {top_n} models: {', '.join(top_models)}")
        print("=" * 60)

        for model_name in top_models:
            if model_name in param_grids:
                try:
                    self.hyperparameter_tuning(
                        model_name, param_grids[model_name], search_type=search_type, cv=cv, n_iter=n_iter
                    )
                except Exception as e:
                    print(f"❌ Error tuning {model_name}: {str(e)[:50]}")

        return self.compare_models()


# Convenience function for quick model training
def quick_ml_pipeline(
    data: pd.DataFrame, target_column: str, task_type: str = "auto", test_size: float = 0.2, tune_top_n: int = 0
) -> Tuple[MLModels, pd.DataFrame]:
    """
    Quick ML pipeline for rapid model comparison.

    Args:
        data (pd.DataFrame): The dataset
        target_column (str): Target column name
        task_type (str): 'classification', 'regression', or 'auto'
        test_size (float): Test set proportion
        tune_top_n (int): Number of top models to tune (0 to skip tuning)

    Returns:
        Tuple of (MLModels instance, comparison DataFrame)
    """
    # Initialize
    ml = MLModels(data=data, target_column=target_column, task_type=task_type, test_size=test_size)

    # Train all models
    comparison = ml.train_all_models()

    # Optionally tune top models
    if tune_top_n > 0:
        comparison = ml.auto_tune_best_models(top_n=tune_top_n)

    return ml, comparison
