"""
Model Explainability Module
Provides SHAP, LIME, and other interpretability methods for machine learning models.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try importing SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try importing LIME
try:
    from lime import lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ModelExplainer:
    """
    Model Explainability using SHAP and LIME.

    Provides interpretable explanations for machine learning model predictions,
    helping understand why models make specific decisions.

    Features:
    - SHAP (SHapley Additive exPlanations) values
    - LIME (Local Interpretable Model-agnostic Explanations)
    - Feature importance analysis
    - Partial dependence plots
    - Interaction effects

    Example:
        >>> explainer = ModelExplainer(model, X_train, feature_names)
        >>> explainer.explain_prediction(X_test[0])
        >>> explainer.plot_shap_summary(X_test)
    """

    def __init__(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        task: str = "classification",
        random_state: int = 42,
    ):
        """
        Initialize the ModelExplainer.

        Args:
            model: Trained sklearn-compatible model
            X_train: Training data used to fit the model
            feature_names: Names of features
            task: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.task = task
        self.random_state = random_state

        # Handle DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            self.X_train = X_train.values
        else:
            self.X_train = X_train
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None

        # Check available libraries
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not installed. Install with: pip install shap")
        if not LIME_AVAILABLE:
            print("⚠️ LIME not installed. Install with: pip install lime")

    def _init_shap_explainer(self, background_samples: int = 100):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        if self.shap_explainer is not None:
            return

        # Use a sample of training data as background
        if len(self.X_train) > background_samples:
            background = shap.sample(self.X_train, background_samples, random_state=self.random_state)
        else:
            background = self.X_train

        # Try different explainer types based on model
        model_type = type(self.model).__name__

        try:
            if "Tree" in model_type or "Forest" in model_type or "Boost" in model_type:
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif "Linear" in model_type or "Logistic" in model_type or "Ridge" in model_type:
                # Linear models
                self.shap_explainer = shap.LinearExplainer(self.model, background)
            else:
                # Use KernelExplainer as fallback (slower but universal)
                if self.task == "classification" and hasattr(self.model, "predict_proba"):
                    self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
                else:
                    self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
        except Exception as e:
            # Final fallback
            print(f"⚠️ Using Kernel explainer (slower): {str(e)[:50]}")
            self.shap_explainer = shap.KernelExplainer(self.model.predict, background)

    def _init_lime_explainer(self):
        """Initialize LIME explainer."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")

        if self.lime_explainer is not None:
            return

        mode = "classification" if self.task == "classification" else "regression"

        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=["Class 0", "Class 1"] if self.task == "classification" else None,
            mode=mode,
            random_state=self.random_state,
        )

    def compute_shap_values(self, X: Union[pd.DataFrame, np.ndarray], max_samples: int = 500) -> np.ndarray:
        """
        Compute SHAP values for given data.

        Args:
            X: Data to explain
            max_samples: Maximum samples to compute SHAP for

        Returns:
            SHAP values array
        """
        self._init_shap_explainer()

        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Limit samples
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]

        print(f"🔍 Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.shap_explainer.shap_values(X)

        return self.shap_values

    def explain_prediction(
        self, instance: Union[pd.DataFrame, np.ndarray, pd.Series], method: str = "shap", num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            instance: Single instance to explain
            method: 'shap' or 'lime'
            num_features: Number of top features to show

        Returns:
            Dictionary with explanation details
        """
        # Handle different input types
        if isinstance(instance, pd.DataFrame):
            instance_array = instance.values.flatten()
        elif isinstance(instance, pd.Series):
            instance_array = instance.values
        else:
            instance_array = instance.flatten()

        explanation = {"method": method, "prediction": None, "feature_contributions": {}, "top_features": []}

        # Get prediction
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(instance_array.reshape(1, -1))[0]
            pred = self.model.predict(instance_array.reshape(1, -1))[0]
            explanation["prediction"] = {"class": int(pred), "probability": float(max(proba))}
        else:
            pred = self.model.predict(instance_array.reshape(1, -1))[0]
            explanation["prediction"] = {"value": float(pred)}

        if method == "shap":
            self._init_shap_explainer()
            shap_values = self.shap_explainer.shap_values(instance_array.reshape(1, -1))

            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            shap_values = shap_values.flatten()

            # Create feature contributions
            for i, name in enumerate(self.feature_names):
                explanation["feature_contributions"][name] = {
                    "value": float(instance_array[i]),
                    "shap_value": float(shap_values[i]),
                }

            # Sort by absolute contribution
            sorted_features = sorted(
                explanation["feature_contributions"].items(), key=lambda x: abs(x[1]["shap_value"]), reverse=True
            )
            explanation["top_features"] = sorted_features[:num_features]

        elif method == "lime":
            self._init_lime_explainer()

            if self.task == "classification":
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

            lime_exp = self.lime_explainer.explain_instance(instance_array, predict_fn, num_features=num_features)

            # Extract feature contributions
            for feature, contribution in lime_exp.as_list():
                explanation["feature_contributions"][feature] = contribution

            explanation["top_features"] = lime_exp.as_list()[:num_features]

        return explanation

    def plot_shap_summary(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 20,
        plot_type: str = "dot",
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Plot SHAP summary plot.

        Args:
            X: Data to explain
            max_display: Maximum features to display
            plot_type: 'dot', 'bar', or 'violin'
            figsize: Figure size
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed.")

        # Compute SHAP values if not already done
        if self.shap_values is None:
            self.compute_shap_values(X)

        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Handle multi-class
        shap_values = self.shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Ensure dimensions match
        if len(X_array) > len(shap_values):
            X_array = X_array[: len(shap_values)]

        plt.figure(figsize=figsize)
        shap.summary_plot(
            shap_values,
            X_array,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False,
        )
        plt.tight_layout()
        return plt.gcf()

    def plot_shap_waterfall(self, instance_idx: int = 0, max_display: int = 10):
        """
        Plot SHAP waterfall plot for a single prediction.

        Args:
            instance_idx: Index of instance to explain
            max_display: Maximum features to display
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed.")

        if self.shap_values is None:
            raise ValueError("Call compute_shap_values first.")

        shap_values = self.shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=(
                self.shap_explainer.expected_value
                if not isinstance(self.shap_explainer.expected_value, list)
                else self.shap_explainer.expected_value[1]
            ),
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return plt.gcf()

    def plot_shap_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Plot SHAP dependence plot for a feature.

        Args:
            feature: Feature to analyze
            interaction_feature: Feature for interaction
            X: Data (uses training data if None)
            figsize: Figure size
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed.")

        if self.shap_values is None:
            raise ValueError("Call compute_shap_values first.")

        shap_values = self.shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Get feature index
        if feature in self.feature_names:
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = int(feature)

        # Handle data
        if X is None:
            X_array = self.X_train[: len(shap_values)]
        elif isinstance(X, pd.DataFrame):
            X_array = X.values[: len(shap_values)]
        else:
            X_array = X[: len(shap_values)]

        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_array,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.tight_layout()
        return plt.gcf()

    def plot_feature_importance(
        self, importance_type: str = "shap", top_n: int = 20, figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot feature importance.

        Args:
            importance_type: 'shap' or 'model'
            top_n: Number of features to display
            figsize: Figure size
        """
        if importance_type == "shap":
            if self.shap_values is None:
                raise ValueError("Call compute_shap_values first.")

            shap_values = self.shap_values
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)
        else:
            # Use model's feature importance
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_)
                if len(importance.shape) > 1:
                    importance = importance.mean(axis=0)
            else:
                raise ValueError("Model doesn't support feature importance.")

        # Create DataFrame
        df = pd.DataFrame({"Feature": self.feature_names[: len(importance)], "Importance": importance})
        df = df.sort_values("Importance", ascending=True).tail(top_n)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(df["Feature"], df["Importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance ({importance_type.upper()})")
        plt.tight_layout()

        return fig

    def get_feature_importance_df(self, importance_type: str = "shap") -> pd.DataFrame:
        """
        Get feature importance as DataFrame.

        Args:
            importance_type: 'shap' or 'model'

        Returns:
            DataFrame with feature importance
        """
        if importance_type == "shap":
            if self.shap_values is None:
                raise ValueError("Call compute_shap_values first.")

            shap_values = self.shap_values
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            importance = np.abs(shap_values).mean(axis=0)
        else:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_)
                if len(importance.shape) > 1:
                    importance = importance.mean(axis=0)
            else:
                raise ValueError("Model doesn't support feature importance.")

        df = pd.DataFrame({"Feature": self.feature_names[: len(importance)], "Importance": importance})
        df = df.sort_values("Importance", ascending=False)
        df["Rank"] = range(1, len(df) + 1)

        return df[["Rank", "Feature", "Importance"]]

    def explain_global(self, X: Union[pd.DataFrame, np.ndarray], method: str = "shap") -> Dict[str, Any]:
        """
        Get global explanation summary.

        Args:
            X: Data to explain
            method: Explanation method ('shap')

        Returns:
            Dictionary with global explanation
        """
        explanation = {
            "method": method,
            "n_samples": len(X) if hasattr(X, "__len__") else 1,
            "n_features": len(self.feature_names),
            "feature_importance": {},
            "top_features": [],
        }

        if method == "shap":
            # Compute SHAP values
            self.compute_shap_values(X)

            shap_values = self.shap_values
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)

            for i, name in enumerate(self.feature_names):
                if i < len(importance):
                    explanation["feature_importance"][name] = float(importance[i])

            # Sort by importance
            sorted_features = sorted(explanation["feature_importance"].items(), key=lambda x: x[1], reverse=True)
            explanation["top_features"] = sorted_features[:10]

        return explanation

    def generate_report(self, X: Union[pd.DataFrame, np.ndarray], output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive explainability report.

        Args:
            X: Data to explain
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report_lines = [
            "=" * 70,
            "🔍 MODEL EXPLAINABILITY REPORT",
            "=" * 70,
            "",
            f"Model Type: {type(self.model).__name__}",
            f"Task: {self.task}",
            f"Number of Features: {len(self.feature_names)}",
            f"Samples Analyzed: {len(X) if hasattr(X, '__len__') else 1}",
            "",
            "=" * 70,
            "FEATURE IMPORTANCE (SHAP)",
            "-" * 70,
        ]

        # Compute SHAP values
        try:
            global_exp = self.explain_global(X, method="shap")

            for i, (feature, importance) in enumerate(global_exp["top_features"][:15], 1):
                report_lines.append(f"  {i:2}. {feature:30} {importance:.4f}")
        except Exception as e:
            report_lines.append(f"  Could not compute SHAP values: {str(e)}")

        report_lines.extend(
            [
                "",
                "=" * 70,
                "INTERPRETATION GUIDELINES",
                "-" * 70,
                "• Higher SHAP values indicate stronger feature impact on predictions",
                "• Positive SHAP values push prediction higher, negative push lower",
                "• Features with consistent high impact across samples are most important",
                "• Use dependence plots to understand feature interactions",
                "",
                "=" * 70,
            ]
        )

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"✅ Report saved to {output_path}")

        return report


def explain_model(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    X_explain: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    task: str = "classification",
) -> Tuple[ModelExplainer, Dict[str, Any]]:
    """
    Convenience function to explain a model.

    Args:
        model: Trained model
        X_train: Training data
        X_explain: Data to explain
        feature_names: Feature names
        task: 'classification' or 'regression'

    Returns:
        Tuple of (ModelExplainer, global explanation dict)
    """
    explainer = ModelExplainer(model=model, X_train=X_train, feature_names=feature_names, task=task)

    global_explanation = explainer.explain_global(X_explain)

    return explainer, global_explanation


if __name__ == "__main__":
    # Example usage
    print("Model Explainability Module Demo")
    print("=" * 60)

    # Create sample data and model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    n_samples = 500

    X = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, n_samples),
            "income": np.random.normal(50000, 15000, n_samples),
            "credit_score": np.random.normal(650, 100, n_samples),
            "years_employed": np.random.randint(0, 30, n_samples),
        }
    )
    y = (X["income"] > 50000).astype(int)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Explain model
    explainer = ModelExplainer(model=model, X_train=X_train, task="classification")

    # Compute SHAP values
    print("\nComputing SHAP values...")
    explainer.compute_shap_values(X_test)

    # Get feature importance
    print("\nFeature Importance:")
    importance_df = explainer.get_feature_importance_df()
    print(importance_df)

    # Explain single prediction
    print("\nSingle Prediction Explanation:")
    explanation = explainer.explain_prediction(X_test.iloc[0])
    print(f"Prediction: {explanation['prediction']}")
    print("Top contributing features:")
    for feature, values in explanation["top_features"][:5]:
        print(f"  - {feature}: SHAP={values['shap_value']:.4f}")
