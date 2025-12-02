"""
Data Visualization Module
Provides comprehensive visualization utilities for data analysis.
"""

import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class DataVisualizer:
    """
    A comprehensive class for data visualization.

    Attributes:
        data (pd.DataFrame): The dataset to visualize
        numerical_columns (List[str]): List of numerical column names
        categorical_columns (List[str]): List of categorical column names
        figsize (Tuple[int, int]): Default figure size
    """

    def __init__(self, data: pd.DataFrame, figsize: Tuple[int, int] = (10, 6), style: str = "whitegrid"):
        """
        Initialize the DataVisualizer with a dataset.

        Args:
            data (pd.DataFrame): The dataset to visualize
            figsize (Tuple[int, int]): Default figure size
            style (str): Seaborn style to use
        """
        self.data = data.copy()
        self.figsize = figsize
        self._identify_column_types()
        sns.set_style(style)

    def _identify_column_types(self):
        """Automatically identify numerical and categorical columns."""
        self.numerical_columns = self.data.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        self.categorical_columns = self.data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def plot_distribution(
        self,
        column: str,
        kind: str = "hist",
        bins: int = 30,
        kde: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        color: str = "#3498db",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of a single column.

        Args:
            column (str): Column to plot
            kind (str): Type of plot ('hist', 'kde', 'box', 'violin')
            bins (int): Number of bins for histogram
            kde (bool): Whether to overlay KDE on histogram
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            color (str): Color for the plot
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        if kind == "hist":
            sns.histplot(data=self.data, x=column, bins=bins, kde=kde, color=color, ax=ax)
        elif kind == "kde":
            sns.kdeplot(data=self.data, x=column, fill=True, color=color, ax=ax)
        elif kind == "box":
            sns.boxplot(data=self.data, y=column, color=color, ax=ax)
        elif kind == "violin":
            sns.violinplot(data=self.data, y=column, color=color, ax=ax)

        ax.set_title(title or f"Distribution of {column}", fontsize=14, fontweight="bold")
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency" if kind == "hist" else "Density", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_distributions_grid(
        self,
        columns: Optional[List[str]] = None,
        kind: str = "hist",
        cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distributions of multiple columns in a grid.

        Args:
            columns (List[str]): Columns to plot (default: all numerical)
            kind (str): Type of plot ('hist', 'kde', 'box')
            cols (int): Number of columns in grid
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        columns = columns or self.numerical_columns
        n_plots = len(columns)
        rows = (n_plots + cols - 1) // cols

        fig_height = rows * 4
        fig_width = cols * 5
        figsize = figsize or (fig_width, fig_height)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        colors = sns.color_palette("husl", n_plots)

        for idx, (col, color) in enumerate(zip(columns, colors)):
            ax = axes[idx]

            if kind == "hist":
                sns.histplot(data=self.data, x=col, kde=True, color=color, ax=ax)
            elif kind == "kde":
                sns.kdeplot(data=self.data, x=col, fill=True, color=color, ax=ax)
            elif kind == "box":
                sns.boxplot(data=self.data, y=col, color=color, ax=ax)

            ax.set_title(col, fontsize=12, fontweight="bold")
            ax.set_xlabel("")

        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_correlation_heatmap(
        self,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        annot: bool = True,
        cmap: str = "RdBu_r",
        figsize: Optional[Tuple[int, int]] = None,
        mask_upper: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot correlation heatmap.

        Args:
            columns (List[str]): Columns to include
            method (str): Correlation method
            annot (bool): Whether to annotate cells
            cmap (str): Colormap to use
            figsize (Tuple[int, int]): Figure size
            mask_upper (bool): Whether to mask upper triangle
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        columns = columns or self.numerical_columns
        corr_matrix = self.data[columns].corr(method=method)

        fig, ax = plt.subplots(figsize=figsize or (12, 10))

        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt=".2f",
            ax=ax,
        )

        ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_scatter(
        self,
        x: str,
        y: str,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        add_regression: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a scatter plot.

        Args:
            x (str): X-axis column
            y (str): Y-axis column
            hue (str): Column for color encoding
            size (str): Column for size encoding
            add_regression (bool): Whether to add regression line
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        if add_regression and hue is None:
            sns.regplot(data=self.data, x=x, y=y, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}, ax=ax)
        else:
            sns.scatterplot(data=self.data, x=x, y=y, hue=hue, size=size, alpha=0.7, ax=ax)

            if add_regression:
                z = np.polyfit(self.data[x].dropna(), self.data[y].dropna(), 1)
                p = np.poly1d(z)
                x_line = np.linspace(self.data[x].min(), self.data[x].max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend")
                ax.legend()

        ax.set_title(title or f"{y} vs {x}", fontsize=14, fontweight="bold")
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_scatter_matrix(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a scatter matrix (pairplot).

        Args:
            columns (List[str]): Columns to include
            hue (str): Column for color encoding
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        columns = columns or self.numerical_columns[:5]  # Limit to 5 for readability

        if hue and hue not in columns:
            plot_data = self.data[columns + [hue]]
        else:
            plot_data = self.data[columns]

        g = sns.pairplot(plot_data, hue=hue, diag_kind="kde", plot_kws={"alpha": 0.6}, height=2.5)

        g.fig.suptitle("Scatter Matrix", y=1.02, fontsize=16, fontweight="bold")

        if save_path:
            g.fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return g.fig

    def plot_categorical(
        self,
        column: str,
        kind: str = "count",
        hue: Optional[str] = None,
        top_n: int = 10,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        rotation: int = 45,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot categorical variable.

        Args:
            column (str): Column to plot
            kind (str): Type of plot ('count', 'pie')
            hue (str): Column for color encoding
            top_n (int): Number of top categories to show
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            rotation (int): X-label rotation
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        if kind == "count":
            # Get top N categories
            top_categories = self.data[column].value_counts().head(top_n).index
            plot_data = self.data[self.data[column].isin(top_categories)]

            sns.countplot(data=plot_data, x=column, hue=hue, order=top_categories, ax=ax)
            ax.set_ylabel("Count", fontsize=12)

        elif kind == "pie":
            value_counts = self.data[column].value_counts().head(top_n)
            colors = sns.color_palette("husl", len(value_counts))

            ax.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.axis("equal")

        ax.set_title(title or f"Distribution of {column}", fontsize=14, fontweight="bold")

        if kind == "count":
            ax.set_xlabel(column, fontsize=12)
            plt.xticks(rotation=rotation)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_boxplot_by_category(
        self,
        numerical_col: str,
        categorical_col: str,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create boxplot of numerical variable grouped by category.

        Args:
            numerical_col (str): Numerical column for y-axis
            categorical_col (str): Categorical column for grouping
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        sns.boxplot(data=self.data, x=categorical_col, y=numerical_col, ax=ax)

        ax.set_title(title or f"{numerical_col} by {categorical_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(categorical_col, fontsize=12)
        ax.set_ylabel(numerical_col, fontsize=12)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_time_series(
        self,
        date_column: str,
        value_column: str,
        freq: str = "D",
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot time series data.

        Args:
            date_column (str): Date column name
            value_column (str): Value column to plot
            freq (str): Frequency for resampling
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or (14, 6))

        # Ensure datetime type
        df = self.data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

        # Resample if needed
        if freq:
            df = df[value_column].resample(freq).mean()
        else:
            df = df[value_column]

        ax.plot(df.index, df.values, linewidth=2, color="#3498db")
        ax.fill_between(df.index, df.values, alpha=0.3, color="#3498db")

        ax.set_title(title or f"{value_column} Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(value_column, fontsize=12)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_feature_importance(
        self,
        importance_dict: dict,
        top_n: int = 15,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Feature Importance",
        color: str = "#3498db",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            importance_dict (dict): Dictionary of feature names and importance scores
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            color (str): Bar color
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize or (10, 8))

        # Sort and get top N
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

        features = [x[0] for x in sorted_importance]
        importances = [x[1] for x in sorted_importance]

        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_missing_values(
        self, figsize: Optional[Tuple[int, int]] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize missing values in the dataset.

        Args:
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 6))

        # Missing values heatmap
        sns.heatmap(self.data.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=axes[0])
        axes[0].set_title("Missing Values Heatmap", fontsize=12, fontweight="bold")

        # Missing values bar chart
        missing_counts = self.data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)

        if len(missing_counts) > 0:
            missing_counts.plot(kind="barh", color="#e74c3c", ax=axes[1])
            axes[1].set_title("Missing Values Count", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Count")
        else:
            axes[1].text(0.5, 0.5, "No Missing Values!", ha="center", va="center", fontsize=14)
            axes[1].set_title("Missing Values Count", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_dashboard(
        self, target_column: Optional[str] = None, figsize: Tuple[int, int] = (20, 16), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.

        Args:
            target_column (str): Target column for analysis
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save the figure

        Returns:
            plt.Figure: The matplotlib figure
        """
        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Correlation heatmap (top-left, spans 2 rows)
        ax1 = fig.add_subplot(gs[0:2, 0])
        if len(self.numerical_columns) >= 2:
            corr = self.data[self.numerical_columns[:8]].corr()
            sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax1, cbar_kws={"shrink": 0.8})
            ax1.set_title("Correlation Matrix", fontweight="bold")

        # 2. Distribution of first numerical column (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if self.numerical_columns:
            sns.histplot(data=self.data, x=self.numerical_columns[0], kde=True, ax=ax2, color="#3498db")
            ax2.set_title(f"{self.numerical_columns[0]} Distribution", fontweight="bold")

        # 3. Distribution of second numerical column (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self.numerical_columns) >= 2:
            sns.histplot(data=self.data, x=self.numerical_columns[1], kde=True, ax=ax3, color="#2ecc71")
            ax3.set_title(f"{self.numerical_columns[1]} Distribution", fontweight="bold")

        # 4. Box plot (middle-middle)
        ax4 = fig.add_subplot(gs[1, 1])
        if self.numerical_columns:
            sns.boxplot(data=self.data[self.numerical_columns[:5]], ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            ax4.set_title("Box Plots", fontweight="bold")

        # 5. Categorical distribution (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        if self.categorical_columns:
            top_cats = self.data[self.categorical_columns[0]].value_counts().head(5)
            ax5.pie(top_cats.values, labels=top_cats.index, autopct="%1.1f%%")
            ax5.set_title(f"{self.categorical_columns[0]} Distribution", fontweight="bold")

        # 6. Scatter plot (bottom-left)
        ax6 = fig.add_subplot(gs[2, 0])
        if len(self.numerical_columns) >= 2:
            sns.scatterplot(data=self.data, x=self.numerical_columns[0], y=self.numerical_columns[1], alpha=0.6, ax=ax6)
            ax6.set_title(f"{self.numerical_columns[1]} vs {self.numerical_columns[0]}", fontweight="bold")

        # 7. Missing values (bottom-middle)
        ax7 = fig.add_subplot(gs[2, 1])
        missing = self.data.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing.plot(kind="bar", ax=ax7, color="#e74c3c")
            ax7.set_title("Missing Values", fontweight="bold")
            ax7.tick_params(axis="x", rotation=45)
        else:
            ax7.text(0.5, 0.5, "No Missing Values", ha="center", va="center")
            ax7.set_title("Missing Values", fontweight="bold")

        # 8. Data overview (bottom-right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis("off")
        info_text = f"""
        Dataset Overview
        ================
        Rows: {self.data.shape[0]:,}
        Columns: {self.data.shape[1]}
        
        Numerical: {len(self.numerical_columns)}
        Categorical: {len(self.categorical_columns)}
        
        Missing Cells: {self.data.isnull().sum().sum():,}
        Memory: {self.data.memory_usage(deep=True).sum()/1024**2:.2f} MB
        """
        ax8.text(
            0.1,
            0.5,
            info_text,
            transform=ax8.transAxes,
            fontsize=11,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle("Data Analysis Dashboard", fontsize=18, fontweight="bold", y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


if __name__ == "__main__":
    # Example usage
    print("Data Visualization Module")
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
            "gender": np.random.choice(["Male", "Female"], 200),
        }
    )

    # Add correlation
    sample_data["savings"] = sample_data["income"] * 0.2 + np.random.normal(0, 2000, 200)

    # Initialize visualizer
    viz = DataVisualizer(sample_data)

    # Create some plots
    print("Creating visualizations...")

    # Distribution grid
    fig1 = viz.plot_distributions_grid(columns=["age", "income", "spending_score", "credit_score"])
    plt.show()

    # Correlation heatmap
    fig2 = viz.plot_correlation_heatmap()
    plt.show()

    # Dashboard
    fig3 = viz.create_dashboard()
    plt.show()

    print("Visualizations created successfully!")
