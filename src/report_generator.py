"""
Automated Report Generator Module
Creates comprehensive HTML reports from data analysis and ML results.
"""

import base64
import io
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

warnings.filterwarnings("ignore")


class ReportGenerator:
    """
    Generate comprehensive HTML reports for data analysis and ML results.
    """

    def __init__(
        self, title: str = "Data Analysis Report", author: str = "DataScienceToolkit", include_plots: bool = True
    ):
        """
        Initialize the Report Generator.

        Args:
            title (str): Report title
            author (str): Report author
            include_plots (bool): Whether to include visualizations
        """
        self.title = title
        self.author = author
        self.include_plots = include_plots and PLOTTING_AVAILABLE
        self.sections = []

    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        if not PLOTTING_AVAILABLE:
            return ""

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        return image_base64

    def add_section(self, title: str, content: str, level: int = 2, plot_fig=None):
        """
        Add a section to the report.

        Args:
            title (str): Section title
            content (str): Section content (HTML or text)
            level (int): Heading level (1-6)
            plot_fig: Optional matplotlib figure to include
        """
        section = {"title": title, "content": content, "level": level, "plot": plot_fig}
        self.sections.append(section)

    def add_dataframe(self, title: str, df: pd.DataFrame, max_rows: int = 20):
        """
        Add a DataFrame table to the report.

        Args:
            title (str): Section title
            df (pd.DataFrame): DataFrame to display
        """
        # Truncate if too large
        display_df = df.head(max_rows)
        html_table = display_df.to_html(
            classes="dataframe", table_id=title.lower().replace(" ", "-"), escape=False, index=False
        )

        if len(df) > max_rows:
            html_table += f"\n<p><em>Showing {max_rows} of {len(df)} rows</em></p>"

        self.add_section(title, html_table, level=3)

    def add_summary_stats(self, title: str, stats_dict: Dict[str, Any]):
        """
        Add summary statistics to the report.

        Args:
            title (str): Section title
            stats_dict (dict): Dictionary of statistics
        """
        html = "<div class='stats-container'>"
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    value = f"{value:.4f}"
                html += f"<div class='stat-item'><strong>{key}:</strong> {value}</div>"
            else:
                html += f"<div class='stat-item'><strong>{key}:</strong> {value}</div>"
        html += "</div>"

        self.add_section(title, html, level=3)

    def generate_html(self, save_path: Optional[str] = None) -> str:
        """
        Generate the complete HTML report.

        Args:
            save_path (str): Optional path to save the HTML file

        Returns:
            str: Complete HTML content
        """
        html_parts = []

        # HTML Header
        html_parts.append(
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>"""
            + self.title
            + """</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h4 {
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }
        .stat-item strong {
            color: #2c3e50;
            display: block;
            margin-bottom: 5px;
        }
        table.dataframe {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        table.dataframe th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        table.dataframe td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        table.dataframe tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        table.dataframe tr:hover {
            background-color: #e8f4f8;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metadata {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .metadata p {
            margin: 5px 0;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e74c3c;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            margin: 5px;
        }
        .badge-success {
            background: #2ecc71;
            color: white;
        }
        .badge-warning {
            background: #f39c12;
            color: white;
        }
        .badge-danger {
            background: #e74c3c;
            color: white;
        }
        .info-box {
            background: #d1ecf1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>"""
            + self.title
            + """</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
            <p><strong>Author:</strong> """
            + self.author
            + """</p>
        </div>
"""
        )

        # Add sections
        for section in self.sections:
            heading_tag = f"h{section['level']}"
            html_parts.append(f"<{heading_tag}>{section['title']}</{heading_tag}>")
            html_parts.append(section["content"])

            # Add plot if available
            if section["plot"] is not None and self.include_plots:
                try:
                    plot_base64 = self._plot_to_base64(section["plot"])
                    if plot_base64:
                        html_parts.append(
                            f'<div class="plot-container">'
                            f'<img src="data:image/png;base64,{plot_base64}" alt="{section["title"]}">'
                            f"</div>"
                        )
                except Exception as e:
                    html_parts.append(f"<p><em>Error generating plot: {str(e)}</em></p>")

        # HTML Footer
        html_parts.append(
            """
        <div class="footer">
            <p>Generated by DataScienceToolkit - A Comprehensive Data Analysis & AI Framework</p>
        </div>
    </div>
</body>
</html>
"""
        )

        html_content = "\n".join(html_parts)

        # Save to file if path provided
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"✅ Report saved to {save_path}")

        return html_content

    def generate_ml_report(
        self,
        ml_results: Dict[str, Any],
        model_comparison: Optional[pd.DataFrame] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive ML results report.

        Args:
            ml_results (dict): ML results dictionary
            model_comparison (pd.DataFrame): Model comparison DataFrame
            feature_importance (pd.DataFrame): Feature importance DataFrame
            save_path (str): Path to save the report

        Returns:
            str: HTML report content
        """
        # Reset sections
        self.sections = []

        # Add overview
        self.add_section(
            "📊 Model Performance Overview",
            f"<p><strong>Task Type:</strong> {ml_results.get('task_type', 'Unknown')}</p>"
            f"<p><strong>Best Model:</strong> {ml_results.get('best_model', 'Unknown')}</p>"
            f"<p><strong>Training Samples:</strong> {ml_results.get('n_train', 'Unknown'):,}</p>"
            f"<p><strong>Test Samples:</strong> {ml_results.get('n_test', 'Unknown'):,}</p>"
            f"<p><strong>Number of Features:</strong> {ml_results.get('n_features', 'Unknown')}</p>",
            level=2,
        )

        # Model comparison table
        if model_comparison is not None and len(model_comparison) > 0:
            self.add_dataframe("🏆 Model Comparison", model_comparison)

        # Feature importance
        if feature_importance is not None and len(feature_importance) > 0:
            self.add_dataframe("🎯 Top Feature Importance", feature_importance.head(15))

        # Metrics summary
        if "metrics" in ml_results:
            self.add_summary_stats("📈 Performance Metrics", ml_results["metrics"])

        return self.generate_html(save_path)


if __name__ == "__main__":
    # Example usage
    print("Report Generator Module")
    print("=" * 50)

    # Create sample report
    generator = ReportGenerator(title="Sample Data Analysis Report", author="DataScienceToolkit Demo")

    # Add sample sections
    generator.add_section("Dataset Overview", "<p>This is a comprehensive analysis of the dataset.</p>", level=2)

    # Sample DataFrame
    sample_df = pd.DataFrame(
        {
            "Model": ["Random Forest", "Gradient Boosting", "SVM"],
            "Accuracy": [0.94, 0.92, 0.89],
            "F1 Score": [0.93, 0.91, 0.88],
        }
    )
    generator.add_dataframe("Model Performance", sample_df)

    # Generate report
    html_content = generator.generate_html("outputs/sample_report.html")
    print("✅ Sample report generated!")
