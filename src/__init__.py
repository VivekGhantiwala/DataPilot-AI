"""
Data Analysis AI Project

A comprehensive data analysis and machine learning framework with AI-powered insights.
"""

__version__ = "1.0.0"
__author__ = "Data Analysis AI Team"

from .ai_insights import AIInsights
from .automl import AutoML
from .data_generator import DataGenerator
from .data_preprocessing import DataPreprocessor
from .eda import ExploratoryAnalysis
from .explainability import ModelExplainer
from .ml_models import MLModels
from .report_generator import ReportGenerator
from .time_series import TimeSeriesAnalyzer
from .visualization import DataVisualizer

__all__ = [
    "DataPreprocessor",
    "ExploratoryAnalysis",
    "DataVisualizer",
    "MLModels",
    "AutoML",
    "TimeSeriesAnalyzer",
    "ModelExplainer",
    "AIInsights",
    "DataGenerator",
    "ReportGenerator",
]
