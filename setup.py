"""
Setup script for Data Analysis AI Project
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="data-analysis-ai",
    version="1.0.0",
    author="Data Analysis AI Contributors",
    author_email="",
    description="A comprehensive data analysis and machine learning toolkit with AutoML, explainability, and time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VivekGhantiwala/data-analysis-ai-project",
    project_urls={
        "Bug Tracker": "https://github.com/VivekGhantiwala/data-analysis-ai-project/issues",
        "Documentation": "https://github.com/VivekGhantiwala/data-analysis-ai-project#readme",
        "Source Code": "https://github.com/VivekGhantiwala/data-analysis-ai-project",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "ml": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
        ],
        "explainability": [
            "shap>=0.42.0",
            "lime>=0.2.0",
        ],
        "timeseries": [
            "statsmodels>=0.14.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
        ],
        "all": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "shap>=0.42.0",
            "lime>=0.2.0",
            "statsmodels>=0.14.0",
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-analysis-ai=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "data-science",
        "machine-learning",
        "automl",
        "data-analysis",
        "exploratory-data-analysis",
        "data-visualization",
        "preprocessing",
        "classification",
        "regression",
        "time-series",
        "explainability",
        "shap",
        "scikit-learn",
    ],
)
