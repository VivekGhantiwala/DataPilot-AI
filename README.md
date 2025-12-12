<div align="center">

# 🧠 DataPilot AI

### Your Intelligent Data Analysis Copilot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0.0-orange?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-blueviolet?style=for-the-badge)

**A comprehensive, production-ready toolkit for automated data analysis, machine learning, and AI-powered insights.**

[✨ Features](#-features) •
[🚀 Quick Start](#-quick-start) •
[📖 Documentation](#-documentation) •
[🤝 Contributing](#-contributing)

---

</div>

## 🌟 Overview

**DataPilot AI** is an end-to-end data science framework designed to accelerate your machine learning workflow. From exploratory data analysis to model explainability, DataPilot AI provides intelligent automation while keeping you in control.

Whether you're a data scientist looking to speed up your workflow, a business analyst seeking quick insights, or a developer integrating ML into your applications, DataPilot AI has you covered.

<div align="center">

### 🎯 **Transform raw data into actionable insights in minutes, not hours.**

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Exploratory Data Analysis
- Automated statistical summaries
- Distribution analysis & visualization
- Correlation heatmaps with insights
- Missing value & outlier detection
- Data quality profiling

</td>
<td width="50%">

### 🤖 AutoML Pipeline
- Automated model selection
- Hyperparameter optimization
- Support for 10+ ML algorithms
- Model comparison leaderboard
- One-click model export

</td>
</tr>
<tr>
<td width="50%">

### 📈 Time Series Analysis
- Trend & seasonality decomposition
- ARIMA, Prophet & ETS forecasting
- Anomaly detection
- Stationarity testing (ADF, KPSS)
- Interactive forecast visualization

</td>
<td width="50%">

### 🔮 Model Explainability
- SHAP value analysis
- LIME explanations
- Feature importance plots
- Partial dependence plots
- Individual prediction insights

</td>
</tr>
<tr>
<td width="50%">

### 🧹 Data Preprocessing
- Intelligent missing value imputation
- Outlier detection & treatment
- Feature encoding (One-Hot, Label)
- Feature scaling (Standard, MinMax)
- Automated feature engineering

</td>
<td width="50%">

### 🧠 AI-Powered Insights
- Automated pattern detection
- Data quality recommendations
- Feature engineering suggestions
- Smart preprocessing advice
- Comprehensive report generation

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
📦 DataPilot-AI
├── 📂 src/                          # Core library modules
│   ├── ai_insights.py               # AI-powered insights engine
│   ├── automl.py                    # Automated machine learning
│   ├── data_preprocessing.py        # Data cleaning & transformation
│   ├── eda.py                       # Exploratory data analysis
│   ├── explainability.py            # SHAP & LIME integrations
│   ├── ml_models.py                 # ML model training & evaluation
│   ├── time_series.py               # Time series analysis & forecasting
│   ├── visualization.py             # Data visualization utilities
│   ├── report_generator.py          # Automated report generation
│   └── data_generator.py            # Synthetic data generation
├── 📂 dashboard/                    # Streamlit web interface
│   └── app.py                       # Interactive dashboard
├── 📂 tests/                        # Test suite
├── 📂 data/                         # Sample datasets
├── 📜 cli.py                        # Command-line interface
├── 📜 pyproject.toml                # Project configuration
├── 📜 requirements.txt              # Dependencies
├── 📜 Dockerfile                    # Docker configuration
└── 📜 README.md                     # You are here!
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (3.11 recommended)
- **pip** or **conda** package manager
- **Git** (for cloning the repository)

### Installation

#### Option 1: Install via pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/VivekGhantiwala/DataPilot-AI.git
cd DataPilot-AI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

#### Option 2: Install Dependencies Only

```bash
# Clone and install dependencies
git clone https://github.com/VivekGhantiwala/DataPilot-AI.git
cd DataPilot-AI

pip install -r requirements.txt
```

#### Option 3: Docker

```bash
# Build the Docker image
docker build -t datapilot-ai .

# Run the container
docker run -p 8501:8501 datapilot-ai
```

### Verify Installation

```bash
# Check CLI is working
python cli.py --help

# Run tests
pytest tests/ -v
```

---

## 📖 Documentation

### 🎨 Interactive Dashboard

Launch the beautiful Streamlit dashboard for a no-code experience:

```bash
# Using CLI
python cli.py dashboard

# Or directly with Streamlit
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501` to access the dashboard.

---

### ⌨️ Command Line Interface

DataPilot AI provides a powerful CLI for automation and scripting:

```bash
# Run exploratory data analysis
python cli.py analyze -i your_data.csv -t target_column -o report.txt

# Train models with AutoML
python cli.py automl -i data.csv -t target --task classification --max-models 10

# Time series forecasting
python cli.py timeseries -i sales.csv --date-column date --value-column sales -f 30

# Data preprocessing
python cli.py preprocess -i raw_data.csv -o clean_data.csv --scale --encode

# Generate HTML report
python cli.py report -i data.csv -o report.html --title "My Analysis"

# Launch dashboard
python cli.py dashboard --port 8501
```

---

### 🐍 Python API

Use DataPilot AI directly in your Python code:

#### Exploratory Data Analysis

```python
import pandas as pd
from src import ExploratoryAnalysis

# Load your data
data = pd.read_csv("your_data.csv")

# Run EDA
eda = ExploratoryAnalysis(data)
eda.print_report()

# Get specific analyses
correlations = eda.correlation_analysis()
missing = eda.missing_value_analysis()
outliers = eda.detect_outliers_summary()
```

#### AutoML Pipeline

```python
from src import AutoML
import pandas as pd

# Load data
data = pd.read_csv("your_data.csv")
X = data.drop(columns=["target"])
y = data["target"]

# Initialize and train AutoML
automl = AutoML(
    task="classification",
    max_models=10,
    cv_folds=5
)
automl.fit(X, y)

# View results
print(automl.get_leaderboard())
print(automl.summary())

# Make predictions
predictions = automl.predict(X_new)

# Save best model
automl.save("best_model.pkl")
```

#### Time Series Forecasting

```python
from src import TimeSeriesAnalyzer
import pandas as pd

# Load time series data
data = pd.read_csv("sales_data.csv")

# Initialize analyzer
ts = TimeSeriesAnalyzer(
    data=data,
    date_column="date",
    value_column="sales"
)

# Run complete analysis
ts.analyze()

# Generate forecast
forecast = ts.forecast(periods=30, method="auto")

# Plot results
ts.plot_forecast(forecast)

# Get report
print(ts.generate_report())
```

#### Model Explainability

```python
from src import ModelExplainer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create explainer
explainer = ModelExplainer(
    model=model,
    X_train=X_train,
    feature_names=X_train.columns.tolist(),
    task="classification"
)

# Get explanations
explanation = explainer.explain_prediction(X_test.iloc[0])

# Plot SHAP summary
explainer.plot_shap_summary(X_test)

# Generate explainability report
report = explainer.generate_report(X_test)
```

#### Data Preprocessing

```python
from src import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load data
preprocessor.load_data("raw_data.csv")

# Run full preprocessing pipeline
clean_data = preprocessor.preprocess_pipeline(
    handle_missing=True,
    remove_dups=True,
    handle_outliers_flag=True,
    scale=True,
    encode=True
)

# Or step-by-step
preprocessor.handle_missing_values(strategy="auto")
preprocessor.remove_duplicates()
preprocessor.handle_outliers(method="clip")
preprocessor.scale_features(method="standard")
preprocessor.encode_categorical(method="onehot")
```

#### AI-Powered Insights

```python
from src import AIInsights
import pandas as pd

data = pd.read_csv("your_data.csv")

# Generate AI insights
ai = AIInsights(data)

# Get automated report
report = ai.generate_automated_report(target_column="target")
print(report)

# Detect data quality issues
issues = ai.detect_data_quality_issues()

# Get recommendations
recommendations = ai.generate_recommendations(task_type="classification")

# Quick insights
quick = ai.get_quick_insights()
```

---

## 🧪 Supported Algorithms

### Classification
| Algorithm | Library | Notes |
|-----------|---------|-------|
| Logistic Regression | scikit-learn | Linear baseline |
| Random Forest | scikit-learn | Ensemble method |
| Gradient Boosting | scikit-learn | Sequential boosting |
| XGBoost | xgboost | High performance |
| LightGBM | lightgbm | Fast training |
| SVM | scikit-learn | Kernel-based |
| K-Nearest Neighbors | scikit-learn | Instance-based |
| Decision Tree | scikit-learn | Interpretable |
| AdaBoost | scikit-learn | Adaptive boosting |
| Extra Trees | scikit-learn | Random splits |

### Regression
| Algorithm | Library | Notes |
|-----------|---------|-------|
| Linear Regression | scikit-learn | Linear baseline |
| Ridge/Lasso | scikit-learn | Regularized |
| Random Forest | scikit-learn | Ensemble method |
| Gradient Boosting | scikit-learn | Sequential boosting |
| XGBoost | xgboost | High performance |
| LightGBM | lightgbm | Fast training |
| SVR | scikit-learn | Kernel-based |
| ElasticNet | scikit-learn | L1+L2 regularization |

### Time Series
| Algorithm | Library | Notes |
|-----------|---------|-------|
| ARIMA | statsmodels | Classic approach |
| SARIMA | statsmodels | Seasonal ARIMA |
| Exponential Smoothing | statsmodels | ETS models |
| Prophet | prophet | Facebook's library |

---

## 📊 Sample Outputs

### EDA Report

```
╔════════════════════════════════════════════════════════════════╗
║                   📊 Exploratory Data Analysis Report          ║
╚════════════════════════════════════════════════════════════════╝

📌 Dataset Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Rows: 10,000
• Total Columns: 25
• Memory Usage: 2.4 MB
• Numerical Columns: 18
• Categorical Columns: 7

📈 Statistical Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
...
```

### AutoML Leaderboard

```
╔═══════════════════════════════════════════════════════════════════╗
║                        🏆 Model Leaderboard                        ║
╚═══════════════════════════════════════════════════════════════════╝

 Rank │ Model               │ Accuracy │ F1 Score │ ROC-AUC │ Time(s)
──────┼─────────────────────┼──────────┼──────────┼─────────┼─────────
  1   │ XGBoost             │  0.9421  │  0.9385  │  0.9712 │   2.3
  2   │ LightGBM            │  0.9398  │  0.9362  │  0.9689 │   1.1
  3   │ Random Forest       │  0.9356  │  0.9318  │  0.9645 │   4.7
  4   │ Gradient Boosting   │  0.9289  │  0.9251  │  0.9601 │   8.2
  5   │ Extra Trees         │  0.9234  │  0.9195  │  0.9567 │   3.9
```

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Key configuration options:

```env
# General
DEBUG=false
ENVIRONMENT=development

# AutoML
AUTOML_TIME_BUDGET=3600
AUTOML_MAX_MODELS=10

# Explainability
SHAP_SAMPLE_SIZE=100
LIME_NUM_FEATURES=10

# Dashboard
STREAMLIT_PORT=8501
```

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_automl.py -v
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t datapilot-ai .

# Run the container
docker run -p 8501:8501 datapilot-ai

# Run with volume mount for data
docker run -p 8501:8501 -v $(pwd)/data:/app/data datapilot-ai
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  datapilot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - DEBUG=false
```

---

## 🗺️ Roadmap

- [x] Core EDA functionality
- [x] AutoML pipeline
- [x] Time series analysis
- [x] Model explainability (SHAP/LIME)
- [x] Streamlit dashboard
- [x] CLI interface
- [x] Docker support
- [ ] Deep learning integration (TensorFlow/PyTorch)
- [ ] Natural language query interface
- [ ] Cloud deployment templates
- [ ] Real-time streaming analysis
- [ ] Feature store integration
- [ ] MLOps pipeline integration

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Start for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/DataPilot-AI.git
cd DataPilot-AI

# Create a feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install -e ".[dev]"

# Make your changes and run tests
pytest tests/ -v

# Commit with conventional commits
git commit -m "feat: add amazing feature"

# Push and create a PR
git push origin feature/amazing-feature
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **scikit-learn** - Machine learning algorithms
- **XGBoost & LightGBM** - Gradient boosting frameworks
- **SHAP & LIME** - Model explainability libraries
- **Streamlit** - Dashboard framework
- **Plotly & Seaborn** - Visualization libraries
- **statsmodels** - Statistical modeling

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/VivekGhantiwala/DataPilot-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VivekGhantiwala/DataPilot-AI/discussions)
- **Email**: vivekghantiwala14@gmail.com

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ by [Vivek Ghantiwala](https://github.com/VivekGhantiwala)

**[⬆ Back to Top](#-datapilot-ai)**

</div>
