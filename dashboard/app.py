"""
Streamlit Interactive Dashboard
Modern, interactive web interface for data analysis and machine learning.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_insights import AIInsights
from src.automl import AutoML
from src.data_preprocessing import DataPreprocessor
from src.eda import ExploratoryAnalysis
from src.ml_models import MLModels
from src.visualization import DataVisualizer

# Page config
st.set_page_config(
    page_title="Data Analysis AI Dashboard", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "data" not in st.session_state:
        st.session_state.data = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "model" not in st.session_state:
        st.session_state.model = None


def render_sidebar():
    """Render sidebar with file upload and options."""
    with st.sidebar:
        st.markdown("## 📊 Data Analysis AI")
        st.markdown("---")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset", type=["csv", "xlsx", "parquet"], help="Upload a CSV, Excel, or Parquet file"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    st.session_state.data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".parquet"):
                    st.session_state.data = pd.read_parquet(uploaded_file)

                st.success(f"✅ Loaded {len(st.session_state.data):,} rows")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        st.markdown("---")

        # Sample data option
        if st.button("🎲 Load Sample Data"):
            st.session_state.data = generate_sample_data()
            st.success("Sample data loaded!")

        st.markdown("---")
        st.markdown("### Quick Stats")

        if st.session_state.data is not None:
            data = st.session_state.data
            st.metric("Rows", f"{len(data):,}")
            st.metric("Columns", f"{len(data.columns)}")
            st.metric("Missing %", f"{data.isnull().mean().mean()*100:.1f}%")


def generate_sample_data():
    """Generate sample dataset for demo."""
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, n),
            "income": np.random.normal(50000, 15000, n).clip(20000, 150000),
            "credit_score": np.random.normal(650, 100, n).clip(300, 850).astype(int),
            "years_employed": np.random.randint(0, 40, n),
            "debt_ratio": np.random.uniform(0, 1, n),
            "num_accounts": np.random.randint(1, 10, n),
            "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
            "employment_type": np.random.choice(["Full-time", "Part-time", "Self-employed"], n),
            "loan_approved": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        }
    )

    # Add some missing values
    data.loc[np.random.choice(n, 50), "income"] = np.nan
    data.loc[np.random.choice(n, 30), "credit_score"] = np.nan

    return data


def render_overview_tab():
    """Render data overview tab."""
    st.markdown("## 📋 Data Overview")

    if st.session_state.data is None:
        st.info("👆 Please upload a dataset using the sidebar to get started.")
        return

    data = st.session_state.data

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(data):,}")
    with col2:
        st.metric("Total Columns", len(data.columns))
    with col3:
        st.metric("Numeric Features", len(data.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Features", len(data.select_dtypes(include=["object"]).columns))

    st.markdown("---")

    # Data preview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 👀 Data Preview")
        st.dataframe(data.head(20), use_container_width=True)

    with col2:
        st.markdown("### 📊 Column Types")
        dtype_df = pd.DataFrame(
            {
                "Column": data.columns,
                "Type": data.dtypes.astype(str),
                "Non-Null": data.count().values,
                "Missing %": (data.isnull().mean() * 100).round(2).values,
            }
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # Missing values visualization
    st.markdown("### 🔍 Missing Values")
    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={"x": "Column", "y": "Missing Count"},
            title="Missing Values by Column",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found!")


def render_eda_tab():
    """Render EDA tab with visualizations."""
    st.markdown("## 📊 Exploratory Data Analysis")

    if st.session_state.data is None:
        st.info("👆 Please upload a dataset first.")
        return

    data = st.session_state.data
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

    # Distribution Analysis
    st.markdown("### 📈 Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if numeric_cols:
            selected_num = st.selectbox("Select numeric column", numeric_cols, key="dist_num")
            fig = px.histogram(data, x=selected_num, marginal="box", title=f"Distribution of {selected_num}")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if categorical_cols:
            selected_cat = st.selectbox("Select categorical column", categorical_cols, key="dist_cat")
            value_counts = data[selected_cat].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {selected_cat}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation Analysis
    if len(numeric_cols) >= 2:
        st.markdown("### 🔗 Correlation Analysis")

        corr_matrix = data[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter plot
    st.markdown("### 🎯 Relationship Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
    with col3:
        color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="scatter_color")

    if x_col and y_col:
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=None if color_col == "None" else color_col,
            title=f"{y_col} vs {x_col}",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_preprocessing_tab():
    """Render preprocessing tab."""
    st.markdown("## 🔧 Data Preprocessing")

    if st.session_state.data is None:
        st.info("👆 Please upload a dataset first.")
        return

    data = st.session_state.data.copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚙️ Preprocessing Options")

        handle_missing = st.checkbox("Handle missing values", value=True)
        missing_strategy = st.selectbox(
            "Missing value strategy", ["mean", "median", "mode", "drop"], disabled=not handle_missing
        )

        remove_duplicates = st.checkbox("Remove duplicates", value=True)
        handle_outliers = st.checkbox("Handle outliers", value=False)
        scale_features = st.checkbox("Scale features", value=False)
        encode_categorical = st.checkbox("Encode categorical", value=False)

    with col2:
        st.markdown("### 📊 Current Data Stats")
        st.metric("Rows", len(data))
        st.metric("Missing Values", data.isnull().sum().sum())
        st.metric("Duplicates", data.duplicated().sum())

    if st.button("🚀 Apply Preprocessing", type="primary"):
        with st.spinner("Processing..."):
            preprocessor = DataPreprocessor()
            preprocessor.load_dataframe(data)

            if handle_missing:
                preprocessor.handle_missing_values(
                    strategy="auto", numerical_strategy=missing_strategy if missing_strategy != "drop" else "mean"
                )

            if remove_duplicates:
                preprocessor.remove_duplicates()

            if handle_outliers:
                preprocessor.handle_outliers(method="clip")

            if encode_categorical:
                preprocessor.encode_categorical(method="label")

            if scale_features:
                preprocessor.scale_features(method="standard")

            st.session_state.data = preprocessor.get_clean_data()
            st.session_state.preprocessor = preprocessor

            st.success("✅ Preprocessing complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Rows", len(st.session_state.data))
            with col2:
                st.metric("Final Missing", st.session_state.data.isnull().sum().sum())


def render_ml_tab():
    """Render machine learning tab."""
    st.markdown("## 🤖 Machine Learning")

    if st.session_state.data is None:
        st.info("👆 Please upload a dataset first.")
        return

    data = st.session_state.data

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Model Configuration")

        target_col = st.selectbox("Target Column", data.columns.tolist())

        task_type = st.radio("Task Type", ["Auto", "Classification", "Regression"])

        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

        model_type = st.selectbox(
            "Model Selection",
            ["AutoML (All Models)", "Random Forest", "Gradient Boosting", "Logistic/Linear Regression"],
        )

    with col2:
        st.markdown("### 📊 Target Distribution")

        if data[target_col].dtype == "object" or data[target_col].nunique() <= 10:
            fig = px.histogram(data, x=target_col, title=f"Distribution of {target_col}")
        else:
            fig = px.histogram(data, x=target_col, nbins=30, title=f"Distribution of {target_col}")

        st.plotly_chart(fig, use_container_width=True)

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a minute."):
            try:
                # Prepare data
                X = data.drop(columns=[target_col])
                y = data[target_col]

                # Handle categorical columns
                X = pd.get_dummies(X, drop_first=True)

                # Run AutoML
                automl = AutoML(task=task_type.lower() if task_type != "Auto" else "auto", max_models=8, verbose=False)
                automl.fit(X, y)

                st.session_state.model = automl

                st.success(f"✅ Training complete! Best model: {automl.best_model_name}")

                # Display leaderboard
                st.markdown("### 🏆 Model Leaderboard")
                leaderboard = automl.get_leaderboard()
                st.dataframe(leaderboard, use_container_width=True, hide_index=True)

                # Feature importance
                st.markdown("### 📊 Feature Importance")
                try:
                    importance = automl.get_feature_importance(top_n=15)
                    fig = px.bar(
                        importance, x="Importance", y="Feature", orientation="h", title="Top 15 Important Features"
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Feature importance not available for this model.")

            except Exception as e:
                st.error(f"Error during training: {e}")


def render_insights_tab():
    """Render AI insights tab."""
    st.markdown("## 🧠 AI-Powered Insights")

    if st.session_state.data is None:
        st.info("👆 Please upload a dataset first.")
        return

    data = st.session_state.data

    if st.button("🔍 Generate Insights", type="primary"):
        with st.spinner("Analyzing your data..."):
            ai = AIInsights(data)

            # Data Quality Issues
            st.markdown("### ⚠️ Data Quality Issues")
            issues = ai.detect_data_quality_issues()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Missing Columns", len(issues["missing_values"]))
            with col2:
                st.metric("Outlier Columns", len(issues["outliers"]))
            with col3:
                st.metric("Duplicates", issues["duplicates"])

            # Patterns
            st.markdown("### 🔍 Detected Patterns")
            patterns = ai.detect_patterns()

            if patterns["strong_correlations"]:
                st.markdown("#### Strong Correlations")
                for p in patterns["strong_correlations"][:5]:
                    st.write(f"• **{p['feature1']}** ↔ **{p['feature2']}**: {p['correlation']:.3f}")

            if patterns["skewed_features"]:
                st.markdown("#### Skewed Features")
                for f in patterns["skewed_features"][:5]:
                    st.write(f"• **{f['feature']}**: Skewness = {f['skewness']:.2f}")

            # Recommendations
            st.markdown("### 💡 Recommendations")
            recommendations = ai.generate_recommendations()

            for rec in recommendations[:5]:
                with st.expander(f"{rec['priority'].upper()}: {rec['category']}"):
                    st.write(f"**Issue:** {rec['issue']}")
                    st.write(f"**Recommendation:** {rec['recommendation']}")
                    st.write(f"**Action:** {rec['action']}")


def render_export_tab():
    """Render export tab."""
    st.markdown("## 📥 Export & Download")

    if st.session_state.data is None:
        st.info("👆 Please upload a dataset first.")
        return

    data = st.session_state.data

    st.markdown("### 📊 Export Processed Data")

    col1, col2 = st.columns(2)

    with col1:
        # CSV Download
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 Download as CSV", data=csv, file_name="processed_data.csv", mime="text/csv")

    with col2:
        # JSON Download
        json_data = data.to_json(orient="records")
        st.download_button(
            label="📥 Download as JSON", data=json_data, file_name="processed_data.json", mime="application/json"
        )

    if st.session_state.model is not None:
        st.markdown("### 🤖 Export Model")
        st.info("Model export functionality - save your trained model for later use.")

        if st.button("💾 Export Model"):
            # Save model logic would go here
            st.success("Model ready for export!")


def main():
    """Main application."""
    init_session_state()
    render_sidebar()

    # Main content area with tabs
    st.markdown('<h1 class="main-header">🤖 Data Analysis AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Comprehensive data analysis and machine learning toolkit*")
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📋 Overview", "📊 EDA", "🔧 Preprocessing", "🤖 ML Models", "🧠 AI Insights", "📥 Export"]
    )

    with tab1:
        render_overview_tab()

    with tab2:
        render_eda_tab()

    with tab3:
        render_preprocessing_tab()

    with tab4:
        render_ml_tab()

    with tab5:
        render_insights_tab()

    with tab6:
        render_export_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        "<center>Built with ❤️ using Streamlit | " "<a href='https://github.com'>GitHub</a></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
