#!/usr/bin/env python
"""
Data Analysis AI CLI
Command-line interface for the data analysis and machine learning toolkit.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_analyze(args):
    """Run exploratory data analysis."""
    from src.ai_insights import AIInsights
    from src.eda import ExploratoryAnalysis

    print(f"📊 Loading data from {args.input}...")
    data = pd.read_csv(args.input)

    print(f"✅ Loaded {data.shape[0]} rows, {data.shape[1]} columns\n")

    # Run EDA
    eda = ExploratoryAnalysis(data)
    eda.analyze_all()

    # Generate AI insights
    if not args.no_insights:
        ai = AIInsights(data)
        report = ai.generate_automated_report(target_column=args.target)
        print("\n" + report)

    # Save report if output specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\n✅ Report saved to {args.output}")


def cmd_automl(args):
    """Run AutoML pipeline."""
    from src.automl import AutoML

    print(f"🚀 Starting AutoML Pipeline...")
    print(f"   Input: {args.input}")
    print(f"   Target: {args.target}\n")

    data = pd.read_csv(args.input)
    X = data.drop(columns=[args.target])
    y = data[args.target]

    automl = AutoML(task=args.task, max_models=args.max_models, cv_folds=args.cv_folds, verbose=True)

    automl.fit(X, y)

    print("\n📊 Leaderboard:")
    print(automl.get_leaderboard().to_string(index=False))

    # Save model if specified
    if args.output:
        automl.save(args.output)


def cmd_train(args):
    """Train ML models."""
    from src.ml_models import MLModels

    print(f"🤖 Training models on {args.input}...")
    data = pd.read_csv(args.input)

    ml = MLModels(data=data, target_column=args.target, task_type=args.task, test_size=args.test_size)

    comparison = ml.train_all_models()

    print("\n📊 Model Comparison:")
    print(comparison.to_string(index=False))

    # Save best model
    if args.output:
        ml.save_model(args.output)


def cmd_predict(args):
    """Make predictions with saved model."""
    from src.automl import AutoML

    print(f"🔮 Loading model from {args.model}...")
    automl = AutoML.load(args.model)

    print(f"📊 Loading data from {args.input}...")
    data = pd.read_csv(args.input)

    predictions = automl.predict(data)

    # Save predictions
    output = pd.DataFrame({"prediction": predictions})
    output.to_csv(args.output, index=False)

    print(f"✅ Predictions saved to {args.output}")
    print(f"   Total predictions: {len(predictions)}")


def cmd_timeseries(args):
    """Run time series analysis."""
    from src.time_series import TimeSeriesAnalyzer

    print(f"📈 Time Series Analysis on {args.input}...")
    data = pd.read_csv(args.input)

    ts = TimeSeriesAnalyzer(data=data, date_column=args.date_column, value_column=args.value_column)

    # Run analysis
    ts.analyze()

    # Generate forecast
    if args.forecast > 0:
        print(f"\n📊 Generating {args.forecast}-period forecast...")
        forecast = ts.forecast(periods=args.forecast)
        print(forecast.to_string(index=False))

        if args.output:
            forecast.to_csv(args.output, index=False)
            print(f"\n✅ Forecast saved to {args.output}")

    # Print report
    print("\n" + ts.generate_report())


def cmd_preprocess(args):
    """Preprocess data."""
    from src.data_preprocessing import DataPreprocessor

    print(f"🔧 Preprocessing {args.input}...")

    preprocessor = DataPreprocessor()
    preprocessor.load_data(args.input)

    # Run preprocessing pipeline
    clean_data = preprocessor.preprocess_pipeline(
        handle_missing=not args.no_missing,
        remove_dups=not args.no_duplicates,
        handle_outliers_flag=not args.no_outliers,
        scale=args.scale,
        encode=args.encode,
    )

    # Save cleaned data
    clean_data.to_csv(args.output, index=False)
    print(f"✅ Cleaned data saved to {args.output}")
    print(f"   Original shape: {preprocessor.data.shape}")
    print(f"   Final shape: {clean_data.shape}")


def cmd_report(args):
    """Generate comprehensive report."""
    from src.report_generator import ReportGenerator

    print(f"📝 Generating report for {args.input}...")
    data = pd.read_csv(args.input)

    report_gen = ReportGenerator(data, title=args.title)

    if args.format == "html":
        report_gen.generate_html_report(args.output)
    else:
        report = report_gen.generate_summary_report()
        with open(args.output, "w") as f:
            f.write(report)

    print(f"✅ Report saved to {args.output}")


def cmd_dashboard(args):
    """Launch interactive dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print("❌ Dashboard not found. Please ensure dashboard/app.py exists.")
        sys.exit(1)

    print("🚀 Launching Streamlit Dashboard...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop\n")

    cmd = ["streamlit", "run", str(dashboard_path)]
    if args.port:
        cmd.extend(["--server.port", str(args.port)])

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        prog="data-analysis-ai",
        description="🤖 Data Analysis & AI Toolkit - A comprehensive ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run exploratory data analysis
  python cli.py analyze -i data.csv -t target_column

  # Train models with AutoML
  python cli.py automl -i data.csv -t target --task classification

  # Generate time series forecast
  python cli.py timeseries -i sales.csv --date-column date --value-column sales -f 30

  # Launch interactive dashboard
  python cli.py dashboard

For more info on each command, use: python cli.py <command> --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run exploratory data analysis")
    analyze_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    analyze_parser.add_argument("-t", "--target", help="Target column name")
    analyze_parser.add_argument("-o", "--output", help="Output report file")
    analyze_parser.add_argument("--no-insights", action="store_true", help="Skip AI insights")

    # AutoML command
    automl_parser = subparsers.add_parser("automl", help="Run AutoML pipeline")
    automl_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    automl_parser.add_argument("-t", "--target", required=True, help="Target column")
    automl_parser.add_argument("-o", "--output", help="Output model file")
    automl_parser.add_argument(
        "--task", choices=["auto", "classification", "regression"], default="auto", help="Task type"
    )
    automl_parser.add_argument("--max-models", type=int, default=10, help="Max models to train")
    automl_parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    train_parser.add_argument("-t", "--target", required=True, help="Target column")
    train_parser.add_argument("-o", "--output", help="Output model file")
    train_parser.add_argument(
        "--task", choices=["auto", "classification", "regression"], default="auto", help="Task type"
    )
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    predict_parser.add_argument("-m", "--model", required=True, help="Model file")
    predict_parser.add_argument("-o", "--output", required=True, help="Output predictions file")

    # Time series command
    ts_parser = subparsers.add_parser("timeseries", help="Time series analysis")
    ts_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    ts_parser.add_argument("--date-column", required=True, help="Date column name")
    ts_parser.add_argument("--value-column", required=True, help="Value column name")
    ts_parser.add_argument("-f", "--forecast", type=int, default=0, help="Forecast periods")
    ts_parser.add_argument("-o", "--output", help="Output forecast file")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    preprocess_parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    preprocess_parser.add_argument("--no-missing", action="store_true", help="Skip missing value handling")
    preprocess_parser.add_argument("--no-duplicates", action="store_true", help="Skip duplicate removal")
    preprocess_parser.add_argument("--no-outliers", action="store_true", help="Skip outlier handling")
    preprocess_parser.add_argument("--scale", action="store_true", help="Scale features")
    preprocess_parser.add_argument("--encode", action="store_true", help="Encode categorical")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    report_parser.add_argument("-o", "--output", required=True, help="Output report file")
    report_parser.add_argument("--title", default="Data Analysis Report", help="Report title")
    report_parser.add_argument("--format", choices=["html", "text"], default="html", help="Report format")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch interactive dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8501, help="Port number")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    commands = {
        "analyze": cmd_analyze,
        "automl": cmd_automl,
        "train": cmd_train,
        "predict": cmd_predict,
        "timeseries": cmd_timeseries,
        "preprocess": cmd_preprocess,
        "report": cmd_report,
        "dashboard": cmd_dashboard,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
