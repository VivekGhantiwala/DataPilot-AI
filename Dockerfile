# Data Analysis AI Project - Docker Image
FROM python:3.11-slim

LABEL maintainer="Data Analysis AI Project"
LABEL description="Comprehensive data analysis and machine learning toolkit"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit plotly shap lime statsmodels

# Copy project files
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Default command - run Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
