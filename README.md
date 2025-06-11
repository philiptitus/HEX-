# HEX Model Project

## Overview

HEX is an advanced, AI-powered analytics and assistant system for M-PESA transaction data. It enables users to upload their M-PESA statements (XLSX or PDF), automatically processes and categorizes transactions, applies machine learning clustering, and provides deep insights, visualizations, and interactive AI-driven assistance. HEX is designed for both technical and non-technical users who want to understand, manage, and optimize their financial behavior using real transaction data.

## Key Features

- **Automated Data Ingestion:**
  - Upload M-PESA statements in XLSX or PDF format (with optional password support for PDFs).
  - Robust extraction and cleaning of transaction data, including handling missing values and standardizing columns.

- **Intelligent Categorization:**
  - Transactions are automatically categorized (e.g., Airtime, Power, Rent, Shopping, Restaurant, Transport, etc.) using rule-based logic and can be enhanced with AI (Gemini API) for more nuanced categorization.

- **Feature Engineering:**
  - Extracts time-based features (hour, day of week, month) and encodes them for analysis.
  - Computes net transaction amounts and prepares data for clustering.

- **Clustering & Pattern Discovery:**
  - Uses K-Means clustering to group transactions by behavioral patterns (optimal cluster count is determined automatically).
  - Summarizes each cluster by average, median, and total amounts, and highlights top categories, times, and days.
  - Calculates and displays the Silhouette Score to assess clustering quality.

- **Visual Analytics:**
  - Interactive bar charts for total spending by hour, day of week, and category.
  - Cluster summary tables for quick pattern recognition.

- **AI-Powered Insights (Gemini):**
  - Natural language summaries and explanations for each cluster.
  - Automated anomaly/outlier detection and explanations.
  - Conversational interface: Ask questions about your spending, clusters, or graphs and get AI-generated answers.

- **Budgeting Assistant:**
  - Personalized budget creation based on your transaction history, clusters, and financial goals, with actionable recommendations.

- **Modular & Extensible:**
  - Core logic is split into reusable modules (e.g., `mpesa.py`, `time_category_insights.py`).
  - Easily extendable for new features, categories, or data sources.

## Folder Structure

- `final.ipynb` - Jupyter notebook for step-by-step analysis, clustering, and AI-powered insights.
- `mpesa.py` - Core logic for data ingestion, cleaning, categorization, clustering, and Streamlit dashboard.
- `time_category_insights.py` - Module for time-based transaction insights.
- `requirements.txt` - Python dependencies for this project.

## Setup

1. Create and activate a virtual environment (optional but recommended):
   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

- Run the Jupyter notebook for interactive, step-by-step analysis:
  ```powershell
  jupyter notebook final.ipynb
  ```
- Or launch the Streamlit dashboard (if available in your setup):
  ```powershell
  streamlit run mpesa.py
  ```
- Use the scripts for data processing, clustering, and AI-powered insights.

## Model Name

The main model in this project is called **HEX**.

## Who Is This For?

- Individuals seeking to understand and optimize their M-PESA spending.
- Data scientists and students learning about clustering, feature engineering, and AI-powered analytics.
- Anyone interested in building intelligent financial dashboards and assistants.

## License

This Project Is Licensed under the MIT License