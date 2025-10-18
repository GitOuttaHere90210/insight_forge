# InsightForge BI Dashboard

## Overview
A business intelligence dashboard built with Streamlit, analyzing sales data with filters, AI-generated summaries, and memory for contextual responses.

## Features
- Loads data from sales_data.csv (columns: Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction)
- Displays filtered raw data and charts for sales by month, region, satisfaction by region, and sales by gender
- Includes interactive filters for Region and Product
- Provides an AI summary with context from previous filter selections
- Allows downloading filtered data as CSV
- Stores and displays filter history for context-aware analysis

## Setup
1. Clone the repository
2. Create a virtual environment: python -m venv .venv
3. Activate: .venv\Scripts\Activate.ps1
4. Install dependencies: pip install -r requirements.txt
5. Add OPENAI_API_KEY to .env file
6. Run: streamlit run app.py

## Files
- app.py: Streamlit dashboard
- bi/data_loader.py: Loads sales_data.csv
- bi/metrics.py: Calculates sales and satisfaction metrics
- run_data.py: Console-based metrics
- test_data.py: Tests data loading
- show_requirements.py: Displays dependencies