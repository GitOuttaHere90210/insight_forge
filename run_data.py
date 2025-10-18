# non prod function for verifying top data

import pandas as pd
from bi.data_loader import load_data
from bi.metrics import sales_by_month, sales_by_region

df = load_data()

# Calculate monthly sales totals
monthly_totals = sales_by_month(df)

# Sort monthly totals in descending order and get the top 5
top_5_monthly_totals = monthly_totals.sort_values(ascending=False).head(5)

print("Top 5 Monthly Sales Totals (descending order):")
print(top_5_monthly_totals)

print("\nRegional totals:")
print(sales_by_region(df))