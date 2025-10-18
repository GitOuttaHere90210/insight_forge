# Non prod app to test data_loader.py
from bi.data_loader import load_data
df = load_data(r"data\sales_data.csv")
print("Columns:", df.columns.tolist())
print(df.head().to_string(index=False))
