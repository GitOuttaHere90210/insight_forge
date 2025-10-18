# non prod to verify data

from bi.data_loader import load_data
df = load_data()
print(df.columns)
print(df.head())