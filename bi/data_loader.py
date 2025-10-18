# Here are column headers from sales_data.csv:
#Date,Product,Region,Sales,Customer_Age,Customer_Gender,Customer_Satisfaction

# Here are column headers from sales_data.csv:
#Date,Product,Region,Sales,Customer_Age,Customer_Gender,Customer_Satisfaction

import pandas as pd

def load_data(filepath: str = "data/sales_data.csv") -> pd.DataFrame:
    """
    Loads and preprocesses the sales data from a CSV file.
    Adds a 'Month' column parsed from the 'Date' column.
    """
    df = pd.read_csv(filepath)

    # Standardize column names (Title case, underscores instead of spaces)
    df.columns = [col.strip().title().replace(" ", "_") for col in df.columns]

    # Convert Date to datetime and extract Month
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.strftime("%B")

    return df
