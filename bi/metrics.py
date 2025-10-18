# this is already install in the venv - pip install pandas scikit-learn seaborn matplotlib, added to requirements.txt

import pandas as pd
from sklearn.cluster import KMeans

def sales_by_month(df):
    """Calculate total sales by month"""
    # Your data uses 'Date' column
    date_col = 'Date' if 'Date' in df.columns else 'Month'
    
    if date_col not in df.columns or 'Sales' not in df.columns:
        return pd.DataFrame()
    
    df_copy = df.copy()
        
    # Parse dates with explicit format to avoid warning
    try:
        # Try ISO format (YYYY-MM-DD) which matches your data
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], format='%Y-%m-%d', errors='coerce')
        if df_copy[date_col].isna().all():
            # Fallback to flexible parsing
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    except Exception:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    df_copy = df_copy.dropna(subset=[date_col])
    
    if df_copy.empty:
        return pd.DataFrame()
    
    # Group by month
    result = df_copy.groupby(df_copy[date_col].dt.to_period('M'))['Sales'].sum()
    result.index = result.index.astype(str)
    
    return result.round(0)


def sales_by_region(df):
    """Calculate total sales by region"""
    if 'Region' not in df.columns or 'Sales' not in df.columns:
        return pd.DataFrame()
    return df.groupby('Region')['Sales'].sum()
    return result.round(0)


def satisfaction_by_region(df):
    """Calculate average satisfaction by region"""
    # Your data uses 'Customer_Satisfaction' column
    satisfaction_col = 'Customer_Satisfaction' if 'Customer_Satisfaction' in df.columns else 'Satisfaction'
    
    if satisfaction_col not in df.columns or 'Region' not in df.columns:
        return pd.DataFrame()
    
    return df.groupby('Region')[satisfaction_col].mean()
    #  return result.round(0)  

def sales_by_gender(df):
    """Calculate total sales by gender"""
    # Your data uses 'Customer_Gender' column
    gender_col = 'Customer_Gender' if 'Customer_Gender' in df.columns else 'Gender'
    
    if gender_col not in df.columns or 'Sales' not in df.columns:
        return pd.DataFrame()
    
    return df.groupby(gender_col)['Sales'].sum()


def sales_trend_over_time(df):
    """Calculate sales trends over time with proper date handling"""
    # Your data uses 'Date' column
    date_col = 'Date' if 'Date' in df.columns else 'Month'
    
    if date_col not in df.columns or 'Sales' not in df.columns:
        return pd.DataFrame()
    
    df_copy = df.copy()
    
    try:
        # Try ISO format (YYYY-MM-DD) which matches your data
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], format='%Y-%m-%d', errors='coerce')
        if df_copy[date_col].isna().all():
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    except Exception:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    df_copy = df_copy.dropna(subset=[date_col])
    
    if df_copy.empty:
        return pd.DataFrame()
    
    df_copy = df_copy.sort_values(date_col)
    result = df_copy.groupby(df_copy[date_col].dt.to_period('M'))['Sales'].sum()
    result.index = result.index.astype(str)
    
    return result

def product_performance_comparison(df):
    """
    Compute product performance comparisons based on sales and customer satisfaction.
    Returns a DataFrame with products as index and aggregated metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing sales data with 'Product' and 'Sales' columns.
                          Optional: 'Customer_Satisfaction' for additional insights.
    
    Returns:
        pd.DataFrame: Aggregated metrics by product (e.g., total sales, average satisfaction).
    """
    if 'Product' not in df.columns or 'Sales' not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Group by product and calculate total sales and average satisfaction
    product_metrics = df.groupby('Product').agg({
        'Sales': 'sum',
        'Customer_Satisfaction': 'mean' if 'Customer_Satisfaction' in df.columns else 'count'
    }).rename(columns={'Customer_Satisfaction': 'Avg_Satisfaction' if 'Customer_Satisfaction' in df.columns else 'Count'})
    
    return product_metrics if not product_metrics.empty else pd.DataFrame()

def customer_demographics_summary(df):
    """
    Compute summary statistics for customer demographics, grouped by gender and region.
    Returns a DataFrame with aggregated metrics (mean, median, std) for sales, age, and satisfaction.
    """
    if 'Customer_Gender' not in df.columns or 'Region' not in df.columns or df.empty:
        return pd.DataFrame()
    
    return df.groupby(['Customer_Gender', 'Region']).agg({
        'Sales': ['mean', 'median', 'std'],
        'Customer_Age': ['mean', 'median', 'std'],
        'Customer_Satisfaction': ['mean', 'median', 'std']
    }).reset_index()

    return result.round(0) 


def customer_age_group_summary(df):
    """
    Summarize metrics by age groups and add AgeGroup column to the DataFrame.
    Returns a DataFrame with mean sales and satisfaction per age bin, and modifies df in place.
    """
    if 'Customer_Age' not in df.columns or df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['AgeGroup'] = pd.cut(df_copy['Customer_Age'], bins=[18, 30, 40, 50, 60, 70], labels=['18-29', '30-39', '40-49', '50-59', '60-69'])
    
    result = df_copy.groupby('AgeGroup', observed=True).agg({
        'Sales': 'mean',
        'Customer_Satisfaction': 'mean'
    }).reset_index()
    
    # Update the original df with AgeGroup (in place modification)
    df['AgeGroup'] = df_copy['AgeGroup']
    
    return result.round(0)

# Customer demographics and segmentation
def customer_segmentation_clusters(df, n_clusters=4):
    """
    Perform K-Means clustering for customer segmentation based on age, sales, and satisfaction.
    Returns a tuple: (clustered DataFrame, DataFrame of cluster summaries, dict of overall averages).
    Modifies df in-place with 'Cluster' column.
    """
    # Check for required columns and non-empty DataFrame
    required_cols = {'Customer_Age', 'Sales', 'Customer_Satisfaction'}
    if not required_cols.issubset(df.columns) or df.empty:
        return df.copy(), pd.DataFrame(), {}
    
    # Prepare data for clustering
    X = df[['Customer_Age', 'Sales', 'Customer_Satisfaction']].dropna()
    if X.empty:
        return df.copy(), pd.DataFrame(), {}
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X)  # Add Cluster only to rows with data
    
    # Generate cluster summaries
    result = df.groupby('Cluster', observed=True).agg({
        'Customer_Age': 'mean',
        'Sales': 'mean',
        'Customer_Satisfaction': 'mean',
        'Customer_Gender': 'count'
    }).reset_index().rename(columns={'Customer_Gender': 'Count'})
    
    # Calculate overall averages
    overall = {
        'Customer_Age': int(df['Customer_Age'].mean()) if not df['Customer_Age'].isna().all() else 0,
        'Sales': int(df['Sales'].mean()) if not df['Sales'].isna().all() else 0,
        'Customer_Satisfaction': int(df['Customer_Satisfaction'].mean()) if not df['Customer_Satisfaction'].isna().all() else 0,
        'Count': len(df)
    }
    
    return df, result, overall