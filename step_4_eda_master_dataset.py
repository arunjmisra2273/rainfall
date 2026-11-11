import pandas as pd
import os

def eda_master_dataset():
    data_path = './data/master_daily_rainfall_1901_2025.csv'
    
    print("1. Loading Master Dataset...")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    print(f"\n--- Dataset Overview ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    print("\n--- Missing Values (NaN) Check ---")
    # Count NaNs per subdivision to see exactly which ones are empty
    nan_counts = df.groupby('Subdivision')['Rainfall_mm'].apply(lambda x: x.isna().sum())
    nan_subs = nan_counts[nan_counts > 0]
    
    if not nan_subs.empty:
        print("Subdivisions with Missing Data (Islands outside the mainland grid):")
        print(nan_subs)
    else:
        print("No missing values found.")
        
    print("\n2. Filtering out empty island subdivisions for statistics...")
    df_valid = df.dropna(subset=['Rainfall_mm'])
    print(f"Remaining valid mainland rows: {len(df_valid):,}")
    
    print("\n--- Global Summary Statistics (Daily Rainfall in mm) ---")
    # Standard descriptive statistics
    print(df_valid['Rainfall_mm'].describe().round(2))
    
    print("\n--- Top 5 Wettest Days on Record (1901-2025) ---")
    # Finding the most extreme historical rainfall events across any subdivision
    top_5 = df_valid.nlargest(5, 'Rainfall_mm')
    print(top_5[['Date', 'Subdivision', 'Rainfall_mm']].to_string(index=False))
    
    print("\n--- Climatology Check: Average Daily Rainfall ---")
    avg_rain = df_valid.groupby('Subdivision')['Rainfall_mm'].mean().sort_values(ascending=False)
    
    print("\nTop 5 Wettest Subdivisions (Overall Average):")
    print(avg_rain.head().round(2))
    
    print("\nTop 5 Driest Subdivisions (Overall Average):")
    print(avg_rain.tail().round(2))

if __name__ == "__main__":
    eda_master_dataset()

# python step_4_eda_master_dataset.py