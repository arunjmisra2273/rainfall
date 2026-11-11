import pandas as pd
import os

def engineer_features():
    data_dir = './data'
    input_path = os.path.join(data_dir, 'master_daily_rainfall_1901_2025.csv')
    
    print("1. Loading Master Dataset...")
    df = pd.read_csv(input_path, parse_dates=['Date'])
    
    print("2. Dropping empty island subdivisions...")
    df = df.dropna(subset=['Rainfall_mm']).copy()
    
    # Ensure strict chronological order per subdivision
    df = df.sort_values(['Subdivision', 'Date']).reset_index(drop=True)
    
    print("3. Extracting Temporal Features...")
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.day_of_year
    
    print("\n4. Engineering Seasonal Dataset (JJAS)...")
    # Filter for core monsoon months (June, July, August, September)
    jjas_df = df[df['Month'].isin([6, 7, 8, 9])]
    
    # Sum rainfall per year per subdivision
    seasonal_df = jjas_df.groupby(['Subdivision', 'Year'])['Rainfall_mm'].sum().reset_index()
    seasonal_df = seasonal_df.rename(columns={'Rainfall_mm': 'JJAS_Total_Rain_mm'})
    
    seasonal_path = os.path.join(data_dir, 'seasonal_target_1901_2025.csv')
    seasonal_df.to_csv(seasonal_path, index=False)
    print(f"Seasonal dataset saved to: {seasonal_path}")
    
    print("\n5. Engineering Sub-Seasonal Dataset (14-Day Lags & Targets)...")
    
    # Calculate Past 14-Day Cumulative Rainfall (The primary autoregressive predictor)
    df['Past_14D_Rain'] = df.groupby('Subdivision')['Rainfall_mm'].transform(
        lambda x: x.rolling(window=14, min_periods=1).sum()
    )
    
    # Calculate Target: The NEXT 14-Day Cumulative Rainfall 
    # (Shift the 14-day rolling sum backwards by 14 days)
    df['Target_Next_14D_Rain'] = df.groupby('Subdivision')['Past_14D_Rain'].shift(-14)
    
    # Drop the very last 14 days of 2025, as they won't have a future target to predict
    sub_seasonal_df = df.dropna(subset=['Target_Next_14D_Rain']).copy()
    
    # Select only the features needed for modeling
    sub_seasonal_df = sub_seasonal_df[
        ['Date', 'Subdivision', 'Year', 'Month', 'DayOfYear', 
         'Rainfall_mm', 'Past_14D_Rain', 'Target_Next_14D_Rain']
    ]
    
    sub_seasonal_path = os.path.join(data_dir, 'subseasonal_target_1901_2025.csv')
    sub_seasonal_df.to_csv(sub_seasonal_path, index=False)
    print(f"Sub-seasonal dataset saved to: {sub_seasonal_path}")
    
    print("\nPreview of Sub-Seasonal Features (Starting at Day 14):")
    # Display an excerpt where the 14-day rolling sum has fully populated
    sample_sub = sub_seasonal_df[sub_seasonal_df['Subdivision'] == 'KERALA'].head(20)
    if sample_sub.empty:
        sample_sub = sub_seasonal_df.head(20)
    print(sample_sub.iloc[13:18].to_string(index=False))

if __name__ == "__main__":
    engineer_features()

# python step_5_feature_engineering.py