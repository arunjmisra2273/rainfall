import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

def run_model_shootout():
    data_dir = './data'
    
    print("1. Loading and Engineering Features...")
    df = pd.read_csv(os.path.join(data_dir, 'subseasonal_target_1901_2025.csv'), parse_dates=['Date'])
    
    df['Past_30D_Rain'] = df.groupby('Subdivision')['Rainfall_mm'].transform(lambda x: x.rolling(30, min_periods=1).sum())
    df['Past_60D_Rain'] = df.groupby('Subdivision')['Rainfall_mm'].transform(lambda x: x.rolling(60, min_periods=1).sum())
    df['Volatility_14D'] = df.groupby('Subdivision')['Rainfall_mm'].transform(lambda x: x.rolling(14, min_periods=1).std().fillna(0))
    df['Lag_14'] = df.groupby('Subdivision')['Rainfall_mm'].shift(14).fillna(0)
    df['Lag_21'] = df.groupby('Subdivision')['Rainfall_mm'].shift(21).fillna(0)
    
    features = [
        'Subdivision', 'DayOfYear', 'Past_14D_Rain', 'Past_30D_Rain', 
        'Past_60D_Rain', 'Volatility_14D', 'Lag_14', 'Lag_21'
    ]
    
    df = df.dropna(subset=features + ['Target_Next_14D_Rain']).copy()
    
    # Crucial for XGBoost and LightGBM to recognize the geographic text natively
    df['Subdivision'] = df['Subdivision'].astype('category')
    
    train_df = df[df['Year'] <= 2015].copy()
    test_df = df[df['Year'] > 2015].copy()
    
    X_train, y_train = train_df[features], train_df['Target_Next_14D_Rain']
    X_test, y_test = test_df[features], test_df['Target_Next_14D_Rain']
    
    # Define the contenders
    models = {
        "Scikit-Learn (HistGradientBoosting)": HistGradientBoostingRegressor(
            loss='absolute_error', max_iter=300, learning_rate=0.05, 
            categorical_features=[0], random_state=42
        ),
        "Microsoft (LightGBM)": lgb.LGBMRegressor(
            objective='mean_absolute_error', n_estimators=300, learning_rate=0.05, 
            random_state=42, n_jobs=-1
        ),
        "DMLC (XGBoost)": xgb.XGBRegressor(
            objective='reg:absoluteerror', n_estimators=300, learning_rate=0.05, 
            enable_categorical=True, tree_method='hist', random_state=42, n_jobs=-1
        )
    }

    print(f"\n2. Starting Shootout (Train: {len(X_train):,} rows | Test: {len(X_test):,} rows)")
    print("-" * 65)
    print(f"{'Model Algorithm':<35} | {'MAE (mm)':<10} | {'Time (s)'}")
    print("-" * 65)

    best_mae = float('inf')
    best_model_name = ""

    for name, model in models.items():
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict & Clip negative rainfall
        predictions = model.predict(X_test)
        predictions = np.clip(predictions, a_min=0, a_max=None)
        
        # Evaluate
        mae = mean_absolute_error(y_test, predictions)
        elapsed_time = time.time() - start_time
        
        print(f"{name:<35} | {mae:<10.2f} | {elapsed_time:.1f}s")
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    print("-" * 65)
    print(f"\nWINNER: {best_model_name} with an MAE of {best_mae:.2f} mm")

if __name__ == "__main__":
    run_model_shootout()

# python step_7b_model_shootout.py