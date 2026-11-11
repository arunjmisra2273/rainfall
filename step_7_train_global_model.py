import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def train_and_test_global_model():
    data_dir = './data'
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    
    print("1. Loading and Engineering Features for ALL Subdivisions...")
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
    df['Subdivision'] = df['Subdivision'].astype('category')
    
    # --- RIGOROUS TESTING PHASE ---
    print("\n2. Splitting Data for Evaluation (Train: 1901-2015 | Test: 2016-2025)...")
    train_df = df[df['Year'] <= 2015].copy()
    test_df = df[df['Year'] > 2015].copy()
    
    # Global Climatology Baseline (Must group by both Subdivision AND DayOfYear now)
    climatology = train_df.groupby(['Subdivision', 'DayOfYear'], observed=False)['Target_Next_14D_Rain'].mean().reset_index()
    climatology = climatology.rename(columns={'Target_Next_14D_Rain': 'Baseline_Pred'})
    
    test_df = test_df.merge(climatology, on=['Subdivision', 'DayOfYear'], how='left')
    test_df['Baseline_Pred'] = test_df.groupby('Subdivision')['Baseline_Pred'].ffill().bfill()
    
    X_train, y_train = train_df[features], train_df['Target_Next_14D_Rain']
    X_test, y_test = test_df[features], test_df['Target_Next_14D_Rain']
    
    print("3. Training Global Evaluation Model...")
    eval_model = HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=300, learning_rate=0.05, 
        categorical_features=[0], random_state=42
    )
    eval_model.fit(X_train, y_train)
    
    test_df['GB_Pred'] = eval_model.predict(X_test).clip(min=0)
    
    # --- EVALUATION METRICS ---
    print("\n--- GLOBAL EVALUATION RESULTS (All 34 Subdivisions) ---")
    base_mae = mean_absolute_error(test_df['Target_Next_14D_Rain'], test_df['Baseline_Pred'])
    gb_mae = mean_absolute_error(test_df['Target_Next_14D_Rain'], test_df['GB_Pred'])
    print(f"Climatology Baseline MAE : {base_mae:.2f} mm")
    print(f"Global ML Model MAE      : {gb_mae:.2f} mm")
    
    print("\n--- LOCAL ACCURACY CHECK (Did geographic encoding work?) ---")
    check_subs = ['KERALA', 'WEST RAJASTHAN', 'PUNJAB', 'COASTAL KARNATAKA']
    for sub in check_subs:
        sub_test = test_df[test_df['Subdivision'] == sub]
        if not sub_test.empty:
            s_base = mean_absolute_error(sub_test['Target_Next_14D_Rain'], sub_test['Baseline_Pred'])
            s_gb = mean_absolute_error(sub_test['Target_Next_14D_Rain'], sub_test['GB_Pred'])
            print(f"{sub:<18} | Baseline MAE: {s_base:>6.2f} mm | ML MAE: {s_gb:>6.2f} mm")
            
    # --- PRODUCTION PHASE ---
    print("\n4. Retraining Master Model on FULL 125-Year Dataset for Production...")
    X_full, y_full = df[features], df['Target_Next_14D_Rain']
    
    prod_model = HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=300, learning_rate=0.05, 
        categorical_features=[0], random_state=42
    )
    prod_model.fit(X_full, y_full)
    
    model_path = os.path.join(model_dir, 'global_monsoon_model.joblib')
    joblib.dump(prod_model, model_path)
    print(f"Success! Production model saved to: {model_path}")

if __name__ == "__main__":
    train_and_test_global_model()

# python step_7_train_global_model.py