import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate():
    data_dir = './data'
    print("1. Loading Sub-Seasonal Dataset...")
    df = pd.read_csv(os.path.join(data_dir, 'subseasonal_target_1901_2025.csv'), parse_dates=['Date'])
    
    # We will test this on 'KERALA' (Gateway of the monsoon)
    kerala_df = df[df['Subdivision'] == 'KERALA'].copy()
    
    print("2. Engineering Advanced Lags & Volatility...")
    # Long-term memory lags (teleconnections)
    kerala_df['Past_30D_Rain'] = kerala_df['Rainfall_mm'].rolling(30, min_periods=1).sum()
    kerala_df['Past_60D_Rain'] = kerala_df['Rainfall_mm'].rolling(60, min_periods=1).sum()
    
    # Volatility (Standard Deviation of daily rain over the last 14 days)
    kerala_df['Volatility_14D'] = kerala_df['Rainfall_mm'].rolling(14, min_periods=1).std().fillna(0)
    
    # Exact Autoregressive daily lags (what happened exactly 14 and 21 days ago)
    kerala_df['Lag_14'] = kerala_df['Rainfall_mm'].shift(14).fillna(0)
    kerala_df['Lag_21'] = kerala_df['Rainfall_mm'].shift(21).fillna(0)
    
    # Drop rows with missing values caused by the new rolling features
    kerala_df = kerala_df.dropna().copy()
    
    # Train on 1901-2015, Test on 2016-2025
    train = kerala_df[kerala_df['Year'] <= 2015].copy()
    test = kerala_df[kerala_df['Year'] > 2015].copy()
    
    print(f"Training samples: {len(train):,} days | Testing samples: {len(test):,} days")

    # ---------------------------------------------------------
    # MODEL 1: Climatology Baseline
    # ---------------------------------------------------------
    print("\n3. Calculating Climatology Baseline...")
    climatology = train.groupby('DayOfYear')['Target_Next_14D_Rain'].mean().to_dict()
    test['Baseline_Pred'] = test['DayOfYear'].map(climatology)
    
    # Fill any leap-year day 366 gaps in the test set with day 365's average
    test['Baseline_Pred'] = test['Baseline_Pred'].ffill()

    # ---------------------------------------------------------
    # MODEL 2: Gradient Boosting (Non-Linear ML)
    # ---------------------------------------------------------
    print("4. Training Gradient Boosting Model...")
    features = [
        'DayOfYear', 'Past_14D_Rain', 'Past_30D_Rain', 'Past_60D_Rain', 
        'Volatility_14D', 'Lag_14', 'Lag_21'
    ]
    
    X_train = train[features]
    y_train = train['Target_Next_14D_Rain']
    X_test = test[features]
    y_test = test['Target_Next_14D_Rain']

    # loss='absolute_error' explicitly tells the trees to optimize for MAE
    gb_model = HistGradientBoostingRegressor(
        loss='absolute_error', 
        max_iter=300,
        learning_rate=0.05,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    
    test['GB_Pred'] = gb_model.predict(X_test)
    test['GB_Pred'] = test['GB_Pred'].clip(lower=0) # Rain cannot be negative

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    print("\n5. Evaluation Results (Test Set 2016-2025):")
    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"--- {model_name} ---")
        print(f"MAE:  {mae:.2f} mm")
        print(f"RMSE: {rmse:.2f} mm")

    evaluate_model(test['Target_Next_14D_Rain'], test['Baseline_Pred'], "Climatology Baseline")
    evaluate_model(test['Target_Next_14D_Rain'], test['GB_Pred'], "Gradient Boosting Model")

if __name__ == "__main__":
    train_and_evaluate()

# python step_6_modeling.py