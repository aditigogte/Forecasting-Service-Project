import pandas as pd
import numpy as np

def prepare_data(file_path, state):
    # Load data
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by state and handle missing dates via resampling
    s_df = df[df['State'] == state].copy().sort_values('Date')
    s_df = s_df.set_index('Date').resample('W').sum().reset_index()
    
    # --- Mandatory Features ---
    # Lags
    for l in [1, 7, 30]:
        s_df[f'lag_{l}'] = s_df['Total'].shift(l)
    
    # Rolling stats
    s_df['rolling_mean'] = s_df['Total'].shift(1).rolling(window=4).mean()
    s_df['rolling_std'] = s_df['Total'].shift(1).rolling(window=4).std()
    
    # Date features
    s_df['month'] = s_df['Date'].dt.month
    s_df['day_of_week'] = s_df['Date'].dt.dayofweek
    s_df['is_holiday'] = s_df['month'].apply(lambda x: 1 if x in [12, 1] else 0)
    
    return s_df.dropna()