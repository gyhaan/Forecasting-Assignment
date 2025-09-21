import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def handle_missing_values(df, target_col=None):
    """
    Handle missing values by forward-filling. If target_col is provided, fill it specifically first.
    """
    if target_col:
        df[target_col] = df[target_col].ffill()
    df = df.ffill()
    return df

def extract_temporal_features(df, datetime_col='datetime'):
    """
    Convert datetime column and extract temporal features: hour, day, month, year.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['day'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    df['year'] = df[datetime_col].dt.year
    return df

def normalize_features(df, features, scaler=None, fit=True):
    """
    Normalize specified features using MinMaxScaler. If fit=True, fit the scaler; else, transform only.
    """
    if fit:
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        return df, scaler
    else:
        df[features] = scaler.transform(df[features])
        return df, scaler

def create_sequences(X, y=None, time_steps=24):
    """
    Create sequences for time series data. If y is None, return only Xs (for test data).
    """
    Xs = []
    if y is not None:
        ys = []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    else:
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
        return np.array(Xs)