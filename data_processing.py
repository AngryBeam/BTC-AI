import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from custom_ta import calculate_technical_indicators
from logger import get_logger

logger, _ = get_logger()
def split_data(data, start_index, batch_size=100, train_ratio=0.8):
    end_index = start_index + batch_size
    if end_index > len(data):
        end_index = len(data)
    
    batch = data.iloc[start_index:end_index]
    train_size = int(len(batch) * train_ratio)
    
    train = batch.iloc[:train_size]
    test = batch.iloc[train_size:]
    
    return train, test, end_index

def resample_data(df, timeframe):
    agg_dict = {
        'Unix': 'first',
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume BTC': 'sum',
        'Volume USDT': 'sum'
    }
    
    if timeframe == 'W-MON':
        resampled = df.resample(timeframe).agg(agg_dict)
    elif timeframe == 'D':
        resampled = df.resample(timeframe).agg(agg_dict)
    elif timeframe == '4H':
        resampled = df.resample(timeframe).agg(agg_dict)
    else:
        resampled = df  # For 1h, no resampling needed
    
    return resampled

def check_timeframe_data(timeframe, df):
    logger.info(f"Checking data for {timeframe} timeframe:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"NaN values:\n{df.isna().sum()}")
    logger.info(f"Data types:\n{df.dtypes}")
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    logger.info(f"Infinite values: {inf_count}")
    
    # Check for consistent feature count
    expected_features = 16  # Adjust this based on your expected number of features
    if df.shape[1] != expected_features:
        logger.warning(f"Unexpected number of features: {df.shape[1]}. Expected: {expected_features}")
    
    # Check for very large or very small values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if abs(min_val) > 1e10 or abs(max_val) > 1e10:
            logger.warning(f"Column {col} has extreme values: min={min_val}, max={max_val}")
    
    logger.info("Data check completed.")
    logger.info("----------------------------------------")

def prepare_data(csv_path):
    logger.info(f"Loading Data Set from: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=1).drop(['Symbol', 'tradecount'], axis=1)
    df_sorted = df.sort_values('Unix', ascending=True).reset_index(drop=True)
    df_sorted['Date'] = pd.to_datetime(df_sorted['Unix'], unit='ms')
    df_sorted.set_index('Date', inplace=True)

    for col in ['Unix', 'Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USDT']:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')

    return df_sorted

def create_timeframes(df_sorted):
    return [
        ('weekly', resample_data(df_sorted, 'W-MON')),
        ('daily', resample_data(df_sorted, 'D')),
        ('4h', resample_data(df_sorted, '4H')),
        ('1h', df_sorted),
    ]