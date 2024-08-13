import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator

def calculate_technical_indicators(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    
    # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นหรือไม่
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USDT']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    # Calculate RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    # Calculate Stochastic RSI
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch_RSI'] = stoch.stoch()
    # Calculate MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    # Calculate SMAs
    for period in [7, 25, 99]:
        sma = SMAIndicator(close=df['Close'], window=period)
        df[f'SMA_{period}'] = sma.sma_indicator()
    # Calculate SMA gaps
    def calculate_sma_gap(row, sma1, sma2):
        if row[f'SMA_{sma1}'] > row['SMA_99']:
            return row[f'SMA_{sma1}'] - row[f'SMA_{sma2}']
        else:
            return row[f'SMA_{sma2}'] - row[f'SMA_{sma1}']
    df['SMA_7_25_GAP'] = df.apply(lambda row: calculate_sma_gap(row, 7, 25), axis=1)
    df['SMA_7_99_GAP'] = df.apply(lambda row: calculate_sma_gap(row, 7, 99), axis=1)
    df['SMA_25_99_GAP'] = df.apply(lambda row: calculate_sma_gap(row, 25, 99), axis=1)
    return df