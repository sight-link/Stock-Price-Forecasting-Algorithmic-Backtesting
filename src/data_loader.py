import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import MinMaxScaler
from src.config import START_DATE, END_DATE, TRAIN_SPLIT_RATIO, MA10_WINDOW, MA50_WINDOW, RETURN_THRESHOLD

def load_raw_stock_data():
    """Download SPY data via AkShare"""
    print("[1/6] 正在从 AkShare 下载标普500指数ETF (SPY) 历史数据...")
    try:
        df_raw = ak.stock_us_daily(symbol="SPY", adjust="qfq")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw.set_index("date", inplace=True)
        df = df_raw.loc[START_DATE:END_DATE][["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
    except Exception as e:
        raise ValueError(f"数据下载失败，错误信息: {e}")
    return df

def build_technical_features(df):
    """Create MA indicators & train/test labels"""
    print("[2/6] 正在进行时序数据预处理与特征工程...")
    df.ffill(inplace=True)
    df["MA10"] = df["Close"].rolling(window=MA10_WINDOW).mean()
    df["MA50"] = df["Close"].rolling(window=MA50_WINDOW)
    
    # Regression target: next day close price
    df["Target_Price"] = df["Close"].shift(-1)
    # Classification target: bull signal
    df["Daily_Return"] = (df["Target_Price"] - df["Close"]) / df["Close"]
    df["Target_Class"] = np.where(df["Daily_Return"] > RETURN_THRESHOLD, 1, 0)
    df.dropna(inplace=True)
    return df

def split_and_scale_dataset(df):
    """Chronological split + MinMax normalization"""
    date_index = df.index
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50"]
    X = df[feature_cols].values
    y_price = df["Target_Price"].values.reshape(-1, 1)
    y_class = df["Target_Class"].values

    split_idx = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_price_raw, y_test_price_raw = y_price[:split_idx], y_price[split_idx:]
    y_train_class, y_test_class = y_class[:split_idx], y_class[split_idx:]
    test_dates = date_index[split_idx:]
    actual_today = df["Close"].values[split_idx:]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)
    y_train_price = scaler_y.fit_transform(y_train_price_raw).ravel()

    return (
        X_train, X_test,
        y_train_price, y_test_price_raw,
        y_train_class, y_test_class,
        test_dates, actual_today, scaler_y
    )

def full_data_pipeline():
    df_raw = load_raw_stock_data()
    df_feature = build_technical_features(df_raw)
    return split_and_scale_dataset(df_feature)
