import pandas as pd
import numpy as np
from scipy.stats import linregress



def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_macd(series, short=12, long=26, signal=9):
    ema_short = calculate_ema(series, short)
    ema_long = calculate_ema(series, long)
    macd_line = ema_short - ema_long
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def calculate_bollinger(series, window=20, num_std=2):
    sma = calculate_sma(series, window)
    rolling_std = series.rolling(window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return sma, upper_band, lower_band


def analyze_stock(stock_id, data, Stock_name, show_plot=False):
    closing_prices =  pd.Series(data['close'])


    returns = np.diff(closing_prices) / closing_prices[:-1]


    mean_return = np.mean(returns)
    volatility = np.std(returns)
      # EMA
    EMA10 = calculate_ema(closing_prices, 10)
    MACD, Signal, MACDH = calculate_macd(closing_prices)

    # RSI
    RSI = calculate_rsi(closing_prices)

    # ATR
    ATR = calculate_atr(data)

    # Bollinger
    Bollinger_Mid, Bollinger_Upper, Bollinger_Lower = calculate_bollinger(closing_prices)


    slope, intercept, r_value, p_value, std_err = linregress(range(len(closing_prices)), closing_prices)
    trend = 'upward' if slope > 0 else 'downward'


    predicted_value = slope * len(closing_prices) + intercept



    result = {
        "Stock Name": Stock_name[int(stock_id)],
        "Current Price": round(closing_prices.iloc[-1], 2),
        "Volatility": round(volatility, 4),
        "10-Days EMA": round(EMA10.iloc[-1],2),
        "MACD": round(MACD.iloc[-1],2),
        "Signal": round(Signal.iloc[-1],2),
        "MACDH": MACDH.iloc[-1],
        "RSI": RSI.iloc[-1],
        "ATR": ATR.iloc[-1],
        "Bollinger_Mid": Bollinger_Mid.iloc[-1],
        "Bollinger_Upper": Bollinger_Upper.iloc[-1],
        "Bollinger_Lower": Bollinger_Lower.iloc[-1],
        "Predicted Value": round(predicted_value, 2),
        "Trend": trend,
    }

    return result

def analyze_all_stocks(stock_data, Stock_name, show_plot=False):
    results = []
    for stock_id, data in stock_data.items():
        if stock_id=="-1":
            continue
        else:
            result = analyze_stock(stock_id, data, Stock_name, show_plot=show_plot)
            results.append(result)
    return pd.DataFrame(results)
