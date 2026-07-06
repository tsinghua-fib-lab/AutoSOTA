import pandas as pd
import sqlite3
import numpy as np
import re
#from decoupledmarket.behavior import normalize_positive_operations

def normalize_positive_operations(data):

    positives = [item['operation'] for item in data if item['operation'] > 0]
    total = sum(positives)


    if total == 0:
        return data


    normalized_data = []
    for item in data:
        if item['operation'] > 0 and total > 1:
            normalized_value = item['operation'] / total
            normalized_data.append({'stock_id': item['stock_id'], 'operation': round(normalized_value, 4)})
        else:
            normalized_data.append(item)
    return normalized_data


def convert_macd_output_to_standard(macd_results):
    output_lines = []
    total_position = 0.0

    for stock_id, action in macd_results.items():


        if action.startswith("HOLD"):
            continue


        if "BUY" in action:
            sign = 1
        elif "SELL" in action:
            sign = -1
        else:
            continue


        match = re.search(r"([0-9]+\.[0-9]+)%", action)
        if not match:
            continue

        pct = float(match.group(1)) / 100.0
        op_value = round(sign * pct, 4)


        total_position += abs(op_value)
        if total_position>=1:
            total_position = 1


        output_lines.append(f"Operation: {op_value}, Stock: {stock_id}")
    result = normalize_positive_operations(output_lines)


    if not result:
        return "HOLD"


    result.append(f"Total position ratio: {round(total_position, 4)}")

    return "\n".join(result)

# ----------------------

# ----------------------
def query_stocks_table_analysis(virtual_date , windows = 30):
    cmd = """
        SELECT stock_id, virtual_date, last_price, highest_price, lowest_price, begin_price
        FROM stock
        WHERE virtual_date BETWEEN {} AND {}""".format(virtual_date - windows , virtual_date - 1)
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute(cmd)
    df = cursor.fetchall()
    df = pd.DataFrame(df, columns=["stock_id", "virtual_date", "last_price", "highest_price", "lowest_price", "begin_price"])
    prices={}
    for stock_id, group in df.groupby("stock_id"):
        prices[str(stock_id)] = group.rename(
            columns={
                "virtual_date": "date",
                "begin_price": "open",
                "highest_price": "high",
                "lowest_price": "low",
                "last_price": "close",
            }
        )[["date", "open", "high", "low", "close"]].reset_index(drop=True)
    return prices


# ----------------------

# ----------------------
def compute_macd(df, fast=12, slow=26, signal=9):
    df["EMA_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df


def macd_trade(df_dict):
    MAX_POSITION = 0.5
    MULTIPLIER = 2.0
    NOISE_THRESHOLD = 0.02

    results = {}

    for stock_id, data in df_dict.items():
        data = compute_macd(data.copy())
        data = data.dropna()

        if len(data) < 30:
            continue

        prev = data.iloc[-2]
        curr = data.iloc[-1]


        macd_prev = prev["MACD"]
        signal_prev = prev["Signal"]
        hist_prev = prev["Hist"]


        macd_curr = curr["MACD"]
        signal_curr = curr["Signal"]
        hist_curr = curr["Hist"]
        price = curr["close"]

        # -------------------------

        # -------------------------
        trend_strength = abs(macd_curr - signal_curr) * MULTIPLIER
        trend_strength = min(MAX_POSITION, trend_strength)


        if abs(macd_curr - signal_curr) < NOISE_THRESHOLD:
            results[stock_id] = f"HOLD (noise filter) at {price} on {curr['date']}"
            continue

        # -------------------------

        # -------------------------


        if macd_prev < signal_prev and macd_curr > signal_curr:
            results[stock_id] = f"BUY {trend_strength*100:.1f}% () at {price}"
        elif macd_prev > signal_prev and macd_curr < signal_curr:
            results[stock_id] = f"SELL {trend_strength*100:.1f}% () at {price}"


        elif macd_curr > signal_curr:

            if hist_curr > hist_prev:
                results[stock_id] = f"BUY {trend_strength*100:.1f}% (bullish momentum) at {price}"
            else:
                results[stock_id] = f"HOLD (weak bullish momentum) at {price}"

        elif macd_curr < signal_curr:

            if hist_curr < hist_prev:
                results[stock_id] = f"SELL {trend_strength*100:.1f}% (bearish momentum) at {price}"
            else:
                results[stock_id] = f"HOLD (weak bearish momentum) at {price}"

        else:
            results[stock_id] = f"HOLD at {price}"

    return results


for i in range(10):
    df = query_stocks_table_analysis(virtual_date=11+i)
    trade_results = macd_trade(df)
    print(trade_results)

    result = convert_macd_output_to_standard(trade_results)
    print(result)
