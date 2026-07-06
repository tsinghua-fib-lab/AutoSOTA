from decoupledmarket.arena_content.run_arena_prompt import tech_model
import pandas as pd
import sqlite3
import re
from decoupledmarket.database_utils import (
    Database_operate,
    query_all_stocks,
    parse_accounts,
    parse_memory,
    parse_persons,
    parse_gossip,
    submit_order,
    stock_name_to_id,
    stock_name_to_price,
    round_two_decimal,
)
from decoupledmarket.load_json import load_tech_traders
from decoupledmarket.constant import Num_Person, Num_agent
from decoupledmarket.constant import Num_Person, STOCK_NAMES, expense_ratio, Save_Path
from decoupledmarket.content.our_run_gpt_prompt import integrate_hold_info, integrate_stock_info

# ==============================

# ==============================
class StrategyBase:

    def normalize_positive_operations(self, data):
        positives = [x["operation"] for x in data if x["operation"] > 0]
        total = sum(positives)
        if total == 0:
            return data

        normalized = []
        for item in data:
            if item["operation"] > 0 and total > 1:
                normalized.append({
                    "stock_id": item["stock_id"],
                    "operation": round(item["operation"] / total, 4)
                })
            else:
                normalized.append(item)
        return normalized

    def convert_to_standard(self, signals):
        """Docstring."""

        output = {"stocks": [], "total_position": None}

        for stock_id, action in signals.items():
            if isinstance(stock_id, int):
                stock_id = chr(ord('A') + stock_id)
            elif isinstance(stock_id, str) and stock_id.isdigit():
                stock_id = chr(ord('A') + int(stock_id))


            if "HOLD" in action.upper():
                continue

            sign = 1 if "BUY" in action.upper() else -1

            match = re.search(r"(\d+(?:\.\d+)?)%", action)
            if not match:
                continue

            pct = float(match.group(1)) / 100.0

            output["stocks"].append({
                "stock_id": stock_id,
                "operation": round(sign * pct, 4)
            })


        output["stocks"] = self.normalize_positive_operations(output["stocks"])


        if output["stocks"]:
            pos = sum(x["operation"] for x in output["stocks"] if x["operation"] > 0)
            pos = min(1.0, round(pos, 4))
            output["total_position"] = pos
        else:
            output["stocks"] = [{"stock_id": None, "operation": 0.0}]
            output["total_position"] = 0.0

        return output

    def trade_signal(self, df_dict):
        raise NotImplementedError("Subclasses must implement trade_signal()")

# ==============================
# Buy & Hold
# ==============================
class BuyAndHoldTrader(StrategyBase):
    def __init__(self, position_size_pct=0.2):
        """Docstring."""
        self.position_size_pct = position_size_pct
        self.has_bought = False

    def trade_signal(self, df_dict):
        """Docstring."""
        signals = {}

        if not self.has_bought:

            for stock_id in df_dict.keys():
                signals[stock_id] = f"BUY {self.position_size_pct * 100:.1f}%"
            self.has_bought = True

        else:

            for stock_id in df_dict.keys():
                signals[stock_id] = "HOLD"

        return signals

# ==============================

# ==============================
class SMATrader(StrategyBase):
    def __init__(self, window=20, position_size_pct=0.2):
        self.window = window
        self.position_size_pct = position_size_pct

    def compute_sma(self, df):
        df["SMA"] = df["close"].rolling(self.window).mean()
        return df

    def trade_signal(self, df_dict):
        results = {}
        for stock_id, df in df_dict.items():
            df = self.compute_sma(df).dropna()
            if len(df) < 2:
                continue
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            last_price = curr["close"]
            sma_value = curr["SMA"]
            if prev["close"] <= prev["SMA"] and last_price > sma_value:
                results[stock_id] = f"BUY {self.position_size_pct*100:.1f}%"
            elif prev["close"] >= prev["SMA"] and last_price < sma_value:
                results[stock_id] = f"SELL {self.position_size_pct*100:.1f}%"
            else:
                results[stock_id] = f"HOLD at {last_price}"
        return results

# ==============================

# ==============================
class ATRTrader(StrategyBase):
    def __init__(self, window=14, position_size_pct=0.2, k=1.5):
        self.window = window
        self.position_size_pct = position_size_pct
        self.k = k

    def compute_atr(self, df):
        df = df.copy()
        df["prev_close"] = df["close"].shift(1)
        df["TR"] = df.apply(lambda row: max(
            row["high"]-row["low"],
            abs(row["high"]-row["prev_close"]),
            abs(row["low"]-row["prev_close"])
        ), axis=1)
        df["ATR"] = df["TR"].rolling(self.window).mean()
        return df

    def trade_signal(self, df_dict):
        results = {}
        for stock_id, df in df_dict.items():
            df = self.compute_atr(df).dropna()
            if len(df) < 2:
                continue
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            last_price = curr["close"]
            atr = curr["ATR"]
            prev_close = prev["close"]
            upper = prev_close + self.k * atr
            lower = prev_close - self.k * atr
            if last_price > upper:
                results[stock_id] = f"BUY {self.position_size_pct*100:.1f}%"
            elif last_price < lower:
                results[stock_id] = f"SELL {self.position_size_pct*100:.1f}%"
            else:
                results[stock_id] = f"HOLD at {last_price}"
        return results

# ==============================

# ==============================
class MACDTrader(StrategyBase):
    def __init__(self, fast=12, slow=26, signal=9, position_size_pct=0.2, noise=0.02):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.position_size_pct = position_size_pct
        self.noise = noise

    def compute_macd(self, df):
        df = df.copy()
        df["EMA_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["EMA_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
        df["Signal"] = df["MACD"].ewm(span=self.signal, adjust=False).mean()
        df["Hist"] = df["MACD"] - df["Signal"]
        return df

    def trade_signal(self, df_dict):
        results = {}
        for stock_id, df in df_dict.items():
            df = self.compute_macd(df).dropna()
            if len(df) < 2:
                continue
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            diff = curr["MACD"] - curr["Signal"]
            prev_diff = prev["MACD"] - prev["Signal"]
            last_price = curr["close"]
            if abs(diff) < self.noise:
                results[stock_id] = f"HOLD at {last_price}"
            elif prev_diff <=0 < diff:
                results[stock_id] = f"BUY {self.position_size_pct*100:.1f}%"
            elif prev_diff >=0 > diff:
                results[stock_id] = f"SELL {self.position_size_pct*100:.1f}%"
            else:
                results[stock_id] = f"HOLD at {last_price}"
        return results

# ==============================

# ==============================
class RSITrader(StrategyBase):
    def __init__(self, period=14, overbought=70, oversold=30, position_size_pct=0.2):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.position_size_pct = position_size_pct

    def compute_rsi(self, df):
        df = df.copy()
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(self.period).mean()
        roll_down = down.rolling(self.period).mean()
        rs = roll_up / roll_down
        df["RSI"] = 100 - 100/(1+rs)
        return df

    def trade_signal(self, df_dict):
        results = {}
        for stock_id, df in df_dict.items():
            df = self.compute_rsi(df).dropna()
            if len(df) < 1:
                continue
            last = df.iloc[-1]
            last_price = last["close"]
            rsi = last["RSI"]
            if rsi < self.oversold:
                results[stock_id] = f"BUY {self.position_size_pct*100:.1f}%"
            elif rsi > self.overbought:
                results[stock_id] = f"SELL {self.position_size_pct*100:.1f}%"
            else:
                results[stock_id] = f"HOLD at {last_price}"
        return results

# ==============================

# ==============================
class BollingerTrader(StrategyBase):
    def __init__(self, window=20, n_std=2, position_size_pct=0.2):
        self.window = window
        self.n_std = n_std
        self.position_size_pct = position_size_pct

    def compute_bollinger(self, df):
        df = df.copy()
        df["MA"] = df["close"].rolling(self.window).mean()
        df["STD"] = df["close"].rolling(self.window).std()
        df["Upper"] = df["MA"] + self.n_std * df["STD"]
        df["Lower"] = df["MA"] - self.n_std * df["STD"]
        return df

    def trade_signal(self, df_dict):
        results = {}
        for stock_id, df in df_dict.items():
            df = self.compute_bollinger(df).dropna()
            if len(df) < 1:
                continue
            last = df.iloc[-1]
            last_price = last["close"]
            upper = last["Upper"]
            lower = last["Lower"]
            if last_price > upper:
                results[stock_id] = f"SELL {self.position_size_pct*100:.1f}%"
            elif last_price < lower:
                results[stock_id] = f"BUY {self.position_size_pct*100:.1f}%"
            else:
                results[stock_id] = f"HOLD at {last_price}"
        return results

class ZMRTrader(StrategyBase):
    def __init__(self, window=20, k=1.5, position_size_pct=0.2):
        self.window = window
        self.k = k
        self.position_size_pct = position_size_pct

    def trade_signal(self, df_dict):
        results = {}
        for stock_id, df in df_dict.items():
            df = df.copy()
            df["MA"] = df["close"].rolling(self.window).mean()
            df["STD"] = df["close"].rolling(self.window).std()
            df = df.dropna()
            if len(df) < 1:
                continue
            last = df.iloc[-1]
            last_price = last["close"]
            ma = last["MA"]
            std = last["STD"]
            upper = ma + self.k*std
            lower = ma - self.k*std
            if last_price > upper:
                results[stock_id] = f"SELL {self.position_size_pct*100:.1f}%"
            elif last_price < lower:
                results[stock_id] = f"BUY {self.position_size_pct*100:.1f}%"
            else:
                results[stock_id] = f"HOLD at {last_price}"
        return results

def query_prices(virtual_date):
    lookback = 20
    db_path = f'{Save_Path}/data.db'
    cmd = f"""
            SELECT stock_id,
                virtual_date,
                last_price   AS close,
                highest_price AS high,
                lowest_price  AS low
            FROM stock
            WHERE stock_id >= 0
            AND virtual_date BETWEEN {virtual_date - lookback} AND {virtual_date}
        """

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    prices = {}
    for stock_id, group in df.groupby("stock_id"):
        prices[str(stock_id)] = group.rename(columns={"virtual_date": "date"})[["date","close","high","low"]].reset_index(drop=True)
    return prices

def create_tech_model(model_name):
    if model_name == "BuyAndHoldTrader":
        return BuyAndHoldTrader()
    elif model_name == "SMATrader":
        return SMATrader()
    elif model_name == "ZMRTrader":
        return ZMRTrader()
    elif model_name == "MACDTrader":
        return MACDTrader()
    elif model_name == "ATRTrader":
        return ATRTrader()
    elif model_name == "RSITrader":
        return RSITrader()
    elif model_name == "BollingerTrader":
        return BollingerTrader()
    else:
        raise ValueError(f"Unknown tech model: {model_name}")


class Tech_traders:
    def __init__(self, person_id, broker, stocks, database: Database_operate, tech_trader_path, k, bid_step=1):
        self.db = database
        self.person_id = person_id
        self.stocks = stocks
        self.broker = broker
        self.today_index = 7  # index of date
        self.sum_quantity = 0
        self.income = None
        self.cash = None
        self.asset = None
        self.wealth = None
        self.principle = None
        self.identity = (
            None  # the dictionary that stores the basic identical information
        )
        self.minimum_living_expense = None
        self.daily_expense = self.minimum_living_expense

        self.reflect_frequency = None
        self.agent_model = None
        self.tech_trader = None

        self.initialize_agent(tech_trader_path, k)

    def initialize_agent(self, tech_trader_path, k):
        agents = load_tech_traders(tech_trader_path)
        for agent in agents:
            if self.person_id == agent["agent_id"] + Num_Person + Num_agent + 100 * k:
                self.identity = agent
                self.income = agent["daily_income_from_job"]
                self.principle = agent["principle"]
                self.cash = agent["cash"]
                self.minimum_living_expense = agent["minimum_living_expense"]
                self.reflect_frequency = agent["reflect_frequency"]
                self.agent_model = agent["tech_model"]
                self.tech_trader = create_tech_model(self.agent_model)

        self.asset = 0

        self.wealth = self.cash + self.asset  # no asset at the beginning

        cmd = "Insert Into person values({},{},{},{},{},{},{},{},'{}')".format(
            self.person_id,
            0,
            self.cash,
            0,
            self.wealth,
            self.income,
            0,
            self.minimum_living_expense,
            self.identity["principle"],
        )
        self.db.execute_sql(cmd)

    def extract(self, trading):
        if trading['operation']>0:
            order_type = "buy"
        elif trading['operation']<0:
            order_type = "sell"
        elif trading['operation']==0:
            order_type = "hold"
        if order_type == "hold":
            return order_type, None, None
        else:
            stock_name = trading['stock_id']
            percentage = trading['operation']
        return order_type, stock_name, percentage

    def create_order(self, op, virtual_date, total_position, iteration=0):
        order_type, stock_name, percentage = self.extract(op)
        stock_id = stock_name_to_id(self.stocks, stock_name)
        current_price = stock_name_to_price(self.stocks, stock_name)
        order_type = order_type.lower()

        # timestamp int, virtual_date text, weekday int, stock_id int, person_id int, type text, price float
        if current_price <= 0:
            return

        if order_type == "buy":

            investable_cash = max(0, ( self.cash - self.minimum_living_expense*10) * total_position)


            intended_expense = percentage * investable_cash
            bid_price = current_price * 1.02


            max_buy_qty = int(self.cash // bid_price)
            intended_qty = int(intended_expense // bid_price)


            buy_qty = min(intended_qty, max_buy_qty)


            if buy_qty > 0 and (buy_qty * bid_price) <= self.cash:
                submit_order(
                    self.db,
                    order_type,
                    self.person_id,
                    stock_id,
                    virtual_date,
                    iteration,
                    bid_price,
                    buy_qty,
                )

        elif order_type == "sell" and current_price > 0:
            bid_price = current_price * 0.98

            stock_onhold = self.query_single_stock(virtual_date, stock_id)
            if stock_onhold is None:
                return

            current_qty = stock_onhold["quantity"]
            intended_sell_qty = int(abs(current_qty * percentage))


            sell_qty = min(intended_sell_qty, current_qty)

            if sell_qty > 0:
                submit_order(
                    self.db,
                    order_type,
                    self.person_id,
                    stock_id,
                    virtual_date,
                    iteration,
                    bid_price,
                    sell_qty,
                )

    def settlement(self, order, price, quantity):

        type = order["type"]
        order_volume = price * quantity
        stock_onhold = self.query_single_stock(order["virtual_date"], order["stock_id"])
        if order["type"] == "buy" and self.cash > order_volume:
            self.cash -= order_volume
            self.asset += order_volume
            if stock_onhold is None:
                cmd = ("insert into account values({},{},{},{},{},{},{},{},{})").format(
                    self.person_id,
                    order["stock_id"],
                    order["virtual_date"],
                    order["virtual_date"] % 7,
                    quantity,
                    price,
                    price,
                    0,
                    order["virtual_date"],
                )
                self.db.execute_sql(cmd)
            else:
                new_quantity = stock_onhold["quantity"] + quantity
                new_price = (
                    stock_onhold["cost_price"] * stock_onhold["quantity"]
                    + price * quantity
                ) / new_quantity
                profit = (stock_onhold["current_price"] - new_price) / new_price
                cmd = (
                    "update account set cost_price={}, quantity={}, profit={} where stock_id={} and virtual_date={} and "
                    "person_id={}"
                ).format(
                    new_price,
                    new_quantity,
                    profit,
                    order["stock_id"],
                    order["virtual_date"],
                    self.person_id,
                )
                self.db.execute_sql(cmd)

        if order["type"] == "sell":
            if stock_onhold["quantity"] <= quantity:
                order_volume = price * stock_onhold["quantity"]
                self.cash += order_volume
                self.asset -= order_volume
                new_quantity=0
            else:
                self.cash += order_volume
                self.asset -= order_volume
                new_quantity = stock_onhold["quantity"] - quantity
            assert new_quantity >= 0
            profit_amount = (price - stock_onhold["cost_price"]) * quantity
            cmd = (
                "update account set  quantity={} where stock_id={} and virtual_date={} and "
                "person_id={}"
            ).format(
                new_quantity, order["stock_id"], order["virtual_date"], self.person_id,
            )
            self.db.execute_sql(cmd)
            # insert memory, to be filled

    def end_of_iteration(self, virtual_date, iteration):
        # update the personal status after a iteration of trading
        all_stocks = query_all_stocks(self.db, virtual_date)
        hold_stocks = self.query_hold_stocks(virtual_date)
        if hold_stocks is None:  # skip for loop
            hold_stocks = []

        # stock price and personal asset update
        total_asset = 0
        capital_gain = 0
        for each_hold_stock in hold_stocks:
            stock_id = each_hold_stock["stock_id"]
            each_hold_stock["current_price"] = all_stocks[stock_id]["last_price"]
            each_hold_stock["profit"] = (
                each_hold_stock["current_price"] - each_hold_stock["cost_price"]
            ) / each_hold_stock["cost_price"]
            total_asset += (
                each_hold_stock["current_price"] * each_hold_stock["quantity"]
            )
            capital_gain += (
                each_hold_stock["current_price"] - each_hold_stock["cost_price"]
            ) * each_hold_stock["quantity"]

            # update price
            cmd = (
                "update account set current_price={}, profit={} where stock_id={} and virtual_date={} and "
                "person_id={}"
            ).format(
                each_hold_stock["current_price"],
                each_hold_stock["profit"],
                stock_id,
                virtual_date,
                self.person_id,
            )
            self.db.execute_sql(cmd)

        # update asset
        self.asset = total_asset
        self.wealth = self.asset + self.cash

        cmd = (
            "update person set cash={}, asset={},wealth={}, capital_gain={} where person_id={} and virtual_date={}"
        ).format(
            self.cash,
            self.asset,
            self.wealth,
            capital_gain,
            self.person_id,
            virtual_date,
        )
        self.db.execute_sql(cmd)

    def end_of_day(self, virtual_date):
        # update the personal status after a day trading
        all_stocks = query_all_stocks(self.db, virtual_date)
        hold_stocks = self.query_hold_stocks(virtual_date)
        if hold_stocks is None:  # skip for loop
            hold_stocks = []

        # stock price and personal asset update
        total_asset = 0
        capital_gain = 0
        dividend = 0
        for each_hold_stock in hold_stocks:
            stock_id = each_hold_stock["stock_id"]
            each_hold_stock["current_price"] = all_stocks[stock_id]["last_price"]
            each_hold_stock["profit"] = (
                each_hold_stock["current_price"] - each_hold_stock["cost_price"]
            ) / each_hold_stock["cost_price"]
            total_asset += (
                each_hold_stock["current_price"] * each_hold_stock["quantity"]
            )
            capital_gain += (
                each_hold_stock["current_price"] - each_hold_stock["cost_price"]
            ) * each_hold_stock["quantity"]
            # calculate dividend
            dividend += each_hold_stock["quantity"] * self.stocks[stock_id].DPS

            # update price for the next day
            day_offset = 1
            new_date = virtual_date + day_offset
            cmd = ("insert into account values({},{},{},{},{},{},{},{},{})").format(
                self.person_id,
                stock_id,
                new_date,
                new_date % 7,
                each_hold_stock["quantity"],
                each_hold_stock["cost_price"],
                each_hold_stock["current_price"],
                each_hold_stock["profit"],
                each_hold_stock["start_date"],
            )
            self.db.execute_sql(cmd)

        # update asset
        self.cash += dividend
        self.asset = total_asset
        self.daily_expense = 0#(
            #total_asset * 0.7 + self.cash
        #) * expense_ratio + self.minimum_living_expense
        self.cash -= self.daily_expense
        self.wealth = self.asset + self.cash
        self.broker.count_expense(self.daily_expense)

        cmd = ("insert into person values({},{},{},{},{},{},{},{},'{}')").format(
            self.person_id,
            virtual_date + 1,
            self.cash,
            self.asset,
            self.wealth,
            self.income,
            capital_gain,
            self.daily_expense,
            self.principle,
        )
        self.db.execute_sql(cmd)

    def query_hold_stocks(self, virtual_date):
        cmd = "select * from account where virtual_date ={} and person_id ={} and quantity >0 ".format(
            virtual_date, self.person_id,
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_accounts(results)
        if len(results) >= 1:
            return results
        else:
            return None

    def query_single_stock(self, virtual_date, stock_id):
        cmd = "select * from account where virtual_date ={} and person_id ={} and stock_id ={}".format(
            virtual_date, self.person_id, stock_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_accounts(results)
        if len(results) >= 1:
            return results[0]
        else:
            return None

    def query_account(self, virtual_date):
        cmd = "select * from account where virtual_date ={} and person_id ={} and quantity >0 ".format(
            virtual_date, self.person_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_accounts(results)
        return_infos = "I am holding the following stock:"
        if len(results) < 1:
            return_infos = "I do not hold any stock right now"
        else:  # holding stock
            for each_hold_stock in results:
                volume = each_hold_stock["cost_price"] * each_hold_stock["quantity"]
                abs_profit = abs(each_hold_stock["profit"]) * volume
                balance = volume + abs_profit
                statement = "gain" if each_hold_stock["profit"] > 0 else "loss"
                stock_id = each_hold_stock["stock_id"]
                return_infos += (
                    "hold {quantity} shares of Stock {name}, bought at an average price of ${cost_price:.2f} per share, "
                    "for {duration} days with portfolio value ${balance:.2f} and {statement} in {profit:.2f}% "
                    "from this investment;"
                ).format(
                    name=STOCK_NAMES[stock_id],
                    quantity=each_hold_stock["quantity"],
                    cost_price=each_hold_stock["cost_price"],
                    duration=virtual_date - each_hold_stock["start_date"],
                    statement=statement,
                    balance=balance,
                    profit=each_hold_stock["profit"] * 100,
                )
        return return_infos

    def query_prompt(self, virtual_date):
        cmd = "select * from account where virtual_date ={} and person_id ={} and quantity >0 ".format(
            virtual_date, self.person_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_accounts(results)
        return_stocks = []
        for each_hold_stock in results:
            current_stock = self.stocks[each_hold_stock["stock_id"]]
            volume = each_hold_stock["cost_price"] * each_hold_stock["quantity"]
            captital_gain = each_hold_stock["profit"] * 100
            price_change = current_stock.query_intraday_percentage(virtual_date) * 100
            prices = current_stock.query_daily_return(virtual_date)
            dic = {
                "Stock_name": current_stock.stock_name,
                "Share_number": each_hold_stock["quantity"],
                "total_value": volume,
                "captital_gain": captital_gain,
                "Price_change": prices,
                "Current_price_change": price_change,
                "Current_price": each_hold_stock["current_price"],
                "Cost_price": each_hold_stock["cost_price"],
            }
            for key, value in dic.items():
                dic[key] = round_two_decimal(value)
            return_stocks.append(dic)
        return return_stocks

    def query_person(self, virtual_date):
        cmd = "select * from person where virtual_date ={} and person_id ={}".format(
            virtual_date, self.person_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_persons(results)
        return_person = []
        for p in results:
            date = p["virtual_date"]
            cash = p["cash"]
            strategy = p["principle"]
            wealth = p["wealth"]
            asset = p["asset"]
            capital_gain = p["capital_gain"]
            daily_expense = p["daily_expense"]
            dic = {
            "virtual_date": date,
            "cash": cash,
            "asset": asset,
            "wealth": wealth,
            "strategy": strategy,
            "daily_expense": daily_expense,
            "capital_gain": capital_gain,
            }
            for key, value in dic.items():
                dic[key] = round_two_decimal(value)
            return_person.append(dic)
        return return_person

    def add_memory(
        self,
        virtual_date,
        iteration,
        stock_op,
        type,
        gossip,
        analysis_stocks,
        analysis_strategy,
        market_index,
        stocks_list,
    ):
       # market_index = "Current market index change: {:.2f}%".format(
         #   market_index.query_market_index_intraday_percentage(virtual_date) * 100
       # )
        cmd = "insert into memory values(?,?,?,?,?,?,?,?,?,?,?,?)"
        params = (
            self.person_id,
            virtual_date,
            iteration,
            str(stock_op),
            str(self.principle),
            str(type),
            str(gossip),
            str(analysis_stocks),
            str(analysis_strategy),
            str([]),  # integrate_stock_info(virtual_date, stocks_list)
            str(market_index),
            str(integrate_hold_info(virtual_date, self)),
        )
        self.db.execute_sql_params(cmd, params)

    def query_memory(self, virtual_date):
        cmd = "select * from memory where virtual_date ={} and person_id ={} and stock_operations <> 'None'".format(
            virtual_date, self.person_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_memory(results)
        return_memory = []
        for memory in results:
            date = memory["virtual_date"]
            iter = memory["iteration"]
            stock_op = memory["stock_operations"]
            strategy = memory["strategy"]
            stock_prices = memory["stock_prices"]
            market_change = memory["market_change"]
            financial_situation = memory["financial_situation"]
            analysis_for_stocks = memory["analysis_for_stocks"]
            gossip = memory["gossip"]
            dic = {
                "Virtual_date": date,
                "Iteration": iter,
                "Stock_op": stock_op,
                "Strategy": strategy,
                "Stock_prices": stock_prices,
                "Market_change": market_change,
                "Financial_situation": financial_situation,
                "Analysis_for_stocks": analysis_for_stocks,
                "Gossip": gossip,
            }
            for key, value in dic.items():
                dic[key] = round_two_decimal(value)
            return_memory.append(dic)

        return return_memory

    def add_gossip(self, virtual_date, gossip):
        cmd = "insert into gossip values({}, {}, \'{}\')".format(
            self.person_id, virtual_date, gossip
        )
        self.db.execute_sql(cmd)

    def query_gossip(self, virtual_date):
        cmd = "select * from gossip where virtual_date ={} and person_id !={}".format(
            virtual_date, self.person_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_gossip(results)
        return results


    def stock_ops(self, virtual_date, persons, stocks, market_index, iter):
        market_data = query_prices(virtual_date)
        ops = self.tech_trader.trade_signal(market_data)
        result = self.tech_trader.convert_to_standard(ops)
        if not result or "stocks" not in result:
            return {"stocks": [{"stock_id": None, "operation": 0.0}], "total_position": 0.0}
        stock_ops_list = result["stocks"]
        total_position = result.get("total_position", None)
        for trading in stock_ops_list:
            if trading['operation']>0:
                self.add_memory(
                virtual_date,
                iter,
                trading,
                "buy",
                [],
                [],
                str(total_position),
                [],
                self.stocks,
            )
            elif trading['operation']<0:
                self.add_memory(
                virtual_date,
                iter,
                trading,
                "sell",
                [],
                [],
                str(total_position),
                [],
                self.stocks,
            )
            elif trading['operation']==0:
                self.add_memory(
                virtual_date,
                iter,
                trading,
                "hold",
                [],
                [],
                str(total_position),
                [],
                self.stocks,
            )
        return result
