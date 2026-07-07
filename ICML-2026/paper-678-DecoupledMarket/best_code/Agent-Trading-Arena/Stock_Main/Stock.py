import datetime
import sqlite3
from load_json import load_stocks
from database_utils import (
    Database_operate,
    parse_stocks,
    query_all_stocks,
    round_lists_two_decimals,
    round_two_decimal,
)



class Stock:
    def __init__(self, stock_id, database: Database_operate, stock_path):
        self.Name = "Default_Stock_Name"
        self.stock_id = stock_id
        self.current_price = None
        self.stock_name = None
        self.quantity = None
        self.book_value = None
        self.DPS = None
        self.intraday_price_list = []
        self.db = database

        self.initialize_stock(stock_path)

    def initialize_stock(self, stock_path):
        stocks = load_stocks(stock_path)
        for s in stocks:
            if self.stock_id == s["stock_id"]:
                self.quantity = s["quantity"]
                self.stock_name = s["stock_name"]
                self.current_price = s["past_stock_last_prices"][-1]
                self.intraday_price_list.append(self.current_price)
                self.book_value = self.current_price * self.quantity
                self.DPS = s["DPS"]
                for day in range(-4, 1):
                    cmd = "Insert Into stock values({},{},{},{},{},{},{},{},{})".format(
                        self.stock_id,
                        day,
                        0,
                        0,
                        0,
                        s["past_stock_last_prices"][-1 + day],
                        s["past_stock_last_prices"][-1 + day],
                        s["past_stock_last_prices"][-1 + day],
                        s["past_stock_last_prices"][-1 + day],
                    )
                    self.db.execute_sql(cmd)

    def end_of_day(self, virtual_date):
        self.intraday_price_list = []
        cmd = "Insert Into stock values({},{},{},{},{},{},{},{},{})".format(
            self.stock_id,
            virtual_date + 1,
            0,
            0,
            0,
            self.current_price,
            self.current_price,
            self.current_price,
            self.current_price,
        )
        self.db.execute_sql(cmd)

    def update_trade_data(self, virtual_date, price, quantity):
        weekday = virtual_date % 7
        # process received data
        stock_status = self.query_price(virtual_date)
        if stock_status is None:
            volume = price * quantity
            cmd = "Insert Into stock values({},{},{},{},{},{},{},{},{})".format(
                self.stock_id,
                virtual_date,
                weekday,
                volume,
                quantity,
                price,
                price,
                price,
                price,
            )
        else:
            self.current_price = price
            self.intraday_price_list.append(self.current_price)
            stock_status["volume"] += price * quantity
            stock_status["quantity"] += quantity
            stock_status["highest_price"] = (
                price
                if price > stock_status["highest_price"]
                else stock_status["highest_price"]
            )
            stock_status["lowest_price"] = (
                price
                if price < stock_status["lowest_price"]
                else stock_status["lowest_price"]
            )
            stock_status["last_price"] = price
            cmd = (
                "update stock set volume={}, quantity={}, highest_price={}, lowest_price={}, last_price={} where "
                "stock_id={} and virtual_date ={}"
            ).format(
                stock_status["volume"],
                stock_status["quantity"],
                stock_status["highest_price"],
                stock_status["lowest_price"],
                stock_status["last_price"],
                stock_status["stock_id"],
                stock_status["virtual_date"],
            )
        self.db.execute_sql(cmd)

    def query_price(self, virtual_date):
        cmd = "select * from stock where stock_id ={} and virtual_date ={}".format(
            self.stock_id, virtual_date
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_stocks(results)
        if len(results) > 1:
            raise TypeError("Multiple stocks are returned")
        elif len(results) == 0:
            return None
        else:
            return results[0]

    def query_intraday_percentage(self, virtual_date):
        # return the price change in percentage within the same day
        cmd = "select * from stock where stock_id ={} and virtual_date ={}".format(
            self.stock_id, virtual_date
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_stocks(results)
        today = results[0]
        percentage = (today["last_price"] - today["begin_price"]) / today["begin_price"]
        return percentage

    def query_daily_return(self, virtual_date, no_days=5):
        # return the past n days return to chatgpt
        cmd = "select * from stock where stock_id ={} and virtual_date between {} and {} order by virtual_date".format(
            self.stock_id, virtual_date - no_days - 2, virtual_date -1
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_stocks(results)
        prices = [self.current_price] * no_days
        daily_returns = [0] * no_days
        volume_fluctuation = [0] * no_days
        date_offset = no_days - len(results) + 1
        last_day_price, last_day_volume = results[0]["last_price"], results[0]["volume"]
        results = results[1:] if len(results) > 1 else []
        for index, each_stock in enumerate(results):
            cur_return = (each_stock["last_price"] - last_day_price) / (
                last_day_price + 1e-9
            )
            cur_fluctuation = (each_stock["volume"] - last_day_volume) / (
                last_day_volume + 1e-9
            )
            prices[index + date_offset] = each_stock["last_price"]
            daily_returns[index + date_offset] = cur_return
            volume_fluctuation[index + date_offset] = cur_fluctuation
            last_day_price, last_day_volume = (
                each_stock["last_price"],
                each_stock["volume"],
            )
        prices = round_lists_two_decimals(prices, False)
        daily_returns = round_lists_two_decimals(daily_returns)
        volume_fluctuation = round_lists_two_decimals(volume_fluctuation)
        return prices

    def query_prompt_values(self, virtual_date, no_days=5):
        # return the past n days return to chatgpt
        cmd = "select * from stock where stock_id ={} and virtual_date between {} and {} order by virtual_date".format(
            self.stock_id, virtual_date - no_days + 1, virtual_date
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_stocks(results)
        prices = [self.current_price] * no_days
        date_offset = no_days - len(results) + 1
        results = results[0:] if len(results) > 1 else []
        for index, each_stock in enumerate(results):
            prices[index] = each_stock["last_price"]
        today = results[-1]
        prices = round_lists_two_decimals(prices, False)
        current_price_change = (self.current_price - today["begin_price"]) / today[
            "begin_price"
        ]
        current_price_change *= 100 # in percentage
        mean_price = (
            sum(self.intraday_price_list) / (len(self.intraday_price_list) + 1e-9)
            if len(self.intraday_price_list) > 0
            else self.current_price
        )
        return_info = {
            "stock_name": self.stock_name,
            "prices": prices,
            "dividend per share": self.DPS,
            "current_price_change": current_price_change,
            "current_price": self.current_price,
            "Intraday_high": today["highest_price"],
            "Intraday_low": today["lowest_price"],
            "Intraday_mean": mean_price,
        }
        for key, value in return_info.items():
            return_info[key] = round_two_decimal(value)
        return return_info


class Market_index:
    def __init__(self, stocks, database: Database_operate):
        self.Name = "Default_Stock_Name"
        self.stock_id = -1  # market index
        self.db = database
        self.stocks = stocks

        total_book_value = sum([stock.book_value for stock in self.stocks])
        self.stock_proportion = [
            stock.book_value / total_book_value for stock in self.stocks
        ]

    def end_of_day(self, virtual_date):
        self.update_market_index(virtual_date + 1)

    def update_market_index(self, virtual_date):
        market_price = self.query_market_index(virtual_date)
        #print("update_market_index market_price:",market_price)
        if market_price is None:
            # calculate market index
            price = 0
            for index, stock in enumerate(self.stocks):
                price += stock.current_price * self.stock_proportion[index]
            cmd = "Insert Into stock values({},{},{},{},{},{},{},{},{})".format(
                -1, virtual_date, 0, 0, 0, price, price, price, price
            )
            self.db.execute_sql(cmd)
        else:
            all_cur_stocks = query_all_stocks(self.db, virtual_date)
           # print("update_market_index all_cur_stocks:",all_cur_stocks)
            price, quantity, volume = (
                0,
                0,
                0,
            )
            for index, stock in enumerate(self.stocks):
                price += stock.current_price * self.stock_proportion[index]
                volume += all_cur_stocks[index]["volume"]
                quantity += all_cur_stocks[index]["quantity"]

            market_price["highest_price"] = (
                price
                if price > market_price["highest_price"]
                else market_price["highest_price"]
            )
            market_price["lowest_price"] = (
                price
                if price < market_price["lowest_price"]
                else market_price["lowest_price"]
            )
            market_price["last_price"] = price

            cmd = (
                "update stock set volume={}, quantity={}, highest_price={}, lowest_price={}, last_price={} where "
                "stock_id={} and virtual_date ={}"
            ).format(
                volume,
                quantity,
                market_price["highest_price"],
                market_price["lowest_price"],
                market_price["last_price"],
                -1,
                virtual_date,
            )
            self.db.execute_sql(cmd)

    def query_market_index(self, virtual_date):
        cmd = "select * from stock where virtual_date ={} and stock_id ={}".format(
            virtual_date, -1
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        results = parse_stocks(results)
        if len(results) >= 1:
            return results[0]
        else:
            return None

#(last-begin)/begin___change price percentage
    def query_market_index_intraday_percentage(self, virtual_date):
        # return the price change in percentage within the same day
        today = self.query_market_index(virtual_date)
        percentage = (today["last_price"] - today["begin_price"]) / today["begin_price"]
        return percentage


class Virtual_date:
    def __init__(self, first_day: datetime.datetime):
        self.first_day = first_day

    def convert_date(self, date_index, iteration):
        output = self.first_day + datetime.timedelta(
            days=date_index, hours=9 + iteration
        )
        return output
