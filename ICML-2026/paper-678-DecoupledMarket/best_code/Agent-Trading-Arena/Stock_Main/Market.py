import datetime
import numpy as np
import sqlite3
import json
import time

from database_utils import Database_operate, parse_orders
from Stock import Stock
#from constant import Daily_Price_Limit, Fluctuation_Constant


class Market:
    def __init__(self, broker, persons, stocks, database: Database_operate, bid_step=1):
        self._bid_step = bid_step  # in percentage
        self.db = database
        self.stocks = stocks
        self.broker = broker
        self.persons = persons
        self.today_index = 7  # index of date

    def end_of_day(self, virtual_date):
        # put all the stocks available for selling, a bank is needed
        all_orders = self._fetch_orders("all", -1)  # fetch all active orders
        for each_order in all_orders:
            cmd = "update active_orders set status='closed' where timestamp={}".format(
                each_order["timestamp"]
            )
            self.db.execute_sql(cmd)

    def end_of_market(self, virtual_date, args):
        # put all the rest active order traded by the broker.
        all_orders = self._fetch_orders("all", -1)  # fetch all active orders
       # print("end_of_market all_orders:",all_orders)
        for each_order in all_orders:
            stock_id = each_order["stock_id"]
            current_stock = self.stocks[stock_id]
            trade_quantity = each_order["quantity"]
            total_quantity = current_stock.quantity
            cur_stock_price = current_stock.current_price
            status = "finished"
            
            deal_price = (current_stock.current_price + each_order["price"]) / 2
            # check if we should skip this order
            if (
                abs(deal_price - cur_stock_price) / cur_stock_price
            ) > args.Daily_Price_Limit:#constant:0.7
                continue
            if (
                self.broker.inventories[stock_id] <= 0
                or each_order["person_id"] == -1
                or trade_quantity <= 0
            ):
                continue
            # check is there enough stock for finishing the order
            if (
                each_order["type"] == "buy"
                and trade_quantity > self.broker.inventories[stock_id]
            ):
                rest_quantity = trade_quantity - self.broker.inventories[stock_id]
                trade_quantity = self.broker.inventories[stock_id]
                status = "partially fulfilled"

            # update the prices of stocks
            self.stocks[stock_id].current_price = (
                deal_price * trade_quantity * args.Fluctuation_Constant
                + cur_stock_price * total_quantity
            ) / (trade_quantity * args.Fluctuation_Constant + total_quantity)

            cur_stock_price = self.stocks[stock_id].current_price
            deal_price = cur_stock_price
            self.stocks[stock_id].update_trade_data(
                virtual_date, cur_stock_price, trade_quantity
            )

            # update both order status
            # finish the order first
            self._update_order(
                each_order, deal_price, status, trade_quantity,
            )
            if status == "partially fulfilled":
                # make the rest order still active
                self._update_order(
                    each_order, deal_price, "update", rest_quantity,
                )

            # broker update
            new_type = "buy" if each_order["type"] == "sell" else "sell"
            order = {
                "stock_id": stock_id,
                "type": new_type,
                "virtual_date": virtual_date,
            }
            
            self.broker.settlement(order, deal_price, trade_quantity)

    def match_order(self, today, args):
        for stock_iter in range(len(self.stocks)):
            buy_orders = self._fetch_orders("buy", stock_iter)
            sell_orders = self._fetch_orders("sell", stock_iter)
            # start to match order
            cur_stock_price = self.stocks[stock_iter].current_price
            total_quantity = self.stocks[stock_iter].quantity
            trade_quantity = 0
            current_buy = buy_orders.pop() if buy_orders else None #status
            current_sell = sell_orders.pop() if sell_orders else None
            residual_order = None  # only part of the quantity have been processed

            # initialize each stock of the day
            self.stocks[stock_iter].update_trade_data(today, cur_stock_price, 0)

            while current_buy is not None and current_sell is not None:
                deal_price = (current_buy["price"] + current_sell["price"]) / 2
                if (
                    abs(deal_price - cur_stock_price) / cur_stock_price
                ) > args.Daily_Price_Limit:
                    # close this round of matching, update orders
                    break

                # update the prices of stocks
                trade_quantity = min(
                    [current_buy["quantity"], current_sell["quantity"]]
                )
               
                self.stocks[stock_iter].current_price = (
                    deal_price * trade_quantity * args.Fluctuation_Constant #20.0
                    + cur_stock_price * total_quantity
                ) / (trade_quantity * args.Fluctuation_Constant + total_quantity)
                cur_stock_price = self.stocks[stock_iter].current_price
                self.stocks[stock_iter].update_trade_data(
                    today, cur_stock_price, trade_quantity
                )
                deal_price = cur_stock_price

                cont_flag = True  # the flag to show is there any order to be matched
                if current_buy["quantity"] > current_sell["quantity"]:
                    
                    self._update_order(
                        current_sell, deal_price, "finished", current_sell["quantity"]
                    )
                    
                    self._update_order(
                        current_buy,
                        deal_price,
                        "partially fulfilled",
                        current_sell["quantity"],
                    )
                    current_buy["quantity"] -= current_sell["quantity"]
                    
                    trade_quantity = current_sell["quantity"]
                    residual_order = current_buy
                    if sell_orders:
                        current_sell = sell_orders.pop()
                    else:
                        current_sell = None
                        cont_flag = False

                if cont_flag and current_buy["quantity"] < current_sell["quantity"]:
                    self._update_order(
                        current_buy, deal_price, "finished", current_buy["quantity"],
                    )
                    self._update_order(
                        current_sell,
                        deal_price,
                        "partially fulfilled",
                        current_buy["quantity"],
                    )
                    current_sell["quantity"] -= current_buy["quantity"]
                    trade_quantity = current_buy["quantity"]
                    residual_order = current_sell
                    if buy_orders:
                        current_buy = buy_orders.pop()
                    else:
                        current_sell = None
                        cont_flag = False

                if cont_flag and current_buy["quantity"] == current_sell["quantity"]:
                    self._update_order(
                        current_sell, deal_price, "finished", current_buy["quantity"]
                    )
                    self._update_order(
                        current_buy, deal_price, "finished", current_buy["quantity"]
                    )
                    current_sell["quantity"] = 0
                    current_buy["quantity"] = 0
                    trade_quantity = current_buy["quantity"]
                    if sell_orders and buy_orders:
                        current_sell = sell_orders.pop()
                        current_buy = buy_orders.pop()
                    else:
                        current_sell = None
                        current_buy = None
                        break

            # process the rest residual order after the matching end
            if residual_order is not None:
                self._update_order(
                    residual_order, deal_price, "update", residual_order["quantity"]
                )

    # Auxiliary Method

    def _fetch_orders(self, type, fetch_stock_id):
        if type == "buy":
            fetch_cmd = (
                "select * from active_orders where type ='{}' and stock_id={} and status='active' "
                "order by price ASC,timestamp ASC;".format(type, fetch_stock_id)
            )
        elif type == "sell":
            fetch_cmd = (
                "select * from (select * from active_orders where type ='{type}' and stock_id={id} "
                "and status='active' and "
                "person_id=-1 order by price DESC,timestamp ASC) union all "
                "select * from (select * from active_orders where type ='{type}' and stock_id={id} "
                "and status='active' and "
                "person_id>=0 order by price DESC,timestamp ASC);".format(
                    type="sell", id=fetch_stock_id
                )
            )
        elif type == "all":
            fetch_cmd = (
                "select * from active_orders where  status='active' "
                "order by price DESC,timestamp ASC;"
            )
        self.db.execute_sql(fetch_cmd)
        results = self.db.fetchall()
        # preprocess orders
        fetch_orders = parse_orders(results)
        return fetch_orders

    def _update_order(self, order, price, type, quantity=0):
        if type == "finished":
            if quantity <= 0:
                quantity = order["quantity"]
            cmd = "update active_orders set quantity={}, status='finished' where timestamp={}".format(
                quantity, order["timestamp"]
            )
            self.db.execute_sql(cmd)
            # settlement of individual trader
            self.persons[order["person_id"]].settlement(order, price, quantity)

        if type == "partially fulfilled":
            cmd = (
                "Insert Into active_orders values({},{},{},{},{},{},'{}',{},{},'{}')"
            ).format(
                order["timestamp"] + 1,
                order["virtual_date"],
                order["weekday"],
                order["iteration"],
                order["stock_id"],
                order["person_id"],
                order["type"],
                price,
                quantity,
                "finished",
            )
            self.db.execute_sql(cmd)
            # settlement of individual trader
            self.persons[order["person_id"]].settlement(order, price, quantity)

        if type == "update":
            cmd = "update active_orders set quantity={} where timestamp={}".format(
                quantity, order["timestamp"]
            )
            self.db.execute_sql(cmd)


if __name__ == "__main__":
    database = Database_operate("Simu0")
    mart = Market("a", database)
    mart.submit_order("buy", 1, 1, 1, 1, 11, 20)
    mart.submit_order("buy", 1, 2, 1, 1, 14, 20)
    mart.submit_order("sell", 2, 1, 1, 1, 10, 30)
    mart.submit_order("sell", 2, 2, 1, 1, 17, 30)
    mart.submit_order("buy", 1, 1, 1, 1, 9.2, 20)
    mart.submit_order("buy", 1, 2, 1, 1, 15, 20)
    mart.match_order(1, 1)
    database.close()
