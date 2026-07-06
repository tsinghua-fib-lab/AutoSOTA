import datetime
import numpy as np
import sqlite3
import json
import time

from decoupledmarket.database_utils import Database_operate, parse_orders
from decoupledmarket.Stock import Stock
from decoupledmarket.constant import Daily_Price_Limit, Fluctuation_Constant
from decoupledmarket.fundamentals import FundamentalsLayer
import random
from typing import List, Tuple, Dict
import math

class Market:
    def __init__(self, broker, persons, stocks, database: Database_operate, bid_step=1):
        self._bid_step = bid_step  # in percentage
        self.db = database
        self.stocks = stocks
        self.broker = broker
        self.persons = persons
        self.today_index = 7  # index of date
        self.sum_quantity=0
      #  self.portfolio = portfolio
        self.fundamentals_layer = FundamentalsLayer(payout_ratio=[0.2, 0.3, 0.5], volatility=0.1)

    def end_of_day(self, virtual_date):
        # put all the stocks available for selling, a bank is needed
        all_orders = self._fetch_orders("all", -1)  # fetch all active orders
        for each_order in all_orders:
            cmd = "update active_orders set status='closed' where timestamp={}".format(
                each_order["timestamp"]
            )
            self.db.execute_sql(cmd)

    def update_price(
        self,
        virtual_date: int,
        stock_id: int,
        cur_stock_price: float,
        deal_price: float,
        trade_quantity: int,
        total_quantity: int,
        price_history: List[float] = None,
        config: Dict = None
        ) -> float:
        """Docstring."""
        if deal_price == cur_stock_price:

            variation = random.uniform(-0.01, 0.01)
            deal_price = cur_stock_price * (1 + variation)


        if config is None:
            config = {}


        spread_pct = config.get('spread_pct', 0.02)
        impact_factor = config.get('impact_factor', 0.01)
        trend_weight = config.get('trend_weight', 0.3)
        use_vwap = config.get('use_vwap', True)
        config.setdefault('volume_exponent', 1.3)
        config.setdefault('weights', [0.9, 0.1, 0.0])


        rel_volume = trade_quantity / total_quantity if total_quantity > 0 else 0



        price_impact = impact_factor * (rel_volume ** 1.3)



        if deal_price > cur_stock_price:

            impact_direction = 1
        else:

            impact_direction = -1


        trend_effect = 0
        if len(price_history) >= 5:
            recent_change = sum(price_history.iloc[i] - price_history.iloc[i+1]
                            for i in range(len(price_history)-1))
            trend_effect = recent_change / cur_stock_price * trend_weight



        method1_price = deal_price * (1 + impact_direction * price_impact)


        method2_price = (deal_price * trade_quantity + cur_stock_price * total_quantity) / (trade_quantity + total_quantity)


        method3_price = cur_stock_price * (1 + trend_effect)


        weights = config.get('weights', [0.8, 0.1, 0.1])
        new_price = (
            method1_price * weights[0] +
            method2_price * weights[1] +
            method3_price * weights[2]
        )


        if config.get('add_random_noise', True):
            noise_level = config.get('noise_level', 0.005)
            import random
            random_factor = 1 + random.uniform(-noise_level, noise_level)
            new_price *= random_factor


        new_price = max(new_price, 0.01)


        bid_price = new_price * (1 - spread_pct/2)
        ask_price = new_price * (1 + spread_pct/2)


        return new_price

    def _calculate_order_effect(
        self,
        cur_price: float,
        deal_price: float,
        trade_qty: int,
        total_qty: int,
        cfg: Dict
        ) -> float:
        """"""
        if trade_qty <= 0:
            return 0.0


        price_diff = (deal_price - cur_price) / cur_price


        volume_ratio = min(1.0, trade_qty / max(100, total_qty * 0.01))
        volume_effect = volume_ratio * 0.02


        if price_diff > 0:
            direction = 1.0
        elif price_diff < 0:
            direction = -1.0
        else:
            direction = 0.0


        order_effect = price_diff * 0.3 + volume_effect * direction * 0.7


        return max(-0.05, min(0.05, order_effect))
    def _calculate_random_effect(self, cur_price: float, cfg: Dict) -> float:
        """"""

        volatility = cfg['base_volatility']


        base_random = random.gauss(0, volatility)


        if cur_price > 1000:
            price_scale = 0.6
        elif cur_price > 500:
            price_scale = 0.8
        elif cur_price > 100:
            price_scale = 1.0
        elif cur_price > 50:
            price_scale = 1.2
        elif cur_price > 10:
            price_scale = 1.5
        else:
            price_scale = 2.0


        volume_impact = 1.0
        if hasattr(self, '_last_trade_qty') and hasattr(self, '_last_total_qty'):
            volume_ratio = self._last_trade_qty / max(1, self._last_total_qty)
            volume_impact = 1.0 + volume_ratio * 3


        hour_of_day = (self.virtual_date % 24) if hasattr(self, 'virtual_date') else 12
        if 9 <= hour_of_day <= 11 or 13 <= hour_of_day <= 15:
            time_factor = random.uniform(1.0, 1.5)
        else:
            time_factor = random.uniform(0.5, 1.0)


        random_effect = base_random * price_scale * volume_impact * time_factor * 0.01


        return max(-0.03, min(0.03, random_effect))

    def _calculate_reversion_effect(
        self,
        cur_price: float,
        price_history: List[float],
        lookback_days: int = 10
        ) -> float:
        """Docstring."""
        if len(price_history) < 5:
            return 0.0


        recent_prices = price_history[-lookback_days:] if len(price_history) >= lookback_days else price_history
        avg_price = np.mean(recent_prices)


        deviation = (cur_price - avg_price) / avg_price


        reversion_strength = 0.3


        if abs(deviation) > 0.1:
            reversion_factor = -deviation * reversion_strength * 1.5
        elif abs(deviation) > 0.05:
            reversion_factor = -deviation * reversion_strength
        else:
            reversion_factor = -deviation * reversion_strength * 0.5

        return reversion_factor

    def end_of_market(self, virtual_date):
        # put all the rest active order traded by the broker.
        all_orders = self._fetch_orders("all", -1)  # fetch all active orders
        for each_order in all_orders:
            stock_id = each_order["stock_id"]
            current_stock = self.stocks[stock_id]
            trade_quantity = each_order["quantity"]
            total_quantity = current_stock.quantity
            cur_stock_price = current_stock.current_price
            status = "finished"
            deal_price = cur_stock_price * np.random.uniform(0.995, 1.005)#(current_stock.current_price + each_order["price"]) / 2
            # check if we should skip this order
            if (
                abs(deal_price - cur_stock_price) / cur_stock_price
            ) > Daily_Price_Limit:
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

            price_history_all = self.stocks[stock_id].query_prices(virtual_date)
            price_history = price_history_all["close"].iloc[-5:]
            self.stocks[stock_id].current_price = self.update_price(
                virtual_date=virtual_date,
                stock_id=stock_id,
               cur_stock_price=current_stock.current_price,
                deal_price=deal_price,
                trade_quantity=trade_quantity,
                total_quantity=total_quantity,
                price_history=price_history,
                config={
                    'order_impact': 0.7,
                    'random_impact': 0.2,
                    'reversion_impact': 0.1,
                    'max_change_per_trade': 0.08,
                    'spread_pct': 0.03,
                    'impact_factor': 0.02,
                    'trend_weight': 0.4,
                    'noise_level': 0.01
                }
            )
            self.sum_quantity = self.sum_quantity + each_order["quantity"]
            self.stocks[stock_id].update_trade_data(
                virtual_date, self.stocks[stock_id].current_price, trade_quantity
            )
            deal_price = self.stocks[stock_id].current_price

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

    def match_order(self, today):
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

            self.stocks[stock_iter].update_trade_data(today, cur_stock_price, 0)
            while current_buy is not None and current_sell is not None:
                deal_price = (current_buy["price"] + current_sell["price"]) / 2
                if (
                    abs(deal_price - cur_stock_price) / cur_stock_price
                ) > Daily_Price_Limit:
                    # close this round of matching, update orders
                    break

                # update the prices of stocks
                trade_quantity = min(
                    [current_buy["quantity"], current_sell["quantity"]]
                )
                price_history_all = self.stocks[stock_iter].query_prices(today)
                price_history = price_history_all["close"].iloc[-5:]
                self.stocks[stock_iter].current_price = self.update_price(
                    virtual_date=today,
                    stock_id=stock_iter,
                    cur_stock_price=cur_stock_price,
                    deal_price=deal_price,
                    trade_quantity=trade_quantity,
                    total_quantity=total_quantity,
                    price_history=price_history,
                    config={
                        'order_impact': 0.7,
                        'random_impact': 0.2,
                        'reversion_impact': 0.1,
                        'max_change_per_trade': 0.08,
                        'spread_pct': 0.03,
                        'impact_factor': 0.02,
                        'trend_weight': 0.4,
                        'noise_level': 0.01
                    }
                )
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

    def _fetch_orders(self, type, fetch_stock_id):
        if type == "buy":
            fetch_cmd = (
                "select * from active_orders where type ='{}' and stock_id={} and status='active' "
                "order by price ASC, timestamp DESC"
            ).format(type, fetch_stock_id)
            # NOTE: match_order uses list.pop(); this ordering makes pop() pick
            # highest buy price first, with earlier timestamp first at same price.
        elif type == "sell":
            fetch_cmd = (
                "select * from (select * from active_orders where type ='{type}' and stock_id={id} "
                "and status='active' and "
                "person_id=-1) "
                "union all "
                "select * from (select * from active_orders where type ='{type}' and stock_id={id} "
                "and status='active' and "
                "person_id>=0) "
                "order by price DESC, timestamp DESC"
            ).format(type="sell", id=fetch_stock_id)
            # NOTE: pop() then picks lowest sell price first, time-priority preserved.
        elif type == "all":
            fetch_cmd = (
                "select * from active_orders where status='active' order by timestamp ASC"
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

            #self.persons[order["person_id"]].settlement(order, price, quantity)

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
            # self.persons[order["person_id"]].settlement(order, price, quantity)

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
