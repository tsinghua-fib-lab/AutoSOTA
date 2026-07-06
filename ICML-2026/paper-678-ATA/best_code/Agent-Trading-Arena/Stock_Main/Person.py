import time
from database_utils import (
    Database_operate,
    query_all_stocks,
    parse_accounts,
    parse_memory,
    parse_persons,
    parse_gossip,
    submit_order,
    stock_name_to_id,
    round_two_decimal,
)
from load_json import load_persona
from constant import STOCK_NAMES
from content.our_run_gpt_prompt import integrate_hold_info, integrate_stock_info


class Broker:
    def __init__(self, stocks, database: Database_operate, folder_mem_saved=False):
        self.person_id = -1
        self.stocks = stocks
        self.db = database
        self.inventories = []
        self.dividends = []

        self.cash = None
        self.asset = None
        self.wealth = None
        self.total_expense = 0

        self.initialize_broker()

    def initialize_broker(self):
        self.cash = 0
        self.asset = 0
        # process IPO of stock
        for each_stock in self.stocks:
            self.inventories.append(each_stock.quantity)
            cmd = ("insert into account values({},{},{},{},{},{},{},{},{})").format(
                self.person_id,
                each_stock.stock_id,
                0,
                0,
                each_stock.quantity,
                0,
                each_stock.current_price,
                0,
                0,
            )
            self.db.execute_sql(cmd)
            self.asset += each_stock.current_price * each_stock.quantity
            self.dividends.append(0)

        self.wealth = self.cash + self.asset  # no asset at the beginning

        cmd = "Insert Into person values({},{},{},{},{},{},{},{},'NA')".format(
            self.person_id, 0, self.cash, self.asset, self.wealth, 0, 0, 0,
        )
        self.db.execute_sql(cmd)

    def count_expense(self, expense):
        self.total_expense += expense
        self.cash += expense

    def settlement(self, order, price, quantity):
        if order["type"] == "sell":
            order_volume = price * quantity
            self.cash += order_volume
            self.dividends[order["stock_id"]] += order_volume
            self.asset -= order_volume
            self.inventories[order["stock_id"]] -= quantity
            new_quantity = self.inventories[order["stock_id"]]
            assert new_quantity >= 0
            cmd = (
                "update account set  quantity={} where stock_id={} and virtual_date={} and "
                "person_id={}"
            ).format(
                new_quantity, order["stock_id"], order["virtual_date"], self.person_id,
            )
            self.db.execute_sql(cmd)
            # insert memory, to be filled
        if order["type"] == "buy":
            order_volume = price * quantity
            self.cash -= order_volume
            self.dividends[order["stock_id"]] -= order_volume
            self.asset += order_volume
            self.inventories[order["stock_id"]] += quantity
            new_quantity = self.inventories[order["stock_id"]]
            assert new_quantity >= 0
            cmd = (
                "update account set  quantity={} where stock_id={} and virtual_date={} and "
                "person_id={}"
            ).format(
                new_quantity, order["stock_id"], order["virtual_date"], self.person_id,
            )
            self.db.execute_sql(cmd)

    def ipo(self, virtual_date):
        for each_stock in self.stocks:
            if self.inventories[each_stock.stock_id] > 0:
                submit_order(
                    self.db,
                    "sell",
                    self.person_id,
                    each_stock.stock_id,
                    virtual_date,
                    0,
                    each_stock.current_price,
                    self.inventories[each_stock.stock_id],
                )

    def end_of_day(self, virtual_date):
        total_asset = 0
        day_offset = 1
        new_date = virtual_date + day_offset
        for each_stock in self.stocks:
            # update price for the next day
            cmd = ("insert into account values({},{},{},{},{},{},{},{},{})").format(
                self.person_id,
                each_stock.stock_id,
                new_date,
                new_date % 7,
                self.inventories[each_stock.stock_id],
                0,
                each_stock.current_price,
                0,
                0,
            )
            self.db.execute_sql(cmd)
            total_asset += (
                each_stock.current_price * self.inventories[each_stock.stock_id]
            )

        # update asset
        self.asset = total_asset
        self.wealth = self.cash + self.asset
        cmd = ("insert into person values({},{},{},{},{},{},{},{},'NA')").format(
            self.person_id, new_date, self.cash, self.asset, self.wealth, 0, 0, 0,
        )
        self.db.execute_sql(cmd)


class Person:
    def __init__(
        self,
        person_id,
        broker: Broker,
        stocks,
        database: Database_operate,
        persona_path,
        folder_mem_saved=False,
    ):
        self.person_id = person_id
        self.broker = broker
        self.stocks = stocks
        self.db = database

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

        self.initialize_person(persona_path)

    def initialize_person(self, persona_path):
        persona = load_persona(persona_path)
        for p in persona:
            if self.person_id == p["person_id"]:
                self.identity = p
                self.income = p["daily_income_from_job"]
                self.principle = p["principle"]
                self.cash = p["cash"]
                self.minimum_living_expense = p["minimum_living_expense"]
                self.reflect_frequency = p["reflect_frequency"]

        self.asset = 0

        self.wealth = self.cash + self.asset  # no asset at the beginning

        cmd = """Insert Into person values({},{},{},{},{},{},{},{},'{}')""".format(
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

    def extract(self, op):
        order_type = op[0]
        if order_type == "hold":
            return order_type, None, None, None
        else:
            stock_name = op[1]
            bid_price = float(op[2])
            quantity = float(op[3])
        return order_type, stock_name, bid_price, quantity

    
    def create_order(self, i, op, virtual_date, iteration=0):

        order_type, stock_name, bid_price, quantity = self.extract(op)
        stock_id = stock_name_to_id(self.stocks, stock_name)
        order_type = order_type.lower()  # convert to lowercase

        # timestamp int, virtual_date text, weekday int, stock_id int, person_id int, type text, price float
        if order_type == "buy":
            cash_expense = bid_price * quantity
            bid_price = bid_price * 1.05  
            if cash_expense < self.cash:
                submit_order(#insert active_orders
                    self.db,
                    order_type,
                    self.person_id,
                    stock_id,
                    virtual_date,
                    iteration,
                    bid_price,
                    quantity,
                )
        if order_type == "sell":
            bid_price = bid_price * 0.95
            stock_onhold = self.query_single_stock(virtual_date, stock_id)
            if stock_onhold is not None:
                new_quantity = stock_onhold["quantity"] - quantity
                if new_quantity >= 0:
                    submit_order(
                        self.db,
                        order_type,
                        self.person_id,
                        stock_id,
                        virtual_date,
                        iteration,
                        bid_price,
                        quantity,
                    )

    def settlement(self, order, price, quantity):
        # financial settlement at the end of a day
        type = order["type"]
        order_volume = price * quantity
        stock_onhold = self.query_single_stock(order["virtual_date"], order["stock_id"])
        print("settlement stock_onhold:",stock_onhold)
        if order["type"] == "buy":
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
        #print("end_of_iteration all_stocks:",all_stocks)
        hold_stocks = self.query_hold_stocks(virtual_date)
        #print("end_of_iteration hold_stocks:",hold_stocks)
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
       # print("person_id:",self.person_id,"total_asset:",self.wealth,"capital_gain:",capital_gain)

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

    def end_of_day(self, virtual_date, args):
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
            #calculate dividend
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
        self.daily_expense = (
            total_asset * 1.0 + self.cash
        ) * args.expense_ratio + self.minimum_living_expense
        self.cash -= self.daily_expense
        self.wealth = self.asset + self.cash
        self.broker.count_expense(self.daily_expense)
        #print("person_id:",self.person_id,";total_wealth:",self.wealth,";cash:",self.cash,";asset:",self.asset)
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
                    name=args.STOCK_NAMES[stock_id],
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
        market_index = "Current market index change: {:.2f}%".format(
            market_index.query_market_index_intraday_percentage(virtual_date) * 100
        )#(last-begin)/begin___change price percentage
        cmd = (
            """insert into memory values({},{},{},"{}","{}","{}","{}","{}","{}","{}","{}","{}")"""
        ).format(
            self.person_id,
            virtual_date,
            iteration,
            stock_op,
            self.principle,
            type,
            gossip,
            analysis_stocks,
            analysis_strategy,
            integrate_stock_info(virtual_date, stocks_list),
            market_index,
            integrate_hold_info(virtual_date, self),
        )
        self.db.execute_sql(cmd)

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
    
    def query_person(self, virtual_date):
        cmd = "select * from person where virtual_date ={} and person_id ={}".format(
            virtual_date, self.person_id
        )
        self.db.execute_sql(cmd)
        results = self.db.fetchall()
        p = parse_persons(results)
        return_person = []
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
            "capital_gain": capital_gain
            }
        for key, value in dic.items():
            dic[key] = round_two_decimal(value)
        return_person.append(dic)

        return return_person

    def add_gossip(self, virtual_date, gossip):
        cmd = ("""insert into gossip values({},{},"{}")""").format(
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
