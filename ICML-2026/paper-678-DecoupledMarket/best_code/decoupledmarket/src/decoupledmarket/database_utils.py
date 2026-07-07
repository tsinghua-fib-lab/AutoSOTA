import sqlite3
import time
import os
import os.path as osp
from decoupledmarket.constant import current_milli_time, Save_Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# SQL query templates for parameterized queries
SQL_QUERY_STOCKS_BY_DATE = "SELECT * FROM stock WHERE virtual_date = ? AND stock_id >= 0 ORDER BY stock_id"
SQL_INSERT_ORDER = "INSERT INTO active_orders VALUES(?,?,?,?,?,?,?,?,?,?)"
SQL_QUERY_ACCOUNT = "SELECT * FROM account WHERE virtual_date = ? AND person_id = ? AND quantity > 0"
SQL_QUERY_SINGLE_STOCK = "SELECT * FROM account WHERE virtual_date = ? AND person_id = ? AND stock_id = ?"
SQL_QUERY_PERSON = "SELECT * FROM person WHERE virtual_date = ? AND person_id = ?"
SQL_QUERY_MEMORY = "SELECT * FROM memory WHERE virtual_date = ? AND person_id = ? AND stock_operations <> 'None'"
SQL_INSERT_GOSSIP = "INSERT INTO gossip VALUES(?,?,?)"
SQL_QUERY_GOSSIP = "SELECT * FROM gossip WHERE virtual_date = ? AND person_id != ?"
SQL_UPDATE_ACCOUNT = "UPDATE account SET cost_price=?, quantity=?, profit=? WHERE stock_id=? AND virtual_date=? AND person_id=?"
SQL_INSERT_ACCOUNT = "INSERT INTO account VALUES(?,?,?,?,?,?,?,?,?)"
SQL_INSERT_PERSON = "INSERT INTO person VALUES(?,?,?,?,?,?,?,?,?)"
SQL_UPDATE_PERSON = "UPDATE person SET cash=?, asset=?, wealth=?, capital_gain=? WHERE person_id=? AND virtual_date=?"
def op_update_reward_strategy(virtual_date, persons, iter):
    for p in persons:
        if p.person_id > -1:
            if p.reflect_frequency == 0:
                pass
            elif (iter + 1) % p.reflect_frequency == 0:
                strategy=p.principle
                reward=op_reward(p, virtual_date)
                update_strategy_reward(reward, strategy)
            else:
                pass

def update_strategy_reward(new_reward, new_strategy):
    default_path = osp.join(Save_Path, "strategy.xlsx")
    file_path = os.getenv("STRATEGY_XLSX_PATH", default_path)

    if osp.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["strategy", "reward"])
    if len(df) > 5:

        min_index = df['reward'].idxmin()
        if df.at[min_index, 'reward'] < new_reward:

            df.at[min_index, 'reward'] = new_reward
            df.at[min_index, 'strategy'] = new_strategy
    else:
      #  new_data = {
       #                 "strategy":new_strategy,
        #                "reward": new_reward
         #           }
        new_data = pd.DataFrame([{
                "strategy": new_strategy,
                "reward": new_reward
            }])


        #df.append(new_data)#, ignore_index=True)
        df = pd.concat([df, new_data], ignore_index=True)

        #df = pd.concat([df, new_data], ignore_index=True)


    df.to_excel(file_path, index=False)

def op_reward(persona, virtual_date):

    curr_wealth = persona.wealth
    if virtual_date == 0:
        # Day 0 has no previous-day baseline; use neutral reward.
        return 0.0
    else:
        person = persona.query_person(virtual_date-1)
        if not person or len(person) == 0:

            return 0.0
        last_wealth = person[0]["wealth"]
    if not last_wealth or last_wealth == 0:
        return 0.0
    new_reward = (curr_wealth - last_wealth) / last_wealth * 100
    return new_reward

def parse_gossip(gossip):
    return_lists = []
    name_tags = [
        "person_id",
        "virtual_date",
        "gossip"
    ]
    for each in gossip:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = round_two_decimal(each[index])
        return_lists.append(return_dic)
    return return_lists

def parse_memory(memory):
    return_lists = []
    name_tags = [
        "person_id",
        "virtual_date",
        "iteration",
        "stock_operations",
        "strategy",
        "type",
        "gossip",
        "analysis_for_stocks",
        "analysis_for_strategy",
        "stock_prices",
        "market_change",
        "financial_situation",
    ]
    for each in memory:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = round_two_decimal(each[index])
        return_lists.append(return_dic)
    return return_lists

def parse_stocks(stock_str):
    return_lists = []
    name_tags = [
        "stock_id",
        "virtual_date",
        "weekday",
        "volume",
        "quantity",
        "last_price",
        "begin_price",
        "highest_price",
        "lowest_price",
    ]
    for each in stock_str:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = round_two_decimal(each[index])
        return_lists.append(return_dic)
    return return_lists

def parse_orders(order):
    # timestamp int, virtual_date text, weekday int, "iteration" stock_id int, person_id int, type text, price float
    return_lists = []
    name_tags = [
        "timestamp",
        "virtual_date",
        "weekday",
        "iteration",
        "stock_id",
        "person_id",
        "type",
        "price",
        "quantity",
        "status",
    ]
    for each in order:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = round_two_decimal(each[index])
        return_lists.append(return_dic)
    return return_lists

def parse_persons(persons):
    return_lists = []
    name_tags = [
        "person_id",
        "virtual_date",
        "cash",
        "asset",
        "wealth",
        "work_income",
        "capital_gain",
        "daily_expense",
        "principle",
    ]
    for each in persons:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = round_two_decimal(each[index])
        return_lists.append(return_dic)
    return return_lists

def parse_accounts(accounts):
    return_lists = []
    name_tags = [
        "person_id",
        "stock_id",
        "virtual_date",
        "weekday",
        "quantity",
        "cost_price",
        "current_price",
        "profit",
        "start_date",
    ]
    for each in accounts:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = round_two_decimal(each[index])
        return_lists.append(return_dic)
    return return_lists

def round_two_decimal(input):
    if not isinstance(input, float):
        return input
    try:
        res = float("{:.2f}".format(input))
        return res
    except Exception:
        return input

def round_lists_two_decimals(lists, in_percentage=True):
    if in_percentage:
        return_list = [round_two_decimal(elem * 100) for elem in lists]
    else:
        return_list = [round_two_decimal(elem) for elem in lists]
    return return_list

def stock_name_to_id(stocks, name):
    for each_stock in stocks:
        if each_stock.stock_name == name:
            return each_stock.stock_id

def stock_name_to_price(stocks, name):
    if isinstance(name, int):
        name = chr(ord('A') + name)
    for each_stock in stocks:
        if each_stock.stock_name == name:
            return each_stock.current_price

    print(f"[Warning] stock {name} not found!")
    return 0.0

def query_all_stocks(db, virtual_date):
    cmd = "select * from stock where virtual_date = ? and stock_id >= 0 order by stock_id"
    db.execute_sql_params(cmd, (virtual_date,))
    results = db.fetchall()
    results = parse_stocks(results)
    if len(results) >= 1:
        return results
    else:
        return None

def submit_order(
    db, order_type, person_id, stock_id, virtual_date, iteration, bid_price, quantity
):
    current_time = current_milli_time()
    assert quantity > 0
    time.sleep(0.01)
    weekday = virtual_date % 7  # a week of 7 days
    params = (
        current_time,
        virtual_date,
        weekday,
        iteration,
        stock_id,
        person_id,
        order_type,
        bid_price,
        quantity,
        'active'
    )
    db.execute_sql_params(SQL_INSERT_ORDER, params)

class Database_operate:
    def __init__(self, db_name):
        self._db_name = db_name
        self._conn = None  # database connections
        self._cur = None  # database cursor

        self.init_database()

    def init_database(self):
        self._conn = sqlite3.connect("{}.db".format(self._db_name))
        self._cur = self._conn.cursor()
        cmdcre_orders = (
            "Create Table active_orders (timestamp Integer NOT NULL, virtual_date Integer, "
            "weekday INTEGER, iteration INTEGER,"
            "stock_id INTEGER, person_id INTEGER, type text check(type IN ('sell','buy')), "
            "price Numeric, quantity INTEGER, "
            "status text check (status IN ('active','closed','finished') ))"
        )
        self.execute_sql(cmdcre_orders)

        cmdcre_stock = (
            "Create Table stock (stock_id Integer NOT NULL, virtual_date Integer, "
            "weekday INTEGER,"
            "volume  Numeric, quantity INTEGER, last_price Numeric, begin_price Numeric,"
            "highest_price Numeric, lowest_price Numeric )"
        )
        self.execute_sql(cmdcre_stock)

        cmdcre_person = (
            "Create Table person (person_id Integer, virtual_date Integer, "
            "cash Numeric, asset Numeric,"
            "wealth Numeric, work_income Numeric,"
            "capital_gain Numeric, daily_expense Numeric,"
            "principle Text)"
        )
        self.execute_sql(cmdcre_person)

        cmdcre_account = (
            "Create Table account (person_id Integer, stock_id Integer, virtual_date Integer, "
            "weekday INTEGER, quantity INTEGER,"
            "cost_price Numeric, current_price Numeric, profit Numeric,"
            "start_date INTEGER)"
        )
        self.execute_sql(cmdcre_account)

        cmdcre_account = (
            "Create Table memory (person_id Integer, virtual_date Integer, iteration INTEGER, "
            "stock_operations Text, strategy Text, type Text check(type IN ('sell','buy','hold','reflect')), gossip Text, "
            "analysis_for_stocks Text, analysis_for_strategy Text, stock_prices Text, market_change Text, financial_situation Text)"
        )
        self.execute_sql(cmdcre_account)

        cmdcre_gossip = (
            "Create Table gossip (person_id Integer, virtual_date Integer, gossip Text)"
        )
        self.execute_sql(cmdcre_gossip)

        cmdcre_agent = (
            "Create Table agent (agent_id Integer, virtual_date Integer, "
            "cash Numeric, asset Numeric,"
            "wealth Numeric, work_income Numeric,"
            "capital_gain Numeric, daily_expense Numeric,"
            "principle Text)"
        )
        self.execute_sql(cmdcre_agent)

    def execute_sql(self, cmd: str) -> bool:
        try:
            self._cur.execute(cmd)
            self._conn.commit()
        except Exception as e:
            print("Database ERROR:{}".format(cmd))
            print(e)
            return False
        return True

    def execute_sql_params(self, cmd: str, params: tuple) -> bool:
        try:
            self._cur.execute(cmd, params)
            self._conn.commit()
        except Exception as e:
            print("Database ERROR (parametrized):{}".format(cmd))
            print(e)
            return False
        return True

    def fetchall(self):
        return self._cur.fetchall()

    def close(self):
        self._conn.commit()
        self._conn.close()

    @property
    def cur(self):
        return self._cur
