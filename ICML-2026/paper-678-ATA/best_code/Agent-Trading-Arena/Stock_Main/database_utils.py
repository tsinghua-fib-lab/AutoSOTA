import sqlite3
import time
import matplotlib.pyplot as plt
import base64
import numpy as np
import mplfinance as mpf
import pandas as pd


current_milli_time = lambda: int(round(time.time() * 1000))

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


def query_all_stocks(db, virtual_date):
    cmd = "select * from stock where virtual_date ={} and stock_id >= 0 order by stock_id".format(
        virtual_date, -1
    )
    db.execute_sql(cmd)
    results = db.fetchall()
    results = parse_stocks(results)
    if len(results) >= 1:
        return results
    else:
        return None
def stock_(stock_id, start_date=-9, end_date=0):
    conn = sqlite3.connect('./save/sim01/data.db')
    cursor = conn.cursor()
    cmd = """
    SELECT *
    FROM stock
    WHERE stock_id = {} 
    AND virtual_date BETWEEN '{}' AND '{}'
    """.format(stock_id, start_date, end_date)

    cursor.execute(cmd)
    results = cursor.fetchall()
    name_tags = [
        "stock_id",
        "virtual_date",
        "weekday",
        "Volume",
        "quantity",
        "Close",
        "Open",
        "High",
        "Low",
    ]
    results = pd.DataFrame(results, columns= name_tags)
    results["date"] = pd.date_range(start='2024-08-06', periods=len(results), freq='D')
    results.set_index('date', inplace=True)

    results = results.drop(columns=['stock_id', 'weekday', 'quantity','virtual_date'])
    return results

#def save_plot_stocks(virtual_date):
#    for i in range(Num_Stock):
#        stock_A=stock_(i, start_date=-9, end_date=virtual_date)
#        #stock_B=stock_(1, start_date=-9, end_date=virtual_date)
        #stock_C=stock_(2, start_date=-9, end_date=virtual_date)
#        virtual_dates=[i-9 for i in range(len(stock_A))]
#        fig, axes =mpf.plot(stock_A, type='candle', style='charles', title='Daily K-line chart of stock {}'.format(STOCK_NAME[i]), ylabel='Price',returnfig=True)
#        axes[0].set_xticks(range(len(virtual_dates)))  # 
#        axes[0].set_xticklabels(virtual_dates)
        #plt.savefig('stock_{}_price.pdf'.format(STOCK_NAME[i]))
        #plt.close(fig)  #
#        plt.savefig('stock_{}_price.jpg'.format(STOCK_NAME[i]), dpi=500)
#        plt.close(fig)

def trans_url(photo_path):
    with open(photo_path, 'rb') as image_file:  
        image_data = image_file.read()  
    image_base64 = base64.b64encode(image_data).decode('utf-8') 
    image_url = f'data:image/jpeg;base64,{image_base64}' 
    return image_url 

def submit_order(
    db, order_type, person_id, stock_id, virtual_date, iteration, bid_price, quantity
):
    current_time = current_milli_time()
    assert quantity > 0
    time.sleep(0.01)
    weekday = virtual_date % 7  # a week of 7 days
    cmd_insert = "Insert Into active_orders values({},{},{},{},{},{},'{}',{},{},'active')".format(
        current_time,
        virtual_date,
        weekday,
        iteration,
        stock_id,
        person_id,
        order_type,
        bid_price,
        quantity,
    )
    db.execute_sql(cmd_insert)


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

    def execute_sql(self, cmd: str) -> bool:
        try:
            self._cur.execute(cmd)
            self._conn.commit()
        except Exception as e:
            print("Database ERROR:{}".format(cmd))
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
