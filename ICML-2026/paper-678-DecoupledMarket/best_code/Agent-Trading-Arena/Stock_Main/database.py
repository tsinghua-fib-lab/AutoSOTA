import sqlite3
import time
import random
from constant import current_milli_time


def parse_ops(ops_str):
    return_lists = []
    name_tags = [
        "timestamp",
        "virtual_date",
        "weekday",
        "iteration",
        "choice",
        "stock_id",
        "person_id",
        "action",
        "price",
        "quantity",
        "importance"
    ]
    for each in ops_str:
        return_dic = {}
        for index, name in enumerate(name_tags):
            return_dic[name] = each[index]
        return_lists.append(return_dic)
    # print(return_lists)
    return return_lists


def query_all_ops(db, action):
    cmd = "select * from operations where action = '{}'".format(
        action, -1
    )
    db.execute_sql(cmd)
    results = db.fetchall()
    results = parse_ops(results)
    if len(results) >= 1:
        return results[0]
    else:
        return None


class Database_operate:
    def __init__(self, db_name):
        self._db_name = db_name
        self._conn = None  # database connections
        self._cur = None  # database cursor

        self.init_database()

    def init_database(self):
        self._conn = sqlite3.connect("{}.db".format(self._db_name))
        self._cur = self._conn.cursor()
        cmdcre_op = (
            "Create Table operations (timestamp Integer NOT NULL PRIMARY KEY, virtual_date text, "
            "weekday INTEGER, iteration INTEGER, choice text check(choice IN ('work','stock')),"
            "stock_id INTEGER, person_id INTEGER, action text check(action IN ('Sell','Buy')), "
            "price Numeric, quantity INTEGER, importance Integer)"
        )
        self.execute_sql(cmdcre_op)

    def execute_sql(self, cmd: str) -> bool:
        try:
            self._cur.execute(cmd)
            self._conn.commit()
        except Exception as e:
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


class operations:
    def __init__(self, database: Database_operate):
        self.db = database

    def insert_op(self, ops, persona):
        for op in ops:
            parts = op.split(', ')
            action = parts[0].split(': ')[0]
            stock_info = parts[0].split(': ')[1]
            quantity = parts[1].split(': ')[1]
            price = parts[2].split(': ')[1]
            time.sleep(0.01)
            current_time = current_milli_time()

            importance_list = [2, 3, 5]
            random_choice = random.choice(importance_list)

            cmd_insert_ops = "Insert Into operations values({},'{}',{},{},'{}',{},{},'{}',{},{},{})".format(
                             current_time, "vir", 1, 2, "stock", 3, 4, action, price, quantity, random_choice)
            self.db.execute_sql(cmd_insert_ops)

    def prompt_generation(self):
        prompts = ''

        cmd_selected_ops = "select * from operations where importance > {}".format(1)
        self.db.execute_sql(cmd_selected_ops)
        selected_ops = self.db.fetchall()
        selected_ops = parse_ops(selected_ops)
        for s_op in selected_ops:
            work_or_stock = "At {}, {} chose to {}.\n".format(s_op["virtual_date"], s_op["person_id"], s_op["choice"])
            if s_op["action"] == "Buy":
                stock_ops = "At {}, {} bought {} shares of {} at {} per share\n"\
                            .format(s_op["virtual_date"],s_op["person_id"], s_op["quantity"], s_op["stock_id"], s_op["price"])
            else:
                stock_ops = "By {}ing {} shares of {} at {} per share, {} earned 3000.\n"\
                            .format(s_op["action"], s_op["quantity"], s_op["stock_id"], s_op["price"], s_op["person_id"])
            price_change = "The past prices of {} are [1, 2, 3, 4]\n".format(s_op["stock_id"])
            prompts += (work_or_stock + stock_ops + price_change)
        print(prompts)

        return prompts


if __name__ == "__main__":
    data = ["Buy: Tesla, Quantity: 100, Quoted price: 468.12", "Sell: Nvidia, Quantity: 100, Quoted price: 439.4", "Sell: Nvidia, Quantity: 200, Quoted price: 439.4"]
    table = Database_operate("test")
    op_table = operations(table)
    op_table.insert_op(data, persona="Mo")
    prompts = op_table.prompt_generation()
