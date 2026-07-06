import datetime
import pickle
import sqlite3
import json
import time
import os
import os.path as osp
from decoupledmarket.database_utils import query_all_stocks, Database_operate, op_update_reward_strategy
from decoupledmarket.Person import Person, Broker
from decoupledmarket.Stock import Stock, Market_index
from decoupledmarket.Market import Market
from decoupledmarket.Arena import Agents
from decoupledmarket.arena_content.tech_analysis_trader import Tech_traders
from decoupledmarket.behavior import stock_ops, reflection, generate_gossip
from decoupledmarket.constant import *
from decoupledmarket.constant import persona_path, stock_path, agent_path, tech_path, Iterations_Daily, No_Days, Save_Path, virtualagent_path, Num_quantity, Num_Person, Num_Stock, Num_agent, Num_virtual_agents, Num_tech_traders, N
from decoupledmarket.load_json import save_all, load_all
import random
from decoupledmarket.virtual_agent import VirtualAgent
import sys
import logging


def setup_simple_logger():
    """Docstring."""

    log_dir = osp.join(Save_Path, "logs")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = osp.join(log_dir, f"simulation_{timestamp}.log")


    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    if logger.handlers:
        logger.handlers.clear()


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    print(f"Log file saved to: {log_file}")
    return logger

def init_all(load=False):
    if load:
        (
            current_date,
            current_iteration,
            broker,
            market_index,
            market,
            stocks,
            persons,
        ) = load_all()
    else:
        # initialize all objects
        database = Database_operate(osp.join(Save_Path, "data"))
        # clear tables
        cmd = "drop table if exists active_orders"
        database.execute_sql(cmd)
        cmd = "drop table if exists stock"
        database.execute_sql(cmd)
        cmd = "drop table if exists person"
        database.execute_sql(cmd)
        cmd = "drop table if exists account"
        database.execute_sql(cmd)
        cmd = "drop table if exists memory"
        database.execute_sql(cmd)
        cmd = "drop table if exists gossip"
        database.execute_sql(cmd)
        cmd = "drop table if exists agent"
        database.execute_sql(cmd)

        database.init_database()

        stocks = []
        persons = []


        for i in range(Num_Stock):
            stocks.append(Stock(i, database, stock_path))
        market_index = Market_index(stocks, database)
        broker = Broker(stocks, database)

        current_id = 0

        for k in range(N):

            for _ in range(Num_Person):
                persons.append(Person(current_id, broker, stocks, database, persona_path, k))
                current_id += 1

            for _ in range(Num_agent):
                persons.append(Agents(current_id, broker, stocks, database, agent_path, k))
                current_id += 1

            for _ in range(Num_tech_traders):
                persons.append(Tech_traders(current_id, broker, stocks, database, tech_path, k))
                current_id += 1

            for _ in range(Num_virtual_agents):
                persons.append(VirtualAgent(current_id, broker, stocks, database, virtualagent_path, k))
                current_id += 1
            print(f"Total initialized trader IDs: {current_id}")
        persons.append(broker)
        market = Market(broker, persons, stocks, database)

    return 0, 0, broker, market_index, market, stocks, persons


def overall_test():
    (
        current_date,
        current_iteration,
        broker,
        market_index,
        market,
        stocks,
        persons,
    ) = init_all(False)
    for virtual_date in range(No_Days):
        if virtual_date == 0:
            broker.ipo(virtual_date)
        market_index.update_market_index(virtual_date)
        generate_gossip(virtual_date, persons, stocks)
        for iter in range(Iterations_Daily):#3
            total_traders = (Num_Person + Num_agent + Num_tech_traders + Num_virtual_agents)*N
            rand = random.sample(range(0,total_traders),total_traders)
            count = 0
            for i in rand:
                agent = persons[i]
                print(agent.person_id)
                agent_ops = agent.stock_ops(virtual_date, persons, stocks, market_index, iter)
                if agent_ops and "stocks" in agent_ops:
                    stock_ops_list = agent_ops["stocks"]
                    total_position = agent_ops.get("total_position", 1.0)
                    for j in range(len(stock_ops_list)):
                        print("op",stock_ops_list[j],i)
                        op = stock_ops_list[j]
                        persons[i].create_order(op, virtual_date, total_position, iter)
                        count = count + 1
                    if count >= 20:
                        count = 0
                        market.match_order(virtual_date) #
                        market.end_of_market(virtual_date)
                        market_index.update_market_index(virtual_date)
            market.match_order(virtual_date) #
            market.end_of_market(virtual_date)
            market_index.update_market_index(virtual_date)
            for each_person in persons:
                if each_person.person_id >= 0:
                    each_person.end_of_iteration(virtual_date, iter)

            save_all(virtual_date, iter, stocks, market_index, persons, market)


        market.end_of_day(virtual_date)
        for each_person in persons:
            each_person.end_of_day(virtual_date)
        for each_stock in stocks:
            each_stock.end_of_day(virtual_date)
        market_index.end_of_day(virtual_date)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    overall_test()
    # time_test()
    # db_op3()
    # pickle_test()
    # pickle_load()
