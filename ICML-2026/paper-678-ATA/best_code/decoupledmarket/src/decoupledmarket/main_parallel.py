"""Parallel simulation entry point."""
import datetime
import time
import os
import os.path as osp
import random
import sys
import logging
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
from decoupledmarket.virtual_agent import VirtualAgent
from decoupledmarket.performance_monitor import get_monitor, reset_monitor
from decoupledmarket.parallel_executor import create_executor, ThreadSafeDatabase
import argparse
from dotenv import load_dotenv
load_dotenv()

# Database table names for cleanup
tables = ["active_orders", "stock", "person", "account", "memory", "gossip", "agent"]

def setup_simple_logger():
    """Configure console and file logging."""
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


def init_all(load=False, use_thread_safe_db=False):
    """Initialize all simulation objects."""
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

        db_path = osp.join(Save_Path, "data")
        if use_thread_safe_db:
            database = ThreadSafeDatabase(db_path)

            for table in tables:
                cmd = f"drop table if exists {table}"
                database.execute_sql(cmd)
            database.init_database()
        else:
            database = Database_operate(db_path)

            for table in tables:
                cmd = f"drop table if exists {table}"
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


def execute_agent_operation(agent, virtual_date, persons, stocks, market_index, iter):
    """Execute one agent operation for parallel workers."""
    monitor = get_monitor() if enable_monitoring else None


    (
        current_date,
        current_iteration,
        broker,
        market_index,
        market,
        stocks,
        persons,
    ) = init_all(False, use_thread_safe_db=use_thread_safe_db)



    total_start_time = time.time()

    try:
        for virtual_date in range(No_Days):
            day_start_time = time.time()
            logging.info("DAY_START virtual_date=%s day=%s/%s", virtual_date, virtual_date + 1, No_Days)

            if monitor:
                monitor.start_timer('day_processing')

            if virtual_date == 0:
                broker.ipo(virtual_date)

            if monitor:
                monitor.start_timer('market_index_update')
            market_index.update_market_index(virtual_date)
            if monitor:
                monitor.end_timer('market_index_update')

            if monitor:
                monitor.start_timer('generate_gossip')
            generate_gossip(virtual_date, persons, stocks)
            if monitor:
                monitor.end_timer('generate_gossip')

            for iter in range(Iterations_Daily):
                iter_start_time = time.time()

                if monitor:
                    monitor.start_timer('iteration_processing')

                total_traders = (Num_Person + Num_agent + Num_tech_traders + Num_virtual_agents) * N
                rand = random.sample(range(0, total_traders), total_traders)


                agents_to_execute = [persons[i] for i in rand]


                if monitor:
                    monitor.start_timer('parallel_agent_execution')

                results = executor.execute_agents(
                    agents_to_execute,
                    execute_agent_operation,
                    virtual_date,
                    persons,
                    stocks,
                    market_index,
                    iter
                )

                if monitor:
                    monitor.end_timer('parallel_agent_execution')


                if monitor:
                    monitor.start_timer('order_processing')

                for result in results:

                    if not result or not isinstance(result, tuple) or len(result) != 2:
                        continue
                    agent, agent_ops = result
                    if agent_ops and isinstance(agent_ops, dict) and "stocks" in agent_ops:
                        stock_ops_list = agent_ops["stocks"]
                        total_position = agent_ops.get("total_position", 1.0)
                        try:
                            total_position = float(total_position)
                        except (TypeError, ValueError):
                            total_position = 1.0
                        if total_position < 0:
                            total_position = 0.0

                        for op in stock_ops_list:
                            agent.create_order(op, virtual_date, total_position, iter)
                            count += 1


                            if count >= batch_size:
                                count = 0
                                if monitor:
                                    monitor.start_timer('market_match_order')
                                market.match_order(virtual_date)
                                if monitor:
                                    monitor.end_timer('market_match_order')

                                if monitor:
                                    monitor.start_timer('market_end_of_market')
                                market.end_of_market(virtual_date)
                                if monitor:
                                    monitor.end_timer('market_end_of_market')

                                if monitor:
                                    monitor.start_timer('market_index_update_iter')
                                market_index.update_market_index(virtual_date)
                                if monitor:
                                    monitor.end_timer('market_index_update_iter')

                if monitor:
                    monitor.end_timer('order_processing')

                    monitor.start_timer('final_market_match')
                market.match_order(virtual_date)
                market.end_of_market(virtual_date)
                if monitor:
                    monitor.end_timer('final_market_match')

                if monitor:
                    monitor.start_timer('final_market_index_update')
                market_index.update_market_index(virtual_date)
                if monitor:
                    monitor.end_timer('final_market_index_update')


                if monitor:
                    monitor.start_timer('end_of_iteration')
                for each_person in persons:
                    if each_person.person_id >= 0:
                        each_person.end_of_iteration(virtual_date, iter)
                if monitor:
                    monitor.end_timer('end_of_iteration')

                if monitor:
                    monitor.start_timer('save_all')
                save_all(virtual_date, iter, stocks, market_index, persons, market)
                if monitor:
                    monitor.end_timer('save_all')

                if monitor:
                    monitor.end_timer('iteration_processing')

                iter_time = time.time() - iter_start_time
                print(f"Iteration completed: day={virtual_date + 1}, iteration={iter + 1}, elapsed={iter_time:.2f}s")


            if monitor:
                monitor.start_timer('end_of_day')
            market.end_of_day(virtual_date)
            for each_person in persons:
                each_person.end_of_day(virtual_date)
            for each_stock in stocks:
                each_stock.end_of_day(virtual_date)
            market_index.end_of_day(virtual_date)
            if monitor:
                monitor.end_timer('end_of_day')

            if monitor:
                monitor.end_timer('day_processing')

            day_time = time.time() - day_start_time
            logging.info("DAY_DONE virtual_date=%s day=%s/%s elapsed=%.2fs", virtual_date, virtual_date + 1, No_Days, day_time)
            print(f"Day completed: day={virtual_date + 1}, elapsed={day_time:.2f}s")

    except Exception:
        logging.exception("SIM_FAILED")
        raise
    finally:
        executor.shutdown()
        mon = get_monitor()
        if monitor:
            total_time = time.time() - total_start_time
            print(f"Total execution time: {total_time:.2f}s")
            mon.print_summary()
            mon.save_report()
        else:
            mon.save_report()


def overall_test_original():
    """"""
    monitor = get_monitor()

    (
        current_date,
        current_iteration,
        broker,
        market_index,
        market,
        stocks,
        persons,
    ) = init_all(False)

    total_start_time = time.time()

    for virtual_date in range(No_Days):
        if virtual_date == 0:
            broker.ipo(virtual_date)
        market_index.update_market_index(virtual_date)
        generate_gossip(virtual_date, persons, stocks)

        for iter in range(Iterations_Daily):
            total_traders = (Num_Person + Num_agent + Num_tech_traders + Num_virtual_agents) * N
            rand = random.sample(range(0, total_traders), total_traders)
            count = 0

            for i in rand:
                agent = persons[i]
                monitor.start_timer('agent_execution', agent.person_id)
                agent_ops = agent.stock_ops(virtual_date, persons, stocks, market_index, iter)
                monitor.end_timer('agent_execution', agent.person_id)

                if agent_ops and "stocks" in agent_ops:
                    stock_ops_list = agent_ops["stocks"]
                    total_position = agent_ops.get("total_position", 1.0)
                    try:
                        total_position = float(total_position)
                    except (TypeError, ValueError):
                        total_position = 1.0
                    if total_position < 0:
                        total_position = 0.0
                    for j in range(len(stock_ops_list)):
                        op = stock_ops_list[j]
                        persons[i].create_order(op, virtual_date, total_position, iter)
                        count = count + 1
                    if count >= 20:
                        count = 0
                        market.match_order(virtual_date)
                        market.end_of_market(virtual_date)
                        market_index.update_market_index(virtual_date)

            market.match_order(virtual_date)
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

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")
    monitor.print_summary()
    monitor.save_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel trading simulation runner')
    parser.add_argument('--mode', type=str, default='parallel',
                       choices=['original', 'parallel'],
                       help='Execution mode: original or parallel')
    parser.add_argument('--executor', type=str, default='thread',
                       choices=['thread', 'batch'],
                       help='Executor type: thread or batch')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker threads; defaults to CPU count')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size used by batch executor')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable performance monitoring')

    args = parser.parse_args()

    setup_simple_logger()
    reset_monitor()

    if args.mode == 'original':
        print("Using original sequential execution mode")
        overall_test_original()
    else:
        print(f"Using parallel execution mode: {args.executor}")
        print(f"Workers: {args.workers or 'auto'}")
        if args.executor == 'batch':
            print(f"Batch size: {args.batch_size}")

        overall_test_parallel(
            executor_type=args.executor,
            max_workers=args.workers,
            batch_size=args.batch_size,
            enable_monitoring=not args.no_monitoring
        )
