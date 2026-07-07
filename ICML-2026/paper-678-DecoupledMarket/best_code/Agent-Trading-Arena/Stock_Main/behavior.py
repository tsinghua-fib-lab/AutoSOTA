import re
import math
from content.our_run_gpt_prompt import (
    run_gpt_prompt_choose_buy_stock,
    run_gpt_prompt_choose_sell_stock,
    analysis,
    pre_reflect,
    long_reflect,
    update_strategy,
    run_gpt_generate_gossip,
)
#from constant import analysis_num, gossip_num_max


def extract_for_choose_buy(choose_buy):
    if "hold" in choose_buy or "Hold" in choose_buy:
        return "hold", 0, 0
    else:
        try:
            match = re.search(
                r"^\s*Operation:\s*buy,\s*Stock name:\s*([A-Z]),\s*Investment Amount:\s*\$?(\d+(\.\d+)?),\s*Best Buying Price:\s*\$?(\d+(\.\d+)?)\s*$",
                choose_buy,
                re.IGNORECASE,#ignorecase
            )
            if match:
                stock_name = match.group(1).upper()
                volume = float(match.group(2))
                price_buy = float(match.group(4))
                if volume == 0 or price_buy == 0:
                    return "hold", 0, 0
                quantity = math.ceil(volume / price_buy)
                return stock_name, quantity, price_buy
        except Exception:
            return False


def extract_for_choose_sell(choose_sell):
    if "hold" in choose_sell or "Hold" in choose_sell:
        return "hold", 0, 0
    else:
        try:
            match = re.search(
                r"^\s*Operation:\s*sell,\s*Stock name:\s*([A-Z]),\s*The number of shares:\s*(\d+),\s*Best Selling Price:\s*\$?(\d+(\.\d+)?)\s*$",
                choose_sell,
                re.IGNORECASE,
            )
            if match:
                stock_name = match.group(1).upper()
                quantity = match.group(2)
                price_sell = float(match.group(3))
                if quantity == 0 or price_sell == 0:
                    return "hold", 0, 0
                return stock_name, quantity, price_sell
        except Exception:
            return False


def extract_analysis_for_reflect(analysis_for_reflect):
    w_s = []
    try:
        match = re.search(
            r"Weakness:\s*(.*?).\s*Strength:\s*(.*?)$", analysis_for_reflect
        )
        if match:
            w_s.append(match.group(1))
            w_s.append(match.group(2))
            return w_s
    except Exception:
        return False


def extract_strategy(new_strategy):
    try:
        match = re.search(r"New investment strategy:\s*(.*?)$", new_strategy)
        if match:
            n_s = match.group(1)
            return n_s
    except Exception:
        return False


def stock_ops(virtual_date, persons, stocks, market_index, iter, args):
    # obtain the stock operations
    ops = []
    for p in persons:
        if p.person_id > -1:
            analysis_results, gossip = analysis(
                virtual_date, p, stocks, market_index, args.analysis_num, args.gossip_num_max
            )
           # print(analysis_results,gossip)
            # p.analysis = analysis_results
            choose_buy = run_gpt_prompt_choose_buy_stock(
                virtual_date, p, stocks, analysis_results
            )
           
            stock_name_buy, quantity, price = extract_for_choose_buy(choose_buy)
            if stock_name_buy == "hold":
                buy_list = ["hold", None, None, None]
                op_memory = "hold"
            else:
                buy_list = [
                    "buy",
                    stock_name_buy,
                    price,
                    quantity,
                ]
            choose_sell = run_gpt_prompt_choose_sell_stock(
                virtual_date, p, stocks, analysis_results
            )
            '''
            "Operation: sell, Stock name: [Stock Name], The number "
            "of shares: [Specific Number of Shares], Best Selling "
            "Price: [Recommended Selling Price]"
            '''
            stock_name_sell, quantity, price = extract_for_choose_sell(choose_sell)
            if stock_name_sell == "hold":
                sell_list = ["hold", None, None, None]
                op_memory = "hold"
            else:
                sell_list = [
                    "sell",
                    stock_name_sell,
                    price,
                    quantity,
                ]
            if stock_name_buy == "hold" and stock_name_sell == "hold":
                p.add_memory(
                    virtual_date,
                    iter,
                    op_memory,
                    "hold",
                    gossip,
                    analysis_results,
                    "None",
                    market_index,
                    stocks,
                )
            if stock_name_buy != "hold":
                op_memory = "buy {} shares of stock {} at ${}".format(
                    buy_list[3], buy_list[1], buy_list[2]
                )
                p.add_memory(
                    virtual_date,
                    iter,
                    op_memory,
                    "buy",
                    gossip,
                    analysis_results,
                    "None",
                    market_index,
                    stocks,
                )
            if stock_name_sell != "hold":
                op_memory = "sell {} shares of stock {} at ${}".format(
                    sell_list[3], sell_list[1], sell_list[2]
                )
                p.add_memory(
                    virtual_date,
                    iter,
                    op_memory,
                    "sell",
                    gossip,
                    analysis_results,
                    "None",
                    market_index,
                    stocks,
                )

            p_list = [buy_list, sell_list]
            ops.append(p_list)
    return ops


def reflection(virtual_date, persons, stocks, market_index, iter):
    for p in persons:
        if p.person_id > -1:
            if p.reflect_frequency == 0:
                pass
            elif (iter + 1) % p.reflect_frequency == 0:
                analysis_for_reflect = pre_reflect(virtual_date, p)
                w_s = extract_analysis_for_reflect(analysis_for_reflect)
                suggestion_for_reflect=long_reflect(virtual_date,p)
                new_strategy = update_strategy(virtual_date, p, w_s,suggestion_for_reflect)
                new_strategy = extract_strategy(new_strategy)
                p.principle = new_strategy
                p.add_memory(
                    virtual_date,
                    iter,
                    "None",
                    "reflect",
                    "None",
                    "None",
                    analysis_for_reflect,
                    market_index,
                    stocks,
                )
            else:
                pass


def generate_gossip(virtual_date, persons, stocks_list):
    # obtain the guidance to update principle
    for p in persons:
        if p.person_id > -1:
            if virtual_date < 1:
                p.add_gossip(virtual_date, "None")
            else:
                gossip = run_gpt_generate_gossip(virtual_date, p)
                p.add_gossip(virtual_date, gossip)
