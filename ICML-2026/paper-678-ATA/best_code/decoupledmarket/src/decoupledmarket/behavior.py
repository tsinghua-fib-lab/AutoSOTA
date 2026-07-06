import re
import math
from decoupledmarket.content.our_run_gpt_prompt import (
    run_gpt_prompt_trading_stock,
    analysis,
    pre_reflect,
    long_reflect,
    text_strategy_reward,
    update_strategy,
    run_gpt_generate_gossip,
)
from decoupledmarket.constant import analysis_num, gossip_num_max, Num_Person, Num_agent
import pandas as pd

def normalize_positive_operations(data):

    positives = [item['operation'] for item in data if item['operation'] > 0]
    total = sum(positives)


    if total == 0:
        return data


    normalized_data = []
    for item in data:
        if item['operation'] > 0 and total > 1:
            normalized_value = item['operation'] / total
            normalized_data.append({'stock_id': item['stock_id'], 'operation': round(normalized_value, 4)})
        else:
            normalized_data.append(item)
    return normalized_data

def extract_for_choose_trading(choose_trading):
    try:
        result = {
            "stocks": [],
            "total_position": None
        }


        if "hold" in choose_trading.lower():
            result["stocks"].append({"stock_id": None, "operation": 0.0})
            result["total_position"] = 0.0
            return result


        matches = re.findall(
            r"Operation:\s*([-+]?\d*\.?\d+)\s*,\s*Stock(?: name)?:\s*([A-Z]+)",
            choose_trading,
            re.IGNORECASE
        )


        if not matches:
            matches = re.findall(
                r"([-+]?\d*\.?\d+)\s*,\s*Stock(?: name)?:\s*([A-Z]+)",
                choose_trading,
                re.IGNORECASE
            )


        for op_val, stock_name in matches:
            op_val = float(op_val)
            if -1.0 <= op_val <= 1.0:
                result["stocks"].append({
                    "stock_id": stock_name.upper(),
                    "operation": op_val
                })


        total_match = re.search(
            r"Total\s+position\s+ratio\s*:\s*([-+]?\d*\.?\d+)",
            choose_trading,
            re.IGNORECASE
        )
        if total_match:
            result["total_position"] = float(total_match.group(1))


        if "normalize_positive_operations" in globals():
            result["stocks"] = normalize_positive_operations(result["stocks"])


        if not result["stocks"]:
            return False
        return result

    except Exception:
        return False

def extract_for_choose_buy(choose_buy):
    if "hold" in choose_buy or "Hold" in choose_buy:
        return "hold", 0, 0
    else:
        try:
            match = re.search(
                r"^\s*Operation:\s*buy,\s*Stock name:\s*([A-Z]),\s*Investment Amount:\s*\$?(\d+(\.\d+)?),\s*Best Buying Price:\s*\$?(\d+(\.\d+)?)\s*$",
                choose_buy,
                re.IGNORECASE,
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

def stock_ops(virtual_date, persons, stocks, market_index, iter):
    ops = []
    for p in persons:
        if 0 <= p.person_id < Num_Person:

            analysis_results, analysis_table = analysis(
                virtual_date, p, stocks, market_index, analysis_num, gossip_num_max
            )
            p.analysis = analysis_results


            choose_trading = run_gpt_prompt_trading_stock(
                virtual_date, p, stocks, analysis_results, iteration=iter
            )
            result = extract_for_choose_trading(choose_trading)


            if not result or "stocks" not in result:
                continue

            stock_ops_list = result["stocks"]
            total_position = result.get("total_position", None)


            for trading in stock_ops_list:
                op_val = trading["operation"]

                action = (
                    "buy" if op_val > 0
                    else "sell" if op_val < 0
                    else "hold"
                )


                p.add_memory(
                    virtual_date,
                    iter,
                    trading,
                    action,
                    analysis_table,
                    analysis_results,
                    str(total_position),
                    market_index,
                    stocks,
                )

            ops.append(result)

    return ops

def reflection(virtual_date, persons, stocks, market_index, iter):
    for p in persons:
        if p.person_id > -1:
            if p.reflect_frequency == 0:
                pass
            elif p.reflect_frequency == 2 and virtual_date % 2 == 0:
                analysis_for_reflect = pre_reflect(virtual_date, p)
                w_s = extract_analysis_for_reflect(analysis_for_reflect)
                better_strategy_reward = text_strategy_reward(virtual_date, p)
                #long_w_s=long_reflect(virtual_date, p)

                new_strategy = update_strategy(virtual_date, p, w_s, better_strategy_reward)#, long_w_s)
                new_strategy = extract_strategy(new_strategy)
                p.principle = new_strategy
                p.add_memory(
                    virtual_date,
                    iter,
                    "None", #long_w_s,
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
            if virtual_date < 1 or p.person_id >= Num_Person + Num_agent:
                p.add_gossip(virtual_date, "None")
            else:
                gossip = run_gpt_generate_gossip(virtual_date, p)
                p.add_gossip(virtual_date, gossip)
