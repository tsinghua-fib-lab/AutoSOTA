from decoupledmarket.content.our_run_gpt_prompt import (
    run_llm_trading_stock,
)

from decoupledmarket.behavior import extract_for_choose_trading

def tech_model(stocks, virtual_date, agent, iteration=0):
    ops = run_llm_trading_stock(virtual_date, agent, stocks)
    result = extract_for_choose_trading(ops)
    if not result or "stocks" not in result:
        return {"stocks": [{"stock_id": None, "operation": 0.0}], "total_position": 0.0}


    stock_ops_list = result["stocks"]
    total_position = result.get("total_position", None)
    for trading in stock_ops_list:
        if trading['operation']>0:
            agent.add_memory(
            virtual_date,
            iteration,
            trading,
            "buy",
            [],
            [],
            str(total_position),
            [],
            stocks,
        )
        elif trading['operation']<0:
            agent.add_memory(
            virtual_date,
            iteration,
            trading,
            "sell",
            [],
            [],
            str(total_position),
            [],
            stocks,
        )
        elif trading['operation']==0:
            agent.add_memory(
            virtual_date,
            iteration,
            trading,
            "hold",
            [],
            [],
            str(total_position),
            [],
            stocks,
        )
    return result

def llm_model(stocks, virtual_date, agent, iteration=0):
    ops = run_llm_trading_stock(virtual_date, agent, stocks)
    result = extract_for_choose_trading(ops)
    if not result or "stocks" not in result:
        return {"stocks": [{"stock_id": None, "operation": 0.0}], "total_position": 0.0}


    stock_ops_list = result["stocks"]
    total_position = result.get("total_position", None)
    for trading in stock_ops_list:
        if trading['operation']>0:
            agent.add_memory(
            virtual_date,
            iteration,
            trading,
            "buy",
            [],
            [],
            str(total_position),
            [],
            stocks,
        )
        elif trading['operation']<0:
            agent.add_memory(
            virtual_date,
            iteration,
            trading,
            "sell",
            [],
            [],
            str(total_position),
            [],
            stocks,
        )
        elif trading['operation']==0:
            agent.add_memory(
            virtual_date,
            iteration,
            trading,
            "hold",
            [],
            [],
            str(total_position),
            [],
            stocks,
        )
    return result

