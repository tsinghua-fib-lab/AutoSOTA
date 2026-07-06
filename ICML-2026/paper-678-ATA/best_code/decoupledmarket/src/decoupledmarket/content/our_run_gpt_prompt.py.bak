from decoupledmarket.content.gpt_structure import generate_prompt, ChatGPT_safe_generate_response, llm_safe_generate_response
import ast
import re
import random
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from decoupledmarket.constant import STOCK_NAMES, our_agent
from decoupledmarket.arena_content.mc_suggestion import daily_decision_narrative
from decoupledmarket.Stock_Advisor import  analyze_stock

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def _template_path(relative_path):
    return str(PACKAGE_ROOT / relative_path)


def integrate_gossip(virtual_date, persona, gossip_num_max):
    gossip = persona.query_gossip(virtual_date)
    if not gossip:
        return "None"

    gossip_num = np.random.randint(0, gossip_num_max + 1)
    if gossip_num == 0:
        return "None"

    gossip_num = min(gossip_num, len(gossip))
    selected_gossip = random.sample(gossip, gossip_num)

    all_gossip = ""
    for g in selected_gossip:
        all_gossip += "- " + g.get("gossip", "") + "\n"

    return all_gossip.strip()



def integrate_gossip_info(virtual_date, persona):
    memory = persona.query_memory(virtual_date - 1)
    gossip_info = ""
    for m in memory:
        with open(_template_path("content/our_prompt_template/gossip_info.txt"), "r", encoding="utf-8") as f:
            gossip_info_template = f.read()
        prompt_input = [
            m["Virtual_date"],
            m["Iteration"],
            m["Stock_op"],
            m["Financial_situation"],
            m["Stock_prices"],
            m["Analysis_for_stocks"],
            m["Strategy"]]
        for count, input in enumerate(prompt_input):
            gossip_info_template = gossip_info_template.replace(
                f"!<INPUT {count}>!", str(input)
            )
            if (
                "<commentblockmarker>###</commentblockmarker>"
                in gossip_info_template
            ):
                gossip_info_template = gossip_info_template.split(
                    "<commentblockmarker>###</commentblockmarker>"
                )[1]
        gossip_info += gossip_info_template
    return gossip_info

def integrate_technical_info(virtual_date, stocks_list):
    technical_info = ""
    with open(_template_path("arena_content/our_prompt_template/technical_information.txt"), "r", encoding="utf-8") as f:
        base_template = f.read()
    for stock in stocks_list:
        technical_info_template = base_template
        prices = stock.query_prices(virtual_date)

        result = analyze_stock(stock.stock_id, prices, STOCK_NAMES)

        mapping = {
            0: result.get("Stock Name", ""),
            1: result.get("Current Price", ""),
            2: result.get("10-Days EMA", ""),
            3: result.get("MACD", ""),
            4: result.get("Signal", ""),
            5: round(result.get("MACDH", ""), 2),
            6: round(result.get("RSI", ""), 2),
            7: round(result.get("ATR", ""), 2),
            8: round(result.get("Bollinger_Mid", ""), 2),
            9: round(result.get("Bollinger_Upper", ""), 2),
            10: round(result.get("Bollinger_Lower", ""), 2),
            11: result.get("Trend", ""),
            12: result.get("Volatility", ""),
        }
        for idx, val in mapping.items():
            technical_info_template = technical_info_template.replace(f"!<INPUT {idx}>!", str(val))
        if "<commentblockmarker>###</commentblockmarker>" in technical_info_template:
            technical_info_template = technical_info_template.split("<commentblockmarker>###</commentblockmarker>")[1]
        technical_info += technical_info_template
    return technical_info


def intergrate_table_analysis_info(virtual_date, stocks_list):
    advisor_analyses = []
    for stock in stocks_list:
        prices = stock.query_stocks_table_analysis(virtual_date)
        advisor_analyses.append(analyze_stock(stock.stock_id, prices, STOCK_NAMES))
    return advisor_analyses

def integrate_stock_info(virtual_date, stocks_list):
    stock_info = ""
    for stock in stocks_list:
        with open(_template_path("content/our_prompt_template/stock_information.txt"), "r", encoding="utf-8") as f:
            stock_info_template = f.read()
        stock_infos = stock.query_prompt_values(virtual_date)
        for count, (key, value) in enumerate(stock_infos.items()):
            if key == "current_price_change":
                value = "+{:.2f}".format(value) if value >= 0 else "-{:.2f}".format(value)
            stock_info_template = stock_info_template.replace(
                f"!<INPUT {count}>!", str(value)
            )
            if (
                "<commentblockmarker>###</commentblockmarker>"
                in stock_info_template
            ):
                stock_info_template = stock_info_template.split(
                    "<commentblockmarker>###</commentblockmarker>"
                )[1]
        stock_info += stock_info_template
    return stock_info


def integrate_hold_info(virtual_date, persona):
    hold_list = persona.query_prompt(virtual_date)
    if len(hold_list) == 0:
        total_value = 0
        hold_info = "you do not hold any stock right now."
    else:
        hold_info = "you are holding the following stocks:"
        total_value = 0
        for hold in hold_list:
            with open(_template_path("content/our_prompt_template/hold_information.txt"), "r", encoding="utf-8") as f:
                hold_info_template = f.read()
            total_value += hold["total_value"]
            for count, (key, value) in enumerate(hold.items()):
                if key == "captital_gain":
                    value = "{:.2f}% PROFIT".format(value) if value >= 1e-12 else "{:.2f}% LOSS".format(abs(value))
                elif key == "Current_price_change":
                    value = "+{:.2f}".format(value) if value >= 0 else "-{:.2f}".format(value)
                hold_info_template = hold_info_template.replace(
                    f"!<INPUT {count}>!", str(value)
                )
                if (
                    "<commentblockmarker>###</commentblockmarker>"
                    in hold_info_template
                ):
                    hold_info_template = hold_info_template.split(
                        "<commentblockmarker>###</commentblockmarker>"
                    )[1]
            hold_info += hold_info_template
    begin = "Your total portfolio balance is ${:.2f}, ".format(total_value)
    hold_info = begin + hold_info
    return hold_info

def text_strategy_reward(virtual_date, p):
    candidate_paths = [
        os.getenv("STRATEGY_XLSX_PATH"),
        "./strategy.xlsx",
        "./save/sim01/strategy.xlsx",
        r"C:\Users\mtm\Desktop\strategy.xlsx",
    ]
    file_path = next((path for path in candidate_paths if path and os.path.exists(path)), None)
    if file_path is None:
        return ""

    df = pd.read_excel(file_path)
    prompt = ""
    for index, row in df.iterrows():
        prompt += f"{index + 1}. The investment strategy: {row['strategy']}\n   Reward: {row['reward']}\n\n"
    return prompt



def integrate_long_reflect_info(virtual_date, persona):
    iteration=0
    pre_reflect_info = ""
    while virtual_date>=0 and iteration<3:
        memory = persona.query_memory(virtual_date)
        if not memory:
            virtual_date -= 1
            iteration += 1
            continue
        for m in [memory[-1]]:
            with open(_template_path("content/our_prompt_template/reflect_info.txt"), "r", encoding="utf-8") as f:
                pre_reflect_info_template = f.read()
            prompt_input = [
                m["Virtual_date"],
                m["Iteration"],
                m["Stock_op"],
                m["Financial_situation"],
                m["Market_change"],
                m["Stock_prices"],
                m["Gossip"],
                m["Strategy"]]
            for count, input in enumerate(prompt_input):
                pre_reflect_info_template = pre_reflect_info_template.replace(
                    f"!<INPUT {count}>!", str(input)
                )
                if (
                    "<commentblockmarker>###</commentblockmarker>"
                    in pre_reflect_info_template
                ):
                    pre_reflect_info_template = pre_reflect_info_template.split(
                        "<commentblockmarker>###</commentblockmarker>"
                    )[1]
            pre_reflect_info += pre_reflect_info_template
        virtual_date-=1
        iteration+=1
    return pre_reflect_info
def integrate_reflect_info(virtual_date, persona):
    memory = persona.query_memory(virtual_date)
    pre_reflect_info = ""
    for m in memory:
        with open(_template_path("content/our_prompt_template/reflect_info.txt"), "r", encoding="utf-8") as f:
            pre_reflect_info_template = f.read()
        prompt_input = [
            m["Virtual_date"],
            m["Iteration"],
            m["Stock_op"],
            m["Financial_situation"],
            m["Market_change"],
            m["Stock_prices"],
            m["Gossip"],
            m["Strategy"]]
        for count, input in enumerate(prompt_input):
            pre_reflect_info_template = pre_reflect_info_template.replace(
                f"!<INPUT {count}>!", str(input)
            )
            if (
                "<commentblockmarker>###</commentblockmarker>"
                in pre_reflect_info_template
            ):
                pre_reflect_info_template = pre_reflect_info_template.split(
                    "<commentblockmarker>###</commentblockmarker>"
                )[1]
        pre_reflect_info += pre_reflect_info_template
    return pre_reflect_info
def integrate_MC_info(virtual_date, stocks_list):
    MC_info = ""
    for stock in stocks_list:
        stock_info = stock.query_prices(virtual_date)
        results = daily_decision_narrative(
                    stock_info,
                    stock_id = STOCK_NAMES[int(stock.stock_id)],
                    window_size = 20,
                    horizon = 10,
                    top_k = 5,
                    n_rollouts = 5,
                    gamma = 0.99,
                    use_returns=True,
                )
        MC_info = MC_info + "\n" + results
    return MC_info


def update_strategy(virtual_date, persona, w_s, better_strategy_reward, iteration=1):#, long_w_s, long_w_s
    def create_prompt_input(virtual_date, persona, w_s, better_strategy_reward, iteration):
        reflect_info = integrate_reflect_info(virtual_date, persona)
        prompt_input = [
            reflect_info,
            w_s[0],
            w_s[1],
            better_strategy_reward
        ]
        return prompt_input

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:

            try:
                data = json.loads(gpt_response)
                output = data.get("output", "").strip()
            except json.JSONDecodeError:

                output = gpt_response.strip()


            if not output:
                return False



            if re.search(r"New investment strategy:\s*", output, re.IGNORECASE):
                return True


            keywords = ["diversify", "portfolio", "investment", "strategy", "risk", "growth", "return"]
            if any(word in output.lower() for word in keywords):
                return True

            return False
        except Exception:
            return False

    prompt_template = _template_path("content/our_prompt_template/reflect.txt")
    prompt_input = create_prompt_input(virtual_date, persona, w_s, better_strategy_reward, iteration)
    prompt = generate_prompt(prompt_input, prompt_template)

    example_output = (
        "New investment strategy: [New investment strategy]"
    )
    special_instruction = ""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )
    if output is not False:
        #with open("relect_result.txt", "w") as file:
         #   file.write(output)
        print(output)
        return output

def long_reflect(virtual_date, persona, iteration=1):
    def create_prompt_input(virtual_date, persona):
        long_reflect_info = integrate_long_reflect_info(virtual_date, persona)
        prompt_input = [
            long_reflect_info
        ]
        return prompt_input

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            return True
        except Exception:
            return False

    prompt_template = _template_path("content/our_prompt_template/pre_long_reflection.txt")
    prompt_input = create_prompt_input(virtual_date, persona)
    prompt = generate_prompt(prompt_input, prompt_template)

    example_output = (
        "Weakness: [Weakness]. Strength: [Strength]"
    )
    special_instruction = ""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )
    if output is not False:
      #  with open("long_pre_reflect_suggestion_result.txt", "w") as file:
      #      file.write(output)
        # print(output)
        return output

def pre_reflect(virtual_date, persona, iteration=1):
    def create_prompt_input(virtual_date, persona):
        pre_reflect_info = integrate_reflect_info(virtual_date, persona)
        prompt_input = [
            pre_reflect_info
        ]
        return prompt_input

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            match = re.search(r"Weakness:\s*(.*?).\s*Strength:\s*(.*?)$", gpt_response)
            if match:
                return True
            else:
                return False
        except Exception:
            return False

    prompt_template = _template_path("content/our_prompt_template/pre_reflect.txt")
    prompt_input = create_prompt_input(virtual_date, persona)
    prompt = generate_prompt(prompt_input, prompt_template)

    example_output = (
        "Weakness: [Weakness]. Strength: [Strength]"
    )
    special_instruction = ""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )
    if output is not False:
        #with open("pre_reflect_result.txt", "w") as file:
          #  file.write(output)
        # print(output)
        return output


def run_gpt_generate_gossip(virtual_date, persona, iteration=1):
    def create_prompt_input(virtual_date, persona):
        gossip_input = integrate_gossip_info(virtual_date, persona)
        prompt_input = [
            gossip_input
        ]
        return prompt_input

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            return True
        except Exception:
            return False

    prompt_template = _template_path("content/our_prompt_template/gossip.txt")
    prompt_input = create_prompt_input(virtual_date, persona)
    prompt = generate_prompt(prompt_input, prompt_template)

    example_output = (
    )
    special_instruction = ""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )
    if output is not False:
        # print(output)
        return output

def analysis_table(virtual_date, stocks_list, persona, iteration=1):
    def create_prompt_input(virtual_date, stocks_list):
        Stock_Table =  intergrate_table_analysis_info(virtual_date, stocks_list)
        prompt_input = [
            Stock_Table
        ]
        return prompt_input

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            return True
        except Exception:
            return False
    prompt_template = _template_path("content/our_prompt_template/analysis_Table.txt")
    prompt_input = create_prompt_input(virtual_date, stocks_list)
    prompt = generate_prompt(prompt_input, prompt_template)

    example_output = (
    )
    special_instruction = ""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )
    if output is not False:
        return output

def technical_analysis(virtual_date, persona, stocks_list, iteration=1):
    def create_prompt_input(virtual_date, stocks_list):
        technical_info = integrate_technical_info(virtual_date, stocks_list)
        prompt_input = [
            technical_info
        ]
        return prompt_input

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            return True
        except Exception:
            return False

    if (persona.person_id % 100) in our_agent:
        prompt_template = _template_path("arena_content/our_prompt_template/technical_analysis.txt")
        prompt_input = create_prompt_input(virtual_date, stocks_list)
    else:
        prompt_template = _template_path("arena_content/our_prompt_template/technical_analysis.txt")
        prompt_input = create_prompt_input(virtual_date, stocks_list)
    prompt = generate_prompt(prompt_input, prompt_template)

    example_output = (
        ""
    )
    special_instruction = """"""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )

    if output is not False:
        return output


def analysis(virtual_date, persona, stocks_list, market_index, analysis_num, gossip_num_max, iteration=1):
    def create_prompt_input(virtual_date, persona, stocks_list, market_index, analysis_num, gossip_num_max):
        gossip = integrate_gossip(virtual_date, persona, gossip_num_max)
        market_index = "Current market index change: {:.2f}%".format(
            market_index.query_market_index_intraday_percentage(virtual_date) * 100
        )
        stock_info = integrate_stock_info(virtual_date, stocks_list)
        hold_info = integrate_hold_info(virtual_date, persona)
        prompt_input = [
            stock_info,
            market_index,
            gossip,
            hold_info,
            persona.principle,
            analysis_num
        ]
        return prompt_input, gossip

    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.replace("The analysis results: \n", "")
        gpt_response = gpt_response.replace("The analysis results:\n", "")
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            return True
        except Exception:
            return False

    if (persona.person_id % 100) in our_agent:#C:\Users\mtm\Documents\a_Jupyter\add_data_analysis_arena_20250801\Stock_Main_gpt_agent\arena_content\our_prompt_template\analysis_advisor.txt
        prompt_template = _template_path("arena_content/our_prompt_template/analysis_advisor.txt")
    else:
        prompt_template = _template_path("content/our_prompt_template/analysis.txt")
    prompt_input, gossip = create_prompt_input(virtual_date, persona, stocks_list, market_index, analysis_num, gossip_num_max)
    prompt = generate_prompt(prompt_input, prompt_template)


    #with open("analysis.txt", "w") as file:
     #  file.write(prompt)

    example_output = (
        "The analysis results: \n[analysis results]"
    )
    special_instruction = """Each analysis result should be started with "-", and ended with line break."""
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )


    if output is not False:
        # print(output)
        return output, gossip

def run_gpt_prompt_trading_stock(virtual_date, persona, stocks_list, analysis_results, iteration=1):
    def create_prompt_input(virtual_date, persona, stocks_list, analysis_results):
        stock_info = integrate_stock_info(virtual_date, stocks_list)
        #MC_suggestion = integrate_MC_info(virtual_date, stocks_list)
        hold_info = integrate_hold_info(virtual_date, persona)
        prompt_input = [
            persona.cash,
            persona.minimum_living_expense*10,
            stock_info,
            hold_info,
            analysis_results,
            #MC_suggestion,# persona.identity["minimum_living_expense"] * 10,
            persona.principle,
        ]
        return prompt_input

    def create_our_prompt_input(virtual_date, persona, stocks_list, analysis_results):
        stock_info = integrate_stock_info(virtual_date, stocks_list)
        tech_analysis = technical_analysis(virtual_date, persona, stocks_list)
        hold_info = integrate_hold_info(virtual_date, persona)
        prompt_input = [
            persona.cash,
            persona.minimum_living_expense*10,
            stock_info,
            hold_info,
            tech_analysis,
            analysis_results,
            persona.principle,
        ]
        return prompt_input


    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip("[]").split("], [")
        return gpt_response


    def __chat_func_validate(gpt_response, prompt=""):
        """Docstring."""
        try:
            results = {"stocks": [], "total_position": None}


            if "hold" in gpt_response.lower():
                results["stocks"].append({"stock_id": None, "operation": 0.0})
                results["total_position"] = 0.0
                return results


            matches = re.findall(
                r"Operation:\s*([-+]?\d*\.?\d+)\s*,\s*Stock(?: name)?:\s*([A-Z]+)",
                gpt_response,
                re.IGNORECASE
            )


            if not matches:
                matches = re.findall(
                    r"([-+]?\d*\.?\d+)\s*,\s*Stock(?: name)?:\s*([A-Z]+)",
                    gpt_response,
                    re.IGNORECASE
                )

            for op_val, stock_name in matches:
                op_val = float(op_val)
                if -1.0 <= op_val <= 1.0:
                    results["stocks"].append({
                        "stock_id": stock_name.upper(),
                        "operation": op_val
                    })


            total_match = re.search(
                r"Total\s+position\s+ratio\s*:\s*([-+]?\d*\.?\d+)",
                gpt_response,
                re.IGNORECASE
            )
            if total_match:
                results["total_position"] = float(total_match.group(1))


            if results["stocks"]:
                return results
            else:
                return False

        except Exception:
            return False



 #   if persona.cash < persona.identity["minimum_living_expense"] * 10:
    #    return "Operation: hold"

    #if persona.person_id in [0,1]:
     #   prompt_template = "./content/our_prompt_template/trading_based_on_analysis_mc.txt"
     #   prompt_input = create_prompt_input_mc(virtual_date, persona, stocks_list, analysis_results)
    #else:
    if (persona.person_id % 100) in our_agent:
        prompt_template = _template_path("arena_content/our_prompt_template/trading_based_on_analysis.txt")
        prompt_input = create_our_prompt_input(virtual_date, persona, stocks_list, analysis_results)
    else:
        prompt_template = _template_path("content/our_prompt_template/trading_based_on_analysis.txt")
        prompt_input = create_prompt_input(virtual_date, persona, stocks_list, analysis_results)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = (
    "Operation: [Signal in -1.0~1.0], "
    "Stock name: [Stock Name];"
    "Operation: [Signal in -1.0~1.0], "
    "Stock name: [Stock Name];"
    "Operation: [Signal in -1.0~1.0], "
    "Stock name: [Stock Name];"
    "Total position ratio: [0.0~1.0 of total funds]"
    )

    special_instruction = "\n-1.0 means strong Sell recommendation; 0.0 means Hold / no action; +1.0 means strong Buy recommendation; Intermediate values (e.g., -0.3, 0.5) indicate weaker sell/buy signals."
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )

    if output is not False:
        return output[0]

def run_llm_trading_stock(virtual_date, persona, stocks_list, iteration=1):
    def create_prompt_input(virtual_date, persona, stocks_list):
        gossip = integrate_gossip(virtual_date, persona, 3)
        stock_info = integrate_stock_info(virtual_date, stocks_list)
        hold_info = integrate_hold_info(virtual_date, persona)
        prompt_input = [
            persona.cash,
            persona.minimum_living_expense*10,
            stock_info,
            hold_info,
            gossip        ]
        return prompt_input


    def get_fail_safe():
        return "error"

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip("[]").split("], [")
        return gpt_response

    import re

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            results = {"stocks": [], "total_position": None}


            if "hold" in gpt_response.lower():
                results["stocks"].append({"stock_id": None, "operation": 0.0})
                results["total_position"] = 0.0
                return results


            matches = re.findall(
                r"Operation:\s*([-+]?\d*\.?\d+)\s*,\s*Stock(?: name)?:\s*([A-Z]+)",
                gpt_response,
                re.IGNORECASE
            )


            if not matches:
                matches = re.findall(
                    r"([-+]?\d*\.?\d+)\s*,\s*Stock(?: name)?:\s*([A-Z]+)",
                    gpt_response,
                    re.IGNORECASE
                )

            for op_val, stock_name in matches:
                op_val = float(op_val)
                if -1.0 <= op_val <= 1.0:
                    results["stocks"].append({
                        "stock_id": stock_name.upper(),
                        "operation": op_val
                    })


            total_match = re.search(
                r"Total\s+position\s+ratio\s*:\s*([-+]?\d*\.?\d+)",
                gpt_response,
                re.IGNORECASE
            )
            if total_match:
                results["total_position"] = float(total_match.group(1))


            if results["stocks"]:
                return results
            else:
                return False

        except Exception:
            return False

    prompt_template = _template_path("content/our_prompt_template/agent_trading_based_on_analysis.txt")##prompt_template = _template_path("content/our_prompt_template/agent_trading_based_on_analysis.txt")
    prompt_input = create_prompt_input(virtual_date, persona, stocks_list)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = (
    "Operation: [Signal in -1.0~1.0], "
    "Stock name: [Stock Name]; "
    "Operation: [Signal in -1.0~1.0], "
    "Stock name: [Stock Name]; "
    "Operation: [Signal in -1.0~1.0], "
    "Stock name: [Stock Name];"
    "Total position ratio: [0.0~1.0 of total funds]"
    )

    special_instruction = "\n-1.0 means strong Sell recommendation; 0.0 means Hold / no action; +1.0 means strong Buy recommendation; Intermediate values (e.g., -0.3, 0.5) indicate weaker sell/buy signals."
    fail_safe = get_fail_safe()
    output = llm_safe_generate_response(
        persona,
        prompt,
        example_output,
        special_instruction,
        100,
        fail_safe,
        __chat_func_validate,
        __chat_func_clean_up,
        True,
        virtual_date=virtual_date,
        iteration=iteration,
    )

    if output is not False:
        return output[0]

# if __name__ == "__main__":
    # run_gpt_prompt_activity_choose("Mo")
    # run_gpt_prompt_stock_operations("Mo")
    # run_gpt_prompt_secret_news("Mo")
    # generate_focal_points("Mo")
    # run_reflect("Mo")
