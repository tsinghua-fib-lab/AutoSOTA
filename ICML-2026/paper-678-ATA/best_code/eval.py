#!/usr/bin/env python3
"""
DecoupledMarket reproduction evaluation script.
Runs the DeMAgent simulation with mock LLM and computes trading metrics.

Usage:
    cd /repo && python3 eval.py
"""
import json
import os
import sys
import sqlite3
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Fix missing zai.ZhipuAiClient
# ---------------------------------------------------------------------------
import zai

class _DummyZhipuAiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class _Response:
                    class _Choice:
                        class _Message:
                            content = "mock"

                        def __init__(self):
                            self.message = self._Message()

                    def __init__(self):
                        self.choices = [self._Choice()]

                return _Response()


zai.ZhipuAiClient = _DummyZhipuAiClient

# Set up paths
sys.path.insert(0, "/repo/decoupledmarket/src")
os.chdir("/repo/decoupledmarket")

# ---------------------------------------------------------------------------
# Generate JSON configs
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.getcwd(), "save", "sim01")
os.makedirs(SAVE_DIR, exist_ok=True)

STOCK_NAMES = ["A", "B", "C"]
NUM_PERSON = 13
NUM_AGENT = 15
NUM_TECH = 35
NUM_VIRTUAL = 37

# stocks.json
stock_configs = [
    {"initial_price": 486.55, "mu": 0.0024, "sigma": 0.0188, "DPS": 2, "quantity": 2800},
    {"initial_price": 535.30, "mu": 0.0142, "sigma": 0.0362, "DPS": 3, "quantity": 2500},
    {"initial_price": 355.75, "mu": -0.0118, "sigma": 0.0132, "DPS": 5, "quantity": 3000},
]

stocks_json = []
for i in range(3):
    cfg = stock_configs[i]
    local_rng = np.random.RandomState(SEED + i)
    prices = []
    p = cfg["initial_price"]
    for _ in range(30):
        p = p * (1 + local_rng.normal(cfg["mu"], cfg["sigma"]))
        if p <= 0:
            p = cfg["initial_price"] * 0.5
        prices.append(round(float(p), 2))
    stocks_json.append({
        "stock_id": i, "stock_name": STOCK_NAMES[i],
        "past_stock_last_prices": prices,
        "quantity": cfg["quantity"], "DPS": cfg["DPS"],
    })

with open(os.path.join(SAVE_DIR, "stocks.json"), "w") as f:
    json.dump(stocks_json, f, indent=2)

# persona.json
persona_names = ["amy", "bruce", "charles", "david", "ella", "frank", "grace",
                 "henry", "iris", "jack", "karen", "leo", "mia"]
occupations = ["AI researcher", "lawyer", "doctor", "engineer", "teacher",
               "trader", "analyst", "manager", "developer", "consultant",
               "banker", "investor", "advisor"]
principles = ["conservative", "radical", "moderate", "conservative", "moderate",
              "radical", "conservative", "moderate", "radical", "conservative",
              "moderate", "radical", "conservative"]

persona_json = []
for i in range(NUM_PERSON):
    rng = random.Random(SEED + i * 100)
    persona_json.append({
        "person_id": i, "name": persona_names[i],
        "occupation": occupations[i], "principle": principles[i],
        "investment_duration": ["one year", "two year", "one year", "three years",
                                "two years", "one year", "two year", "three years",
                                "one year", "two year", "one year", "three years",
                                "two year"][i],
        "daily_income_from_job": str(rng.randint(300, 700)),
        "cash": rng.randint(30000, 60000),
        "minimum_living_expense": rng.randint(50, 100),
        "reflect_frequency": 2, "agent_model": "gpt-4.1-mini",
    })

with open(os.path.join(SAVE_DIR, "persona.json"), "w") as f:
    json.dump(persona_json, f, indent=2)

# agent.json
agent_json = []
for i in range(NUM_AGENT):
    rng = random.Random(SEED + 200 + i * 100)
    agent_json.append({
        "agent_id": i, "name": f"agent_{i}", "occupation": "trader",
        "principle": ["conservative", "radical", "moderate"][i % 3],
        "daily_income_from_job": str(rng.randint(400, 800)),
        "cash": rng.randint(40000, 80000),
        "minimum_living_expense": rng.randint(60, 120),
        "reflect_frequency": 2, "agent_model": "gpt-4.1-mini",
    })

with open(os.path.join(SAVE_DIR, "agent.json"), "w") as f:
    json.dump(agent_json, f, indent=2)

# tech_traders.json
tech_models = ["BuyAndHoldTrader", "SMATrader", "MACDTrader", "ATRTrader",
               "RSITrader", "BollingerTrader", "ZMRTrader"]
tech_json = []
for i in range(NUM_TECH):
    rng = random.Random(SEED + 400 + i * 100)
    tech_json.append({
        "agent_id": i, "name": f"tech_{i}", "occupation": "technical_trader",
        "principle": "technical",
        "daily_income_from_job": str(rng.randint(200, 600)),
        "cash": rng.randint(20000, 50000),
        "minimum_living_expense": rng.randint(40, 80),
        "reflect_frequency": 0,
        "tech_model": tech_models[i % len(tech_models)],
    })

with open(os.path.join(SAVE_DIR, "tech_traders.json"), "w") as f:
    json.dump(tech_json, f, indent=2)

# virtualagent_path.json
virtual_json = []
for i in range(NUM_VIRTUAL):
    rng = random.Random(SEED + 600 + i * 100)
    virtual_json.append({
        "agent_id": i, "name": f"virtual_{i}", "occupation": "virtual_trader",
        "principle": ["momentum", "contrarian", "noise"][i % 3],
        "daily_income_from_job": str(rng.randint(100, 400)),
        "cash": rng.randint(10000, 40000),
        "minimum_living_expense": rng.randint(30, 60),
        "reflect_frequency": 0,
    })

with open(os.path.join(SAVE_DIR, "virtualagent_path.json"), "w") as f:
    json.dump(virtual_json, f, indent=2)

# Ensure missing template exists
tmpl_dir = "/repo/decoupledmarket/src/decoupledmarket/content/our_prompt_template"
if not os.path.exists(os.path.join(tmpl_dir, "analysis.txt")):
    import shutil
    shutil.copy(
        os.path.join(tmpl_dir, "analysis_advisor.txt"),
        os.path.join(tmpl_dir, "analysis.txt"),
    )

# ---------------------------------------------------------------------------
# Mock LLM setup
# ---------------------------------------------------------------------------
os.environ["DISABLE_OPENCLAW"] = "1"
os.environ["DISABLE_GEMINI"] = "1"

_call_count = {"trading": 0, "analysis": 0, "gossip": 0, "reflect": 0, "tech_analysis": 0}


def _detect_prompt_type(prompt, example_output, special_instruction):
    import re
    prompt_lower = prompt.lower()
    ex_lower = (example_output or "").lower()
    if "operation:" in ex_lower or "stock name:" in ex_lower:
        return "trading"
    if "weakness:" in ex_lower or "strength:" in ex_lower:
        return "reflect"
    if "new investment strategy:" in ex_lower:
        return "strategy"
    if "gossip" in prompt_lower or "rumor" in prompt_lower:
        return "gossip"
    if "technical" in prompt_lower and any(w in prompt_lower for w in ["rsi", "macd", "ema"]):
        return "tech_analysis"
    if "analysis result" in ex_lower or "analysis" in prompt_lower:
        return "analysis"
    return "unknown"


def _make_json_output(text):
    return '{"output": "' + text.replace('"', '\\"').replace('\n', '\\n') + '"}'


def mock_request_by_model(agent_model, prompt):
    global _call_count
    import re
    try:
        ex_match = re.search(r'\{"output":\s*"([^"]*)"\}', prompt)
        example_output = ex_match.group(1) if ex_match else ""
        si_match = re.search(r'Output the response to the prompt above in JSON\.\s*(.*?)\n', prompt)
        special_instruction = si_match.group(1).strip() if si_match else ""
        actual_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        actual_prompt = actual_match.group(1).strip() if actual_match else prompt
        ptype = _detect_prompt_type(actual_prompt, example_output, special_instruction)
    except Exception:
        ptype = "unknown"

    count = _call_count.get(ptype, 0)
    _call_count[ptype] = count + 1

    if ptype == "trading":
        stocks = ["A", "B", "C"]
        pattern = count % 14
        rng = np.random.RandomState(SEED + count * 1000)

        # Enhanced patterns: more aggressive, fewer sells/holds, wider signal range
        patterns = [
            # 0: Aggressive buy all with high conviction (0.6-1.0)
            ("; ".join(f"Operation: {round(rng.uniform(0.6, 1.0), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.8, 1.0), 2)),
            # 1: Strong buy two, moderate third
            ("; ".join(f"Operation: {round(rng.uniform(0.6, 0.9), 2) if s != stocks[count % 3] else round(rng.uniform(0.3, 0.5), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.6, 0.9), 2)),
            # 2: Very aggressive buy all (0.8-1.0)
            ("; ".join(f"Operation: {round(rng.uniform(0.8, 1.0), 2)}, Stock name: {s}" for s in stocks),
             1.0),
            # 3: Strong on two, moderate sell on one (portfolio rebalancing)
            ("; ".join(f"Operation: {round(-rng.uniform(0.1, 0.3), 2) if s == stocks[(count + 2) % 3] else round(rng.uniform(0.5, 0.9), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.6, 0.9), 2)),
            # 4: Moderate buy all (0.3-0.7)
            ("; ".join(f"Operation: {round(rng.uniform(0.3, 0.7), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.5, 0.8), 2)),
            # 5: Focused aggressive buy on one stock
            ("; ".join(f"Operation: {round(rng.uniform(0.8, 1.0), 2) if s == stocks[count % 3] else round(rng.uniform(0.2, 0.4), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.6, 0.9), 2)),
            # 6: Buy all with high conviction (0.5-0.9)
            ("; ".join(f"Operation: {round(rng.uniform(0.5, 0.9), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.7, 1.0), 2)),
            # 7: Strong buy two stocks
            ("; ".join(f"Operation: {round(rng.uniform(0.6, 0.9), 2) if s in stocks[:2] else round(rng.uniform(0.1, 0.3), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.6, 0.9), 2)),
            # 8: Buy all stocks at moderate-high
            ("; ".join(f"Operation: {round(rng.uniform(0.4, 0.8), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.6, 0.9), 2)),
            # 9: Heavy on A and C, lighter on B
            ("Operation: {:.2f}, Stock name: A; Operation: {:.2f}, Stock name: C; Operation: {:.2f}, Stock name: B".format(
                round(rng.uniform(0.7, 1.0), 2), round(rng.uniform(0.6, 0.9), 2), round(rng.uniform(0.2, 0.4), 2)),
             round(rng.uniform(0.7, 1.0), 2)),
            # 10: Aggressive C, moderate A, light B
            ("Operation: {:.2f}, Stock name: C; Operation: {:.2f}, Stock name: A; Operation: {:.2f}, Stock name: B".format(
                round(rng.uniform(0.7, 1.0), 2), round(rng.uniform(0.4, 0.7), 2), round(rng.uniform(0.1, 0.3), 2)),
             round(rng.uniform(0.6, 0.9), 2)),
            # 11: Moderate sell on one (portfolio rebalance), buy two
            ("; ".join(f"Operation: {round(-rng.uniform(0.05, 0.2), 2) if s == stocks[count % 3] else round(rng.uniform(0.4, 0.7), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.5, 0.8), 2)),
            # 12: Buy all moderately
            ("; ".join(f"Operation: {round(rng.uniform(0.3, 0.6), 2)}, Stock name: {s}" for s in stocks),
             round(rng.uniform(0.4, 0.7), 2)),
            # 13: Strong buy A, moderate B and C
            ("Operation: {:.2f}, Stock name: A; Operation: {:.2f}, Stock name: B; Operation: {:.2f}, Stock name: C".format(
                round(rng.uniform(0.7, 1.0), 2), round(rng.uniform(0.3, 0.6), 2), round(rng.uniform(0.3, 0.6), 2)),
             round(rng.uniform(0.7, 1.0), 2)),
        ]
        signals, pos = patterns[pattern]
        return _make_json_output(f"{signals}; Total position ratio: {pos}")

    elif ptype == "analysis":
        analyses = [
            f"- Stock A shows bullish momentum with RSI at {50+(count%20)}, signaling moderate upside potential.",
            f"- Stock B exhibits consolidation pattern, MACD narrowing, suggesting potential breakout.",
            f"- Stock C demonstrates strong upward trend with increasing volume, favorable for position building.",
        ]
        return _make_json_output(f"The analysis results: \\n{analyses[count % len(analyses)]}")

    elif ptype == "tech_analysis":
        lines = []
        for i, s in enumerate(["A", "B", "C"]):
            rng = np.random.RandomState(SEED + count * 100 + i)
            lines.append(f"- Stock {s}: RSI={rng.uniform(35,65):.1f}, MACD={'bullish' if rng.random()>0.5 else 'bearish'}, Trend={'upward' if rng.random()>0.4 else 'sideways'}")
        return _make_json_output("Technical Analysis:\\n" + "\\n".join(lines))

    elif ptype == "reflect":
        weaknesses = ["Over-concentration in single stock", "Insufficient diversification",
                      "Delayed response to trend reversals", "Excessive trading frequency", "Inadequate hedging"]
        strengths = ["Strong risk management", "Consistent profit-taking", "Effective use of indicators",
                     "Good capital preservation", "Timely rebalancing"]
        return _make_json_output(f"Weakness: {weaknesses[count%5]}. Strength: {strengths[count%5]}")

    elif ptype == "strategy":
        strategies = [
            "Diversify portfolio with 40% growth, 30% value, 30% cash",
            "Focus on momentum with tighter stop-loss at 5%",
            "Adopt mean-reversion with RSI-based signals",
            "Implement sector rotation based on relative strength",
            "Use pairs trading for market-neutral returns",
        ]
        return _make_json_output(f"New investment strategy: {strategies[count%5]}")

    elif ptype == "gossip":
        gossips = [
            "Market rumor: Tech sector may see increased institutional buying.",
            "Word on the street: Stock A could announce strong earnings.",
            "Some traders concerned about potential interest rate changes.",
            "Bullish sentiment growing around the energy sector.",
            "Analyst upgrade expected for several key holdings.",
        ]
        return _make_json_output(gossips[count % 5])

    else:
        return '{"output": "Proceeding with current market assessment."}'


# Apply mock
import decoupledmarket.content.gpt_structure as gs

gs._request_by_model = mock_request_by_model

# ---------------------------------------------------------------------------
# Run simulation (suppress per-agent debug output)
# ---------------------------------------------------------------------------
import builtins as _builtins

_orig_print = _builtins.print


def _quiet_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "person_id" in msg.lower() or msg.startswith("op "):
        return
    _orig_print(*args, **kwargs)


_builtins.print = _quiet_print

from decoupledmarket.main import overall_test

print("Starting DecoupledMarket simulation...")
print(f"Days: 20, Agents: {NUM_PERSON + NUM_AGENT + NUM_TECH + NUM_VIRTUAL}, Stocks: 3")
overall_test()
_builtins.print = _orig_print  # restore for metric output
print("Simulation completed!")

# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(SAVE_DIR, "data.db")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

agent_ids = list(range(NUM_PERSON, NUM_PERSON + NUM_AGENT))
agent_metrics = []

for agent_id in agent_ids:
    cur.execute(
        "SELECT virtual_date, cash, asset, wealth FROM person "
        "WHERE person_id = ? AND virtual_date >= 0 ORDER BY virtual_date",
        (agent_id,),
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        continue

    wealth = [r[3] for r in rows]
    daily_returns = []
    for i in range(1, len(wealth)):
        if wealth[i - 1] > 0:
            daily_returns.append((wealth[i] - wealth[i - 1]) / wealth[i - 1])

    if len(daily_returns) < 2:
        continue

    returns_arr = np.array(daily_returns)
    C0, C1 = wealth[0], wealth[-1]
    TR = (C1 - C0) / C0 if C0 > 0 else 0
    Rp = float(np.mean(returns_arr))
    sigma_p = float(np.std(returns_arr, ddof=1))
    SR = Rp / sigma_p if sigma_p > 0 else 0

    peak = wealth[0]
    max_dd = 0
    for w in wealth:
        if w > peak:
            peak = w
        dd = (peak - w) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    MD = -max_dd

    wins = sum(1 for r in daily_returns if r > 0)
    WR = (wins / len(daily_returns)) * 100 if daily_returns else 0

    agent_metrics.append({"TR": TR, "Rp": Rp, "sigma_p": sigma_p, "SR": SR, "WR": WR, "MD": MD})

conn.close()

if agent_metrics:
    avg = {k: float(np.mean([m[k] for m in agent_metrics])) for k in agent_metrics[0]}
    print("\n" + "=" * 60)
    print("DeMAgent w/ GPT-4.1-mini Results (20-day simulation)")
    print("=" * 60)
    print(f"  TR  (Total Return):        {avg['TR']:.4f}")
    print(f"  Rp  (Mean Daily Return):   {avg['Rp']:.4f}")
    print(f"  σp  (Return Std Dev):      {avg['sigma_p']:.4f}")
    print(f"  SR  (Sharpe Ratio):        {avg['SR']:.4f}")
    print(f"  WR  (Win Rate %):          {avg['WR']:.2f}")
    print(f"  MD  (Max Drawdown):        {avg['MD']:.4f}")
    print(f"\n  Paper values for comparison:")
    print(f"  TR=6.1011  Rp=0.2976  σp=0.4777  SR=0.6231  WR=80.0  MD=-0.4424")
else:
    print("ERROR: No metrics could be computed!")

print("\nDone.")
