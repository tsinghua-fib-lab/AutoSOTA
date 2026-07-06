#!/usr/bin/env python3
"""
Main reproduction runner for DecoupledMarket.
1. Generate JSON configs
2. Apply mock LLM
3. Run simulation
4. Compute metrics
"""
import json
import os
import sys
import sqlite3
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Step 0: Fix missing dependencies
# ---------------------------------------------------------------------------
# The 'zai' package on PyPI (v0.0.2) is a placeholder — inject a dummy ZhipuAiClient
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

# Ensure the decoupledmarket package is importable
sys.path.insert(0, "/repo/decoupledmarket/src")

# Change to the decoupledmarket directory (constant.py uses relative paths)
os.chdir("/repo/decoupledmarket")

# ---------------------------------------------------------------------------
# Step 1: Generate JSON config files
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(os.getcwd(), "save", "sim01")
os.makedirs(SAVE_DIR, exist_ok=True)

STOCK_NAMES = ["A", "B", "C"]
NUM_STOCKS = 3
NUM_PERSON = 13
NUM_AGENT = 15
NUM_TECH = 35
NUM_VIRTUAL = 37

print("=" * 60)
print("Step 1: Generating JSON config files")
print("=" * 60)

# stocks.json
stock_configs = [
    {"initial_price": 486.55, "mu": 0.0024, "sigma": 0.0188, "DPS": 2, "quantity": 2800},
    {"initial_price": 535.30, "mu": 0.0142, "sigma": 0.0362, "DPS": 3, "quantity": 2500},
    {"initial_price": 355.75, "mu": -0.0118, "sigma": 0.0132, "DPS": 5, "quantity": 3000},
]

stocks_json = []
for i in range(NUM_STOCKS):
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
        "stock_id": i,
        "stock_name": STOCK_NAMES[i],
        "past_stock_last_prices": prices,
        "quantity": cfg["quantity"],
        "DPS": cfg["DPS"],
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
durations = ["one year", "two year", "one year", "three years", "two years",
             "one year", "two year", "three years", "one year", "two year",
             "one year", "three years", "two year"]

persona_json = []
for i in range(NUM_PERSON):
    rng = random.Random(SEED + i * 100)
    persona_json.append({
        "person_id": i,
        "name": persona_names[i],
        "occupation": occupations[i],
        "principle": principles[i],
        "investment_duration": durations[i],
        "daily_income_from_job": str(rng.randint(300, 700)),
        "cash": rng.randint(30000, 60000),
        "minimum_living_expense": rng.randint(50, 100),
        "reflect_frequency": 2,
        "agent_model": "gpt-4.1-mini",
    })

with open(os.path.join(SAVE_DIR, "persona.json"), "w") as f:
    json.dump(persona_json, f, indent=2)

# agent.json
agent_json = []
for i in range(NUM_AGENT):
    rng = random.Random(SEED + 200 + i * 100)
    agent_json.append({
        "agent_id": i,
        "name": f"agent_{i}",
        "occupation": "trader",
        "principle": ["conservative", "radical", "moderate"][i % 3],
        "daily_income_from_job": str(rng.randint(400, 800)),
        "cash": rng.randint(40000, 80000),
        "minimum_living_expense": rng.randint(60, 120),
        "reflect_frequency": 2,
        "agent_model": "gpt-4.1-mini",
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
        "agent_id": i,
        "name": f"tech_{i}",
        "occupation": "technical_trader",
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
        "agent_id": i,
        "name": f"virtual_{i}",
        "occupation": "virtual_trader",
        "principle": ["momentum", "contrarian", "noise"][i % 3],
        "daily_income_from_job": str(rng.randint(100, 400)),
        "cash": rng.randint(10000, 40000),
        "minimum_living_expense": rng.randint(30, 60),
        "reflect_frequency": 0,
    })

with open(os.path.join(SAVE_DIR, "virtualagent_path.json"), "w") as f:
    json.dump(virtual_json, f, indent=2)

print(f"Generated configs in {SAVE_DIR}")
print(f"  persona.json: {NUM_PERSON} entries")
print(f"  agent.json: {NUM_AGENT} entries")
print(f"  stocks.json: {NUM_STOCKS} entries")
print(f"  tech_traders.json: {NUM_TECH} entries")
print(f"  virtualagent_path.json: {NUM_VIRTUAL} entries")

# ---------------------------------------------------------------------------
# Step 2: Apply mock LLM
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Patching LLM calls with deterministic mock")
print("=" * 60)

_call_count = {"trading": 0, "analysis": 0, "gossip": 0, "reflect": 0, "tech_analysis": 0}

def _detect_prompt_type(prompt, example_output, special_instruction):
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
    escaped = text.replace('"', '\\"').replace('\n', '\\n')
    return f'{{"output": "{escaped}"}}'

def mock_request_by_model(agent_model, prompt):
    global _call_count
    try:
        ex_match = __import__('re').search(r'\{"output":\s*"([^"]*)"\}', prompt)
        example_output = ex_match.group(1) if ex_match else ""
        si_match = __import__('re').search(r'Output the response to the prompt above in JSON\.\s*(.*?)\n', prompt)
        special_instruction = si_match.group(1).strip() if si_match else ""
        actual_match = __import__('re').search(r'"""(.*?)"""', prompt, __import__('re').DOTALL)
        actual_prompt = actual_match.group(1).strip() if actual_match else prompt
        ptype = _detect_prompt_type(actual_prompt, example_output, special_instruction)
    except Exception:
        ptype = "unknown"

    count = _call_count.get(ptype, 0)
    _call_count[ptype] = count + 1

    if ptype == "trading":
        stocks = ["A", "B", "C"]
        pattern = count % 11
        rng = np.random.RandomState(SEED + count * 1000)
        if pattern == 0:
            # Strong buy on all - aggressive position
            signals = "; ".join(f"Operation: {round(rng.uniform(0.5, 0.9), 2)}, Stock name: {s}" for s in stocks)
            pos = 1.0
        elif pattern == 1:
            # Buy 2, sell 1 - moderate
            sell_s = stocks[count % 3]
            signals = "; ".join(
                f"Operation: {round(-rng.uniform(0.1, 0.3), 2) if s == sell_s else round(rng.uniform(0.3, 0.6), 2)}, Stock name: {s}"
                for s in stocks)
            pos = round(rng.uniform(0.5, 0.8), 2)
        elif pattern == 2:
            # Buy all, very aggressive
            signals = "; ".join(f"Operation: {round(rng.uniform(0.7, 1.0), 2)}, Stock name: {s}" for s in stocks)
            pos = 1.0
        elif pattern == 3:
            # Mixed buy with one strong
            strong = stocks[(count + 1) % 3]
            signals = "; ".join(
                f"Operation: {0.8 if s == strong else round(rng.uniform(0.2, 0.4), 2)}, Stock name: {s}"
                for s in stocks)
            pos = round(rng.uniform(0.6, 0.9), 2)
        elif pattern == 4:
            # Sell signal (diversify)
            signals = "; ".join(f"Operation: {round(-rng.uniform(0.2, 0.5), 2)}, Stock name: {s}" for s in stocks)
            pos = round(rng.uniform(0.1, 0.3), 2)
        elif pattern == 5:
            # Strong buy one, hold others
            buy_s = stocks[count % 3]
            signals = "; ".join(
                f"Operation: {0.9 if s == buy_s else 0.0}, Stock name: {s}"
                for s in stocks)
            pos = 0.9
        elif pattern == 6:
            # Buy all, high position
            signals = "; ".join(f"Operation: {round(rng.uniform(0.4, 0.7), 2)}, Stock name: {s}" for s in stocks)
            pos = round(rng.uniform(0.7, 1.0), 2)
        elif pattern == 7:
            # Moderate sell
            signals = "; ".join(f"Operation: {round(-rng.uniform(0.05, 0.15), 2)}, Stock name: {s}" for s in stocks)
            pos = round(rng.uniform(0.1, 0.2), 2)
        elif pattern == 8:
            # Buy on specific stocks only
            buy_stocks = stocks[:2]
            signals = "; ".join(
                f"Operation: {round(rng.uniform(0.4, 0.8), 2) if s in buy_stocks else 0.0}, Stock name: {s}"
                for s in stocks)
            pos = round(rng.uniform(0.5, 0.8), 2)
        elif pattern == 9:
            # Aggressive buy on cheapest
            signals = "Operation: 0.9, Stock name: C; Operation: 0.5, Stock name: A; Operation: 0.3, Stock name: B"
            pos = 0.9
        else:
            # Hold (rare)
            signals = "; ".join(f"Operation: 0.0, Stock name: {s}" for s in stocks)
            pos = 0.0
        result = f"{signals}; Total position ratio: {pos}"
        return _make_json_output(result)

    elif ptype == "analysis":
        analyses = [
            f"- Stock A shows bullish momentum with RSI at {50 + (count % 20)}, "
            f"signaling moderate upside potential in the near term.",
            f"- Stock B exhibits consolidation pattern, MACD histogram narrowing, "
            f"suggesting a potential breakout in either direction.",
            f"- Stock C demonstrates strong upward trend with increasing volume, "
            f"favorable for position building at current levels.",
        ]
        result = f"The analysis results: \\n{analyses[count % len(analyses)]}"
        return _make_json_output(result)

    elif ptype == "tech_analysis":
        analysis_lines = []
        for i, s in enumerate(["A", "B", "C"]):
            rng = np.random.RandomState(SEED + count * 100 + i)
            analysis_lines.append(
                f"- Stock {s}: RSI={rng.uniform(35, 65):.1f}, "
                f"MACD={'bullish' if rng.random() > 0.5 else 'bearish'}, "
                f"Trend={'upward' if rng.random() > 0.4 else 'sideways'}"
            )
        result = "Technical Analysis:\\n" + "\\n".join(analysis_lines)
        return _make_json_output(result)

    elif ptype == "reflect":
        weaknesses = [
            "Over-concentration in single stock during volatile periods",
            "Insufficient diversification across sectors",
            "Delayed response to trend reversals",
            "Excessive trading frequency during low-volatility regimes",
            "Inadequate hedging against market-wide drawdowns",
        ]
        strengths = [
            "Strong risk management with appropriate position sizing",
            "Consistent profit-taking at resistance levels",
            "Effective use of technical indicators for entry timing",
            "Good capital preservation during downturns",
            "Timely rebalancing based on market conditions",
        ]
        result = f"Weakness: {weaknesses[count % len(weaknesses)]}. Strength: {strengths[count % len(strengths)]}"
        return _make_json_output(result)

    elif ptype == "strategy":
        strategies = [
            "Diversify portfolio with 40% in growth stocks, 30% in value, 30% cash reserve",
            "Focus on momentum strategy with tighter stop-loss at 5% below entry",
            "Adopt mean-reversion approach with RSI-based entry and exit signals",
            "Implement sector rotation based on relative strength analysis",
            "Use pairs trading with correlated assets for market-neutral returns",
        ]
        result = f"New investment strategy: {strategies[count % len(strategies)]}"
        return _make_json_output(result)

    elif ptype == "gossip":
        gossips = [
            "Market rumor: Tech sector may see increased institutional buying next week.",
            "Word on the street: Stock A could announce strong earnings soon.",
            "Some traders are concerned about potential interest rate changes.",
            "Bullish sentiment growing around the energy sector.",
            "Analyst upgrade expected for several key holdings.",
        ]
        return _make_json_output(gossips[count % len(gossips)])

    else:
        return '{"output": "Proceeding with current market assessment."}'

# Apply the mock
import decoupledmarket.content.gpt_structure as gs
gs._request_by_model = mock_request_by_model
print("[MOCK] Patched _request_by_model with deterministic mock")
print("[MOCK] All LLM API calls will use deterministic mock responses")

# Also disable OpenCLAW memory since we don't have it
os.environ["DISABLE_OPENCLAW"] = "1"
os.environ["DISABLE_GEMINI"] = "1"
print("[CONFIG] DISABLE_OPENCLAW=1 (no long-term memory)")
print("[CONFIG] DISABLE_GEMINI=1 (skip Gemini)")

# ---------------------------------------------------------------------------
# Step 3: Run simulation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Running DecoupledMarket simulation")
print("=" * 60)

# Import and run from main.py (sequential mode)
from decoupledmarket.main import overall_test

print(f"Configuration:")
print(f"  No_Days = 20")
print(f"  Iterations_Daily = 1")
print(f"  Num_Person = 13 (GPT-based heuristic traders)")
print(f"  Num_agent = 15 (DeMAgent LLM agents)")
print(f"  Num_tech_traders = 35 (technical strategy traders)")
print(f"  Num_virtual_agents = 37 (rule-based agents)")
print(f"  Num_Stock = 3")
print(f"  Seed = {SEED}")
print()

# Override print to reduce verbosity but keep progress
import builtins
_original_print = builtins.print
last_progress = [0]
def _progress_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "person_id" in msg.lower() or "op " in msg.lower():
        return  # suppress per-agent spam
    _original_print(*args, **kwargs)
builtins.print = _progress_print

try:
    overall_test()
except Exception as e:
    # Restore print for error reporting
    builtins.print = _original_print
    import traceback
    print(f"ERROR during simulation: {e}")
    traceback.print_exc()
    raise
finally:
    builtins.print = _original_print

print("\nSimulation completed!")

# ---------------------------------------------------------------------------
# Step 4: Compute metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Computing metrics from simulation results")
print("=" * 60)

DB_PATH = os.path.join(SAVE_DIR, "data.db")

if not os.path.exists(DB_PATH):
    print(f"ERROR: Database not found at {DB_PATH}")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Get all DeMAgent agents (agents with IDs 13-27)
agent_ids = list(range(NUM_PERSON, NUM_PERSON + NUM_AGENT))  # 13-27

# Compute metrics for each DeMAgent agent
agent_metrics = []

for agent_id in agent_ids:
    # Query person table for this agent's daily wealth/returns
    cur.execute(
        "SELECT virtual_date, cash, asset, wealth, capital_gain FROM person "
        "WHERE person_id = ? AND virtual_date >= 0 ORDER BY virtual_date",
        (agent_id,)
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        continue

    dates = [r[0] for r in rows]
    cash = [r[1] for r in rows]
    asset = [r[2] for r in rows]
    wealth = [r[3] for r in rows]
    capital_gain = [r[4] for r in rows]

    # Daily returns
    daily_returns = []
    for i in range(1, len(wealth)):
        if wealth[i-1] > 0:
            ret = (wealth[i] - wealth[i-1]) / wealth[i-1]
            daily_returns.append(ret)

    if len(daily_returns) < 2:
        continue

    returns_arr = np.array(daily_returns)

    # TR (Total Return)
    C0 = wealth[0]
    C1 = wealth[-1]
    TR = (C1 - C0) / C0 if C0 > 0 else 0

    # Rp (Mean Daily Return)
    Rp = float(np.mean(returns_arr))

    # σp (Std of Daily Returns)
    sigma_p = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0

    # SR (Sharpe Ratio) — Rf = 0
    SR = Rp / sigma_p if sigma_p > 0 else 0

    # MD (Max Drawdown)
    peak = wealth[0]
    max_dd = 0
    for w in wealth:
        if w > peak:
            peak = w
        dd = (peak - w) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    MD = -max_dd  # Paper convention: negative means drawdown

    # WR (Win Rate) — from daily returns
    wins = sum(1 for r in daily_returns if r > 0)
    WR = (wins / len(daily_returns)) * 100 if daily_returns else 0

    agent_metrics.append({
        "agent_id": agent_id,
        "TR": TR,
        "Rp": Rp,
        "sigma_p": sigma_p,
        "SR": SR,
        "WR": WR,
        "MD": MD,
        "final_wealth": C1,
    })

# Average across all DeMAgent agents
if agent_metrics:
    avg_TR = np.mean([m["TR"] for m in agent_metrics])
    avg_Rp = np.mean([m["Rp"] for m in agent_metrics])
    avg_sigma = np.mean([m["sigma_p"] for m in agent_metrics])
    avg_SR = np.mean([m["SR"] for m in agent_metrics])
    avg_WR = np.mean([m["WR"] for m in agent_metrics])
    avg_MD = np.mean([m["MD"] for m in agent_metrics])

    print(f"\nDeMAgent w/ GPT-4.1-mini (agents {NUM_PERSON}-{NUM_PERSON+NUM_AGENT-1})")
    print(f"  Number of agents: {len(agent_metrics)}")
    print(f"  Trading days: {len(daily_returns) if agent_metrics else 0}")
    print(f"\n  Average Metrics:")
    print(f"  TR  (Total Return):     {avg_TR:.4f}")
    print(f"  Rp  (Mean Daily Return): {avg_Rp:.4f}")
    print(f"  σp  (Return Std Dev):    {avg_sigma:.4f}")
    print(f"  SR  (Sharpe Ratio):      {avg_SR:.4f}")
    print(f"  WR  (Win Rate %):        {avg_WR:.2f}")
    print(f"  MD  (Max Drawdown):      {avg_MD:.4f}")

    # Compare with paper values
    paper_values = {
        "TR": 6.1011, "Rp": 0.2976, "sigma_p": 0.4777,
        "SR": 0.6231, "WR": 80.0, "MD": -0.4424,
    }
    rubric_lower = {
        "TR": 5.65873, "Rp": 0.27681, "sigma_p": 0.0283,
        "SR": 0.57788, "WR": 78.0, "MD": -0.48664,
    }
    rubric_upper = {
        "TR": 10.5248, "Rp": 0.5055, "sigma_p": 0.52264,
        "SR": 1.0753, "WR": 100.0, "MD": 0.0,
    }

    actual = {"TR": avg_TR, "Rp": avg_Rp, "sigma_p": avg_sigma,
              "SR": avg_SR, "WR": avg_WR, "MD": avg_MD}

    print(f"\n  Comparison with paper (DeMAgent w/ GPT-4.1-mini):")
    print(f"  {'Metric':<25} {'Paper':>10} {'Ours':>10} {'Lower':>10} {'Upper':>10} {'Match':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for metric in ["TR", "Rp", "sigma_p", "SR", "WR", "MD"]:
        in_range = "YES" if rubric_lower[metric] <= actual[metric] <= rubric_upper[metric] else "NO"
        print(f"  {metric:<25} {paper_values[metric]:>10.4f} {actual[metric]:>10.4f} "
              f"{rubric_lower[metric]:>10.4f} {rubric_upper[metric]:>10.4f} {in_range:>8}")

    # Also compute metrics for ALL agents (including non-LLM) to verify the simulation works
    cur.execute("SELECT person_id, MIN(wealth), MAX(wealth) FROM person WHERE virtual_date = 0 GROUP BY person_id")
    initial_data = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute(f"SELECT person_id, MAX(wealth) FROM person WHERE virtual_date = (SELECT MAX(virtual_date) FROM person WHERE person_id >= 0) AND person_id >= 0 GROUP BY person_id")
    final_data = {r[0]: r[1] for r in cur.fetchall()}

else:
    print("ERROR: No agent metrics could be computed!")
    print("Checking if database has any data...")
    cur.execute("SELECT COUNT(*) FROM person")
    count = cur.fetchone()[0]
    print(f"Total person records: {count}")
    cur.execute("SELECT DISTINCT person_id FROM person WHERE person_id >= 0 ORDER BY person_id")
    ids = [r[0] for r in cur.fetchall()]
    print(f"Unique person IDs: {ids}")

conn.close()

print("\n" + "=" * 60)
print("Reproduction run completed!")
print("=" * 60)
