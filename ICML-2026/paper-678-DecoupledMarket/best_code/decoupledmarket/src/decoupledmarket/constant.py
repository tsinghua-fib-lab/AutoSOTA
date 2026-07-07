import time
import os
import os.path as osp

current_milli_time = lambda: int(round(time.time() * 1000))


FORMAT = "%Y-%m-%d%H:%M:%S"
STOCK_NAMES = ["A", "B", "C", "D", "E"]

Daily_Price_Limit = 0.7
expense_ratio = 0.02
Fluctuation_Constant = 20.0
verbose = False
N = 1
Model_Names = ["gpt-3.5-turbo", "glm-5"]
# Simulation parameters
Iterations_Daily = 1
Num_virtual_agents = 37
No_Days = 30

Num_Person = 13
Num_Stock = 3
Num_agent = 15
Num_tech_traders = 35
SAVE_NAME = "sim01"  # Changed for comparison test without OpenCLAW
persona_name = "persona.json"
stock_name = "stocks.json"
Save_Path = osp.join("./save", SAVE_NAME)

Num_quantity=2500

if not os.path.exists(Save_Path):

    os.makedirs(Save_Path)

OUR_AGENT_COUNT = min(Num_Person, int(os.getenv("OUR_AGENT_COUNT", "8")))
our_agent = list(range(OUR_AGENT_COUNT))
agents = ["gpt-3.5-turbo", "glm-5"]
persona_path = osp.join(Save_Path, "persona.json")
agent_path = osp.join(Save_Path, "agent.json")
stock_path = osp.join(Save_Path, "stocks.json")
tech_path = osp.join(Save_Path, "tech_traders.json")
virtualagent_path = osp.join(Save_Path, "virtualagent_path.json")
table_analysis_id = [0]
analysis_num = Num_Stock
gossip_num_max = 2

# ===========================================
# Trading Pricing Constants
# ===========================================

# Order pricing multipliers
BID_PRICE_PREMIUM = 1.02  # Buy orders: slight premium (2% above market)
ASK_PRICE_DISCOUNT = 0.98  # Sell orders: slight discount (2% below market)

# Cash management
LIVING_EXPENSE_BUFFER_DAYS = 10  # Days of living expenses to reserve
INVESTABLE_CASH_RATIO = 0.7  # Ratio of surplus cash available for investment

# Market pricing configuration
DEFAULT_SPREAD_PCT = 0.03  # Default bid-ask spread (2%)
DEFAULT_IMPACT_FACTOR = 0.015  # Price impact factor
DEFAULT_TREND_WEIGHT = 0.4  # Weight for historical trend in pricing
DEFAULT_NOISE_LEVEL = 0.005  # Random noise level (0.5%)
MIN_PRICE = 0.01  # Minimum stock price

# Price change limits
MAX_CHANGE_PER_TRADE = 0.08  # Maximum 5% change per trade
ORDER_IMPACT_WEIGHT = 0.8  # Weight for order impact
RANDOM_IMPACT_WEIGHT = 0.25  # Weight for random fluctuations
REVERSION_IMPACT_WEIGHT = 0.15  # Weight for mean reversion

# Volatility settings
BASE_VOLATILITY = 0.04  # Base volatility (2%)
HIGH_PRICE_THRESHOLD_1000 = 1000  # Price threshold for reduced volatility
HIGH_PRICE_SCALE_1000 = 0.6  # Volatility scale for stocks > 1000
HIGH_PRICE_THRESHOLD_500 = 500
HIGH_PRICE_SCALE_500 = 0.8
HIGH_PRICE_THRESHOLD_100 = 100
HIGH_PRICE_SCALE_100 = 1.0
HIGH_PRICE_THRESHOLD_50 = 50
HIGH_PRICE_SCALE_50 = 1.2
HIGH_PRICE_THRESHOLD_10 = 10
HIGH_PRICE_SCALE_10 = 1.5
LOW_PRICE_SCALE = 2.0  # Volatility scale for low-priced stocks

# Mean reversion settings
REVERSION_STRENGTH = 0.5  # Mean reversion coefficient
REVERSION_LOOKBACK_DAYS = 10  # Days to look back for mean calculation
DEVIATION_THRESHOLD_HIGH = 0.1  # High deviation threshold (10%)
DEVIATION_THRESHOLD_LOW = 0.05  # Low deviation threshold (5%)
