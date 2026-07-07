#!/bin/bash

# 设置环境变量（可选）
export PYTHONPATH=$(pwd)

# 运行模拟系统
python Stock_Main/main.py \
  --Iterations_Daily 5 \
  --No_Days 10 \
  --Num_Person 12 \
  --Num_Stock 4 \
  --SAVE_NAME sim_test01 \
  --persona_name persona.json \
  --stock_name stocks.json \
  --Daily_Price_Limit 0.6 \
  --expense_ratio 0.02 \
  --Fluctuation_Constant 15.0 \
  --analysis_num 3 \
  --gossip_num_max 2 \
  --verbose
