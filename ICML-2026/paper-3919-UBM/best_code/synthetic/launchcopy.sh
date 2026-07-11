# python3 main.py --bias_S --bias_Y1 --bias_type "selection_bias" --d 5 6 7 8 --n_rct 2000 --n_val 2000 &
# python3 main.py --bias_A --bias_Y1 --bias_type "confounding_bias" --d 5 6 7 8 --n_rct 2000 --n_val 2000 &
# python3 main.py --bias_trs --bias_Y1 --bias_type "transportability_bias" --d 5 6 7 8 --n_rct 2000 --n_val 2000 &
# python3 main.py --bias_S --bias_A --bias_trs --bias_Y1 --full_adj --bias_type "all_bias_full_adj" --d 5 6 7 8 --n_rct 2000 --n_val 2000

# wait

# python3 plot.py --bias_type "selection_bias" --d 5 6 7 --n_rct 2000 --n_val 2000 &
# python3 plot.py --bias_type "confounding_bias" --d 5 6 7 --n_rct 2000 --n_val 2000 &
# python3 plot.py --bias_type "transportability_bias" --d 5 6 7 --n_rct 2000 --n_val 2000 &
# python3 plot.py --bias_type "all_bias_full_adj" --d 5 6 7 --n_rct 2000 --n_val 2000

# wait 

# python3 main.py --bias_S --bias_Y1 --bias_type "selection_bias" --d 5 6 7 --n_rct 50000 --n_val 2000 &
# python3 main.py --bias_A --bias_Y1 --bias_type "confounding_bias" --d 5 6 7 --n_rct 50000 --n_val 2000 &
# python3 main.py --bias_trs --bias_Y1 --bias_type "transportability_bias" --d 5 6 7 --n_rct 50000 --n_val 2000 &
# python3 main.py --bias_S --bias_A --bias_trs --bias_Y1 --full_adj --bias_type "all_bias_full_adj" --d 5 6 7 --n_rct 50000 --n_val 2000

# wait

python3 plot.py --bias_type "selection_bias" --d 5 6 7 --n_rct 50000 --n_val 2000 &
python3 plot.py --bias_type "confounding_bias" --d 5 6 7 --n_rct 50000 --n_val 2000 &
python3 plot.py --bias_type "transportability_bias" --d 5 6 7 --n_rct 50000 --n_val 2000 &
python3 plot.py --bias_type "all_bias_full_adj" --d 5 6 7 --n_rct 50000 --n_val 2000

