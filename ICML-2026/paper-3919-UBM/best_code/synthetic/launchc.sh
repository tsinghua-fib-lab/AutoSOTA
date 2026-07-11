python main.py --bias_type "sel_bias_type2_9951" --bias_sel_type2_probs .9 .9 .5 .1 --d 2 --n_rct 50000 --n_val 2000 --num_trials 200
# python main.py --bias_trs --bias_Y1 --bias_type "transportability_bias_sel_bias_type2_9951" --bias_sel_type2_probs .9 .9 .5 .1 --d 6 --n_rct 50000 --n_val 2000 --num_trials 200

# python main.py --bias_trs --bias_Y1 --bias_type "transportability_bias_sel_bias_type2_1119" --bias_sel_type2_probs .1 .1 .1 .9 --d 6 --n_rct 50000 --n_val 2000 --num_trials 200
# python main.py --bias_trs --bias_Y1 --bias_type "transportability_bias_sel_bias_type2_9999" --bias_sel_type2_probs .9 .9 .9 .9 --d 6 --n_rct 50000 --n_val 2000 --num_trials 200
# python main.py --bias_trs --bias_Y1 --bias_type "transportability_bias_sel_bias_type2_99999999" --bias_sel_type2_probs .99 .99 .99 .99 --d 6 --n_rct 50000 --n_val 2000 --num_trials 200

# # other combinations 
# python main.py --bias_S --bias_A --bias_Y1 --bias_type "selection_bias_type1+confounding" --d 6 --n_rct 50000 --n_val 2000 
# python main.py --bias_S --bias_Y1 --bias_type "selection_bias_type1+type2" --bias_sel_type2_probs .1 .1 .1 .9 --d 6 --n_rct 50000 --n_val 2000 
# python main.py --bias_trs --bias_A --bias_Y1 --bias_type "transportability+confounding" --d 6 --n_rct 50000 --n_val 2000 
# python main.py --bias_A --bias_Y1 --bias_type "selection_bias_type2+confounding" --bias_sel_type2_probs .1 .1 .1 .9 --d 6 --n_rct 50000 --n_val 2000 

