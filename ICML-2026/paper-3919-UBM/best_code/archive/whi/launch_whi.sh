python main.py --selection_flag biased --censored --outcome_name CHD --model_type LR 
python main.py --selection_flag biased --censored --outcome_name CHD --model_type RF 
python main.py --selection_flag biased --censored --outcome_name STROKE --model_type LR 
python main.py --selection_flag biased --censored --outcome_name STROKE --model_type RF
python main.py --selection_flag unbiased --censored --outcome_name CHD --model_type LR 
python main.py --selection_flag unbiased --censored --outcome_name CHD --model_type RF 
python main.py --selection_flag unbiased --censored --outcome_name STROKE --model_type LR 
python main.py --selection_flag unbiased --censored --outcome_name STROKE --model_type RF
python main.py --selection_flag manually_biased --censored --outcome_name CHD --model_type LR
python main.py --selection_flag manually_biased --censored --outcome_name CHD --model_type RF 
python main.py --selection_flag manually_biased --censored --outcome_name STROKE --model_type LR 
python main.py --selection_flag manually_biased --censored --outcome_name STROKE --model_type RF

python main.py --selection_flag biased --outcome_name CHD --model_type LR 
python main.py --selection_flag biased --outcome_name CHD --model_type RF 
python main.py --selection_flag biased --outcome_name STROKE --model_type LR 
python main.py --selection_flag biased --outcome_name STROKE --model_type RF
python main.py --selection_flag unbiased --outcome_name CHD --model_type LR 
python main.py --selection_flag unbiased --outcome_name CHD --model_type RF 
python main.py --selection_flag unbiased --outcome_name STROKE --model_type LR 
python main.py --selection_flag unbiased --outcome_name STROKE --model_type RF
python main.py --selection_flag manually_biased --outcome_name CHD --model_type LR 
python main.py --selection_flag manually_biased --outcome_name CHD --model_type RF 
python main.py --selection_flag manually_biased --outcome_name STROKE --model_type LR 
python main.py --selection_flag manually_biased --outcome_name STROKE --model_type RF


