import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')
TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'table')

# make dirs if they don't exist
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(TABLE_DIR): os.makedirs(TABLE_DIR)


REVERSED_METRICS = [
    'margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_byte_corr', 'correct_prob_per_char',
]

ALL_METRICS = [
    'primary_metric', 'correct_prob', 'correct_prob_per_token', 'correct_prob_per_char', 'margin', 'margin_per_token', 'margin_per_char', 'total_prob', 'total_prob_per_token', 'total_prob_per_char', 'norm_correct_prob', 'norm_correct_prob_per_token', 'norm_correct_prob_per_char',
    'acc_raw', 'acc_per_token', 'acc_per_char'
]

def get_title_from_task(task):
    if isinstance(task, list):
        if len(task) == 1:
            return task[0]
        title_mapping = {
            'mmlu_pro_': 'mmlu_pro',
            'mmlu_abstract_algebra:mc': 'mmlu_mc',
            'mmlu': 'mmlu',
            'minerva': 'minerva',
            'agi_eval': 'agi_eval',
            'bbh': 'bbh',
            'arc_challenge:para': 'olmes_core9_para',
            'arc_challenge:distractors': 'olmes_core9_distractors',
            'arc_challenge:enlarge': 'olmes_core9_enlarge',
            'arc_challenge:mc': 'olmes_core9_mc',
            'arc_challenge': 'olmes_core9',
            'drop': 'olmes_gen',
        }
        for key, title in title_mapping.items():
            if key in task[0]:
                return title
        return 'aggregate'
    return task
