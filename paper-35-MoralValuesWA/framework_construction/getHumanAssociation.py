import json
import pandas as pd
import math
from loguru import logger
from collections import Counter

def read_csv_datafile(file_path):
    df_data = pd.read_csv(file_path)
    return df_data

def is_legal(word):
    def is_empty(response):
        if isinstance(response, float):
            if math.isnan(response):
                return True
        else:
            return False

    unks = {'aeroplane', 'arse', 'ax', 'bandana', 'bannister', 'behaviour', 'bellybutton', 'centre',
            'cheque', 'chequered', 'chilli', 'colour', 'colours', 'corn-beef', 'cosy', 'doughnut',
            'extravert', 'favour', 'fibre', 'hanky', 'harbour', 'highschool', 'hippy', 'honour',
            'hotdog', 'humour', 'judgment', 'labour', 'light bulb', 'lollypop', 'neighbour',
            'neighbourhood', 'odour', 'oldfashioned', 'organisation', 'organise', 'paperclip',
            'parfum', 'phoney', 'plough', 'practise', 'programme', 'pyjamas',
            'racquet', 'realise', 'recieve', 'saviour', 'seperate', 'theatre', 'tresspass',
            'tyre', 'verandah', 'whisky', 'WIFI', 'yoghurt', 'smokey'}
    if is_empty(word) or word in unks:
        return False
    else:
        return True
    
def unify_spellings(word):
    sub_dict = {'black out': 'blackout',
                'break up': 'breakup',
                'breast feeding': 'breastfeeding',
                'bubble gum': 'bubblegum',
                'cell phone': 'cellphone',
                'coca-cola': 'Coca Cola',
                'good looking': 'good-looking',
                'goodlooking': 'good-looking',
                'hard working': 'hardworking',
                'hard-working': 'hardworking',
                'lawn mower': 'lawnmower',
                'seat belt': 'seatbelt',
                'tinfoil': 'tin foil',
                'bluejay': 'blue jay',
                'bunk bed': 'bunkbed',
                'dingdong': 'ding dong',
                'dwarves': 'dwarfs',
                'Great Brittain': 'Great Britain',
                'lightyear': 'light year',
                'manmade': 'man made',
                'miniscule': 'minuscule',
                'pass over': 'passover'}

    if word in sub_dict:
        return sub_dict[word]
    else:
        return word
    

def preprocess(file_path):
    df_data = read_csv_datafile(file_path)
    n_records = df_data.shape[0]

    processed_data = {}
    for participant_id, record in df_data.iterrows():
        if participant_id % 50000 == 0:
            logger.info(f'{participant_id} processed, {n_records} in total')

        cue = record['cue']
        # whether cue is legal
        if not is_legal(cue):
            continue
        cue = unify_spellings(cue)

        responses = []
        for response in (record['R1'], record['R2'], record['R3']):
            if is_legal(response):
                responses.append(unify_spellings(response))

        if cue not in processed_data:
            processed_data[cue] = responses
        else:
            processed_data[cue] += responses

    processed_data = {cue: Counter(processed_data[cue]) for cue in processed_data}

    with open('data/human_association.json', 'w', encoding='utf-8') as json_file:
        json.dump(processed_data, json_file, ensure_ascii=False, indent=4)

    return processed_data

file_path = 'data/SWOW-EN.R100.csv'
processed_data = preprocess(file_path)