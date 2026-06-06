import re
import json
from utils.readkey import set_env
from prompt.predict import BEHAVIOR_PREDICT_PROMPT_TEMPLATE
import openai

set_env()

iteration = 1
prev = iteration-1
curr = iteration
fut = iteration+1

def predict_result_parse(predict_result):
    result = re.search(r'\[.*\]', predict_result, re.DOTALL)
    result_with_items = result.group(0) if result else None
    if result_with_items:
        try:
            
            result_with_items_json = json.loads(result_with_items)
            rating_list = [i["predict_rating"] for i in result_with_items_json]
            return rating_list
        except json.JSONDecodeError:
            print("Error decoding JSON:", result_with_items)
            return None
    return None


def behavior_predict(sample_json):
    item_type = sample_json['item_type']  
    items = sample_json[f'I{curr}']  
    persona = sample_json[f'S{prev}']  
    behavior_predict_prompt = BEHAVIOR_PREDICT_PROMPT_TEMPLATE.format(
        item_type=item_type,
        persona=persona,
        items=items
    )
    
    max_retries = 1
    attempts = 0

    while attempts < max_retries:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", 
                messages=[
                    {"role": "user", "content": behavior_predict_prompt}
                ],
                max_tokens=16384,  
                temperature=0  
            )
            result = response.choices[0].message.content.strip()
            predict = predict_result_parse(result)
            if predict is not None:
                return predict 
            else: 
                print("No valid prediction found.")
                attempts += 1  

        except Exception as e:
            print(f"API call error: {str(e)}")
            attempts += 1  
