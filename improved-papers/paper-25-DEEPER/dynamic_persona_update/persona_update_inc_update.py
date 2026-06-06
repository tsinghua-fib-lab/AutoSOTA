import re
import json
from utils.readkey import set_env
from prompt.inc_update import INC_UPDATE_PROMPT_TEMPLATE
import openai

set_env()

iteration = 1
prev = iteration-1
curr = iteration
fut = iteration+1

def inc_update(sample_json):
    item_type = sample_json['item_type']  
    items = sample_json[f'I{curr}']  
    user_ratings = sample_json[f'U{curr}'] 
    persona = sample_json[f'S{prev}']  
    actual = '\n'.join([
        str({"item_name": items[i], "user_rating": user_ratings[i]})
        for i in range(10)
    ])
    persona_encode_prompt = INC_UPDATE_PROMPT_TEMPLATE.format(
        item_type = item_type,
        old_persona=persona,
        inc_actual=actual
    )
    
    max_retries = 1
    attempts = 0

    while attempts < max_retries:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", 
                messages=[
                    {"role": "user", "content": persona_encode_prompt}
                ],
                max_tokens=16384,  
                temperature=0  
            )
            result = response.choices[0].message.content.strip()
            if result is not None:
                return result 
            else: 
                attempts += 1  

        except Exception as e:
            print(f"API call error: {str(e)}")
            attempts += 1  
