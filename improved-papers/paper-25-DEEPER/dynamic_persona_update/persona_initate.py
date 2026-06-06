import re
import json
from utils.readkey import set_env
from prompt.initiate import PERSONA_INITIATE_PROMPT_TEMPLATE
import openai

set_env()

iteration = 1
prev = iteration-1
curr = iteration
fut = iteration+1


def generate_jsonl_string(items, user_ratings):
    jsonl_string = ""
    for item, user_rating in zip(items, user_ratings):
        json_record = {
            "item": item,
            "user_rating": user_rating
        }
        jsonl_string += json.dumps(json_record, ensure_ascii=False) + '\n'
    return jsonl_string

def extract_json_code(text):
    pattern = r'```json(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    return code_blocks

def extract_result_in_last_json_code(text):
    json_code = extract_json_code(text)
    if len(json_code) == 0:
        return None
    try:
        result = json.loads(json_code[-1].replace("\n"," "))
    except json.JSONinititateError:
        return None
    
    if not isinstance(result, dict) or "persona" not in result:
        return None
    return result

def persona_initate(sample_json):
    item_type = sample_json['item_type']  
    prev_items = sample_json[f'I{prev}']  
    prev_ratings = sample_json[f'U{prev}']
    item_ratings_jsonl = generate_jsonl_string(prev_items, prev_ratings)
    
    persona_inititate_prompt = PERSONA_INITIATE_PROMPT_TEMPLATE.format(
        item_type=item_type,
        prev_user_behaviours_with_items=item_ratings_jsonl
    )
    
    max_retries = 1
    attempts = 0
    while attempts < max_retries:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18", 
                messages=[
                    {"role": "user", "content": persona_inititate_prompt}
                ],
                max_tokens=16384,  
                temperature=0  
            )
            result = response.choices[0].message.content.strip()
            persona = extract_result_in_last_json_code(result)

            if persona is not None:
                return persona 
            else: 
                pass
            attempts += 1  

        except Exception as e:
            print(f"API call error: {str(e)}")
            attempts += 1  

    return None  
