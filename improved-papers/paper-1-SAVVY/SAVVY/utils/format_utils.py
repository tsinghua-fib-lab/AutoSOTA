"""
Format utils code for SAVVY pipeline postprocessing - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import json
import re
import numpy as np
import math


def na_postprocess(pred):
    """
    Process prediction strings in various formats.

    Possible formats:
    - List notation: [x, y, z]
    - Tuple notation: (x, y, z)
    - Single value: c

    Returns a list of extracted values.
    """
    try:
        # Check if the prediction contains a list or tuple
        numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', str(pred))
        if len(numbers) > 1:
            return [float(num) for num in numbers]
        elif len(numbers) == 1:
            return float(numbers[0])
        else:
            # If no brackets/parentheses, return the first value as a single-item list
            return float(pred)
    except:
        return None


def fix_json(data):
    data = re.sub(r'\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', r'[\1, \2]', data)
    data = re.sub(r',\s*}', '}', data)
    data = re.sub(r',\s*]', ']', data)

    return data

def parse(content, res_path="result.json"):
    content = content.text
    json_start = content.find('```json')
    if json_start != -1:
        json_start = content.find('{', json_start)
        # Find the closing brace and backticks
        json_end = content.rfind('}')
        content = content[json_start:json_end+1]
        try:
            response_json = json.loads(content)
        except:
            try:
                content = '[' + content + ']'
                response_json = json.loads(content)
            except:
                try:
                    response_json = json.loads(fix_json(content))
                except:
                    json.dump(content, open(res_path, "w"), indent=4)
                    return content
    else:
        response_json = content.text
    json.dump(response_json, open(res_path, "w"), indent=4)
    return response_json


def parse_qa(content):
    content = content.text
    pred_json_item = {}
    try:
        json_start = content.find('```json')
        if json_start == -1:
            pred_json_item["prediction"] = content
            pred_json_item["reasoning"] = ""
        else:
            json_start = content.find('{', json_start)
            # Find the closing brace and backticks
            json_end = content.rfind('}')
            content = content[json_start:json_end+1]
            response_json = json.loads(content)
            pred_json_item["prediction"] = response_json.get("prediction", content)
            pred_json_item["reasoning"] = response_json.get("reasoning", "")
    except:
        # If JSON parsing fails, use original behavior
        pred_json_item["prediction"] = content
        pred_json_item["reasoning"] = ""
        import pdb; pdb.set_trace()
    return pred_json_item

def time_to_second(time_str):
    time_parts = time_str.split(":")
    time_in_seconds = 0
    for idx, time_p in enumerate(time_parts):
        time_in_seconds += 60**(len(time_parts)-1-idx) * float(time_p)
    return int(time_in_seconds)

def second_to_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"


def string_to_tuple(s):
    """
    Converts a string representation of a tuple (e.g., "(0.96, -5.56)")
    to an actual tuple of floats.

    Args:
    s: The string to convert.

    Returns:
    A tuple of floats, or None if the string is not in the correct format.
    """
    # Remove the parentheses.
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None  # Handle cases where the string is not a valid tuple representation
    s = s[1:-1]

    # Split the string by the comma.
    parts = s.split(",")

    # Convert the parts to floats and create the tuple.
    try:
        result_tuple = tuple(float(part.strip()) for part in parts)
        return result_tuple
    except ValueError:
        return None # Handle cases where the parts are not valid floats


def find_closest(numbers, target):
    """
    Find the number in an array that is closest to the target number using array operations.
    
    Args:
        numbers (list or array): Array of numbers to search through
        target (float or int): Target number to find closest value to
        
    Returns:
        The number from the array closest to the target
    """
    # Convert to numpy array if not already
    numbers_array = np.array(numbers)
    
    # Calculate absolute differences
    differences = np.abs(numbers_array - target)
    
    # Find index of minimum difference
    min_index = np.argmin(differences)
    
    # Return the closest value
    return numbers_array[min_index]




def get_fuzzy_match_name(cur_name, target_list, thr=0.5):
    if cur_name in target_list:
        return cur_name
    max_match = 0
    match_gt = None
    for cur_target in target_list:
        score = longest_common_subsequence(cur_target, cur_name)
        if score > max_match:
            max_match = score
            match_gt = cur_target
    if max_match < thr:
        match_gt = None
    return match_gt

def longest_common_subsequence(s1, s2):
    """Find the longest common subsequence of two strings using NumPy for efficiency."""
    # Create arrays for lengths
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    m, n = len(s1), len(s2)
    
    # We only need two rows at a time
    current = np.zeros(n+1, dtype=int)
    previous = np.zeros(n+1, dtype=int)
    for i in range(1, m+1):
        # Swap rows
        previous, current = current, previous
        
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                current[j] = previous[j-1] + 1
            else:
                current[j] = max(previous[j], current[j-1])
    
    lcs_length = current[n]
    
    # Calculate similarity
    max_len = max(m, n)
    if max_len == 0:
        return 0.0
    
    return lcs_length / max_len

