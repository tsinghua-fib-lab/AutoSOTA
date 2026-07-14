import json 


def find_json_bounds(text):
    """Find (start,end) positions of all outermost {} pairs"""
    bounds = []
    start = None
    bracket_stack = 0

    for i, char in enumerate(text):
        if char == '{':
            if start is None:  # New opening
                start = i
            bracket_stack += 1
        elif char == '}' and bracket_stack > 0:
            bracket_stack -= 1 
            if bracket_stack == 0:
                # Found a complete JSON object
                bounds.append((start, i + 1))
                start = None

    return bounds  # List of (start, end) tuples


def mask_dict_values(data):
    """Recursively mask all values in the dict/json."""
    if isinstance(data, dict):
        # Process dictionaries while preserving order
        masked_dict = {}
        for key, value in data.items():
            masked_dict[key] = mask_dict_values(value)
        return masked_dict
    elif isinstance(data, (list, tuple)):
        # Process lists/tuples (mask non-dict elements, recurse into dict elements)
        return [mask_dict_values(item) for item in data]
    else:
        # Mask all non-dict values
        return "<*>"


def mask_dict_values_in_log(log):
    """Mask all values in the dict/json in a log."""
    # Find all JSON bounds
    bounds = find_json_bounds(log)
    if not bounds:
        return log

    # Process each JSON object found
    for start, end in bounds:
        json_str = log[start:end]

        retry = False
        try:
            data = json.loads(json_str.replace("'", '"'))
            masked_data = mask_dict_values(data)
            masked_json_str = json.dumps(masked_data)
            log = log[:start] + masked_json_str + log[end:]
        except json.JSONDecodeError as e:
            pass
            #print(f"[WARN] Invalid JSON object: {json_str}")
            #raise Exception(f"[WARN] Invalid JSON object: {json_str}")
    return log


