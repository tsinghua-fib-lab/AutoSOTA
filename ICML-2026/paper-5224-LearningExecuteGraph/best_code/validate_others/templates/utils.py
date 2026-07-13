import numpy as np
from collections import defaultdict

def nested_dict(levels):
    if levels == 1:
        return defaultdict(dict)
    return defaultdict(lambda: nested_dict(levels - 1))

def flatten_sample(sample):
    flat = []
    def _flatten(value):
        if isinstance(value, dict):
            # The order of values matters
            for v in value.values():
                _flatten(v)
        elif isinstance(value, list):
            for item in value:
                _flatten(item)
        else:
            # This is the base case: a non-dict, non-list item (e.g., a number).
            flat.append(value)
    
    _flatten(sample)
    return flat
    
def set_sample(x, template):
    for k1 in template:
        for k2 in template[k1]:
            x[k1][k2] = template[k1][k2]
    return x

def _recursive_match(xhat_val, template_val):
    """
    Recursively checks if a template value matches the corresponding xhat value.
    """
    # Base case: If the template is not a dictionary, it's a final value to compare.
    # This handles cases like {'blue': 0} where 0 is the template_val.
    if not isinstance(template_val, dict):
        return xhat_val == template_val

    # Recursive step: The template is a dictionary, so we need to go deeper.
    # We expect the corresponding xhat value to be a list we can index into.
    if not isinstance(xhat_val, list):
        return False  # Cannot apply a nested template to a non-list value.

    for key, next_template_val in template_val.items():
        # 'key' must be an integer index for the list 'xhat_val'.
        # Check if the index is valid.
        if not (isinstance(key, int) and 0 <= key < len(xhat_val)):
            return False

        # Make the recursive call for the next level down.
        # If any deeper match fails, this entire branch fails.
        if not _recursive_match(xhat_val[key], next_template_val):
            return False
            
    # If the loop completes without any failures, it means all items in this
    # level of the template matched successfully.
    return True

def match_sample(xhat, T):
    """
    Checks if any part of dictionary xhat matches a template in the list T.

    This revised version uses a recursive helper to handle arbitrary nesting
    levels in the templates and different data types in xhat.
    """
    matches = []
    
    for i, template in enumerate(T):
        is_a_match = True
        
        for k1 in template:  # This is the top-level key (e.g., 'message', 'blue').
            # The top-level key must exist in xhat.
            if k1 not in xhat:
                is_a_match = False
                break
            
            # Use the recursive helper to check this entire branch.
            if not _recursive_match(xhat[k1], template[k1]):
                is_a_match = False
                break
        
        if is_a_match:
            matches.append(i)
            
    return matches


def encode_data(xhat, T):
    x = np.zeros(len(T))
    matches = match_sample(xhat, T)
    x[matches] = 1
    no_norm_x = x
    if x.sum() > 0:
        x = x/np.sqrt(x.sum())
    return no_norm_x, x


## get the first index of a key e.g. 'message_slot'. This is used for separating message part from computation part

def get_key_start_index(target_key, template):
    """
    Finds the starting index of a key's values in the flattened list.

    Args:
        target_key (str): The key to search for.
        template (dict): The dictionary template with the same structure as the original sample.

    Returns:
        int: The starting index of the key's first element in the flattened list,
             or -1 if the key is not found.
    """
    # Use a list to hold the index so it's mutable across recursive calls
    current_index = [0] 
    # Store the result, initialized to -1 (not found)
    found_index = [-1]

    def _count_elements(structure):
        # Stop traversing if the key has already been found and its index stored
        if found_index[0] != -1:
            return

        if isinstance(structure, dict):
            for key, value in structure.items():
                if key == target_key:
                    # Found the key! The current index is its starting position.
                    found_index[0] = current_index[0]
                    return # Stop the search in this branch
                
                # Recursively process the value to count its elements
                _count_elements(value)

        elif isinstance(structure, list):
            for item in structure:
                _count_elements(item)
        else:
            # Base case: This is a leaf node (a single element). Increment the count.
            current_index[0] += 1
    
    _count_elements(template)
    return found_index[0]

# The new, recursive unflattening function.
def unflatten_sample(flat_list, template):
    """
    Reconstructs any sample structure from a flat list using the original
    template and a recursive helper function.
    """
    # Create an iterator for the flat list to consume elements one by one.
    flat_iter = iter(flat_list)

    # This recursive function will be the core of the logic.
    def _reconstruct(structure_template):
        # If the template part is a list, recurse on each of its elements.
        if isinstance(structure_template, list):
            return [_reconstruct(item) for item in structure_template]
        
        # If the template part is a dict, recurse on each of its values.
        elif isinstance(structure_template, dict):
            return {key: _reconstruct(value) for key, value in structure_template.items()}
            
        # Base Case: If it's not a list or dict, it's a placeholder for a value.
        # Take the next item from the iterator.
        else:
            return next(flat_iter)

    # Start the recursion with the top-level template.
    return _reconstruct(template)


