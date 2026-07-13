import numpy as np
from .utils import flatten_sample, nested_dict
import csv

def get_sample(l,D):    #total size of the un-encoded vector: k = 24 D^3 l + 8 D^3 + 35 D^2 l + 11 D^2 + 63 D l + 25 D + 52 l + 19
    num_slots = D**2+1
    sample = {
        ##Checking if the node id is equal to current; size :8l+1
        'u_id': [0]*l, # conflicts= 4
        'current': [0]*l, # conflicts= 1
        'compare_current': [0]*l, # conflicts= 1
        'id_equals': [0]*l, # conflicts= 3
        'id_equals_counter': [0]*l, # conflicts= 3
        'accept_stamp': [0]*l, # conflicts= 1
        'comparison_counter': [0]*(l+1), # conflicts= 1
        'reset_is_equal': [0]*l, # conflicts= 1
        
        ##Get the message time stamp value and increment it and activate end of loop flags : 7l+D

        'stamp': [0]*l, # conflicts= 1
        'u_stamp_addend_delay': [0]*D, # conflicts= 2
        'u_stamp_augend': [0]*l, # conflicts= 3
        'u_stamp_addend': [0]*l, # conflicts= 2
        'to_copy_stamp_augend': [0]*l, # conflicts= 1
        'u_stamp_carry': [0]*l, # conflicts= 2
        'u_stamp_sum_counter': [0]*(2*l), # conflicts= 2

        ## Checking which neighboring node should be attended and activate flags to update its features : 5D+2

        'v_turn': [0]*(D+1), # conflicts= 2
        'u_is_current': [0]*(D+1), # conflicts= 2
        'attend2v': [0]*D, # conflicts= 1
        'v_white': [0]*D, # conflicts= 1
        'v_white_overwrite': [0]*D, # conflicts= 1

        ## Setting the current index to the next unexplored neighbor or the parent if all is explored: 4lD+2l

        'set_current_back': [0]*l, # conflicts= 1
        'u_pi': [0]*l, # conflicts= 3
        'set_current': [[0]*l for _ in range(D)], # conflicts= 1
        'v_id': [[0]*l for _ in range(D)], # conflicts= 3
        'upload_v_id': [[0]*l for _ in range(D)], # conflicts= 1
        'u_current_delay': [[0]*l for _ in range(D)], # conflicts= 2

        ## Getting this node's distance and incrementing it and assigning to the next node to explore : 8lD+2l

        'u_dist': [0]*l, # conflicts= 3
        'update_v_dist_augend': [0]*l, # conflicts= 1
        'v_dist_augend': [[0]*l for _ in range(D)], # conflicts= 3
        'v_dist_addend': [[0]*l for _ in range(D)], # conflicts= 1
        'to_copy_dist_augend': [[0]*l for _ in range(D)], # conflicts= 1
        'v_dist_carry': [[0]*l for _ in range(D)], # conflicts= 2
        'v_dist_sum_counter': [[0]*(2*l) for _ in range(D)], # conflicts= 1
        'v_dist': [[0]*l for _ in range(D)], # conflicts= 4
        'upload_v_dist': [[0]*l for _ in range(D)], # conflicts= 1

        ## Getting this node's id and assigning it as parent to the next node to explore : 4lD

        'v_set_pi': [[0]*l for _ in range(D)], # conflicts= 1
        'u_id_asparent': [[0]*l for _ in range(D)], # conflicts= 4
        'v_pi': [[0]*l for _ in range(D)], # conflicts= 4
        'upload_v_pi': [[0]*l for _ in range(D)], # conflicts= 1

        ## Uploading this nodes color as well as the neighbours' colors : 2D+3

        'v_gray': [0]*D, # conflicts= 4
        'upload_v_gray': [0]*D, # conflicts= 1
        'u_gray': 0, # conflicts= 2
        'u_white': 0, # conflicts= 1
        'u_white_overwrite': 0, # conflicts= 1

        ## Uploading the next node id to be explored and a flag to declare outgoing message : 2l+1

        'u_current': [0]*l, # conflicts= 3
        'upload_u_current': [0]*l, # conflicts= 1
        'upload_message_flag': 0, # conflicts= 1

        ## Guide the placeholder content to appropriate slots : 4lS+6lDS+2DS+3S

        'u_stamp_placeholder': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'v_id_placeholder': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 2
        'v_dist_placeholder': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 2
        'v_pi_placeholder': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 2
        'v_gray_placeholder': [[0]*D for _ in range(num_slots)], # conflicts= 2
        'u_current_placeholder': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'message_flag_placeholder': [0]*num_slots, # conflicts= 2
        'determine_slot':[0]*num_slots, # conflicts= 1
        'determine_u_stamp_slot': [[0]*l for _ in range(num_slots)], # conflicts= 3
        'determine_v_id_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 3
        'determine_v_dist_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 3
        'determine_v_pi_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 3
        'determine_v_gray_slot': [[0]*D for _ in range(num_slots)], # conflicts= 3
        'determine_u_current_slot': [[0]*l for _ in range(num_slots)], # conflicts= 3
        'determine_message_flag_slot': [0]*num_slots, # conflicts= 3

        ## Preserve temp data till the new message check is complete : 5lS + 9lDS+ 3DS

        'temp_v_id_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 2
        'temp_v_dist_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 2
        'temp_v_pi_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 2
        'temp_v_gray_slot': [[0]*D for _ in range(num_slots)], # conflicts= 2
        'temp_u_current_slot': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'to_accept_temp_v_id': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 1
        'to_accept_temp_v_dist': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 1
        'to_accept_temp_v_pi': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 1
        'to_accept_temp_v_gray': [[0]*D for _ in range(num_slots)], # conflicts= 1
        'to_accept_temp_u_current': [[0]*l for _ in range(num_slots)], # conflicts= 1
        'overwrite_temp_v_id': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 3
        'overwrite_temp_v_dist': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 3
        'overwrite_temp_v_pi': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 3
        'overwrite_temp_v_gray': [[0]*D for _ in range(num_slots)], # conflicts= 3
        'overwrite_temp_u_current': [[0]*l for _ in range(num_slots)], # conflicts= 3
        'persistent_last_stamp': [[0]*l for _ in range(num_slots)], # conflicts= 3
        'overwrite_last_stamp': [[0]*l for _ in range(num_slots)], # conflicts= 2

        ## Check if the stamp is new : 7lS+S

        'temp_u_stamp_slot': [[0]*l for _ in range(num_slots)], # conflicts= 5
        'to_accept_temp_u_stamp': [[0]*l for _ in range(num_slots)], # conflicts= 1
        'overwrite_temp_u_stamp': [[0]*l for _ in range(num_slots)], # conflicts= 3
        'last_stamp': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'compare_last_stamp': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'new_message': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'old_message': [[0]*l for _ in range(num_slots)], # conflicts= 2
        'return_message_flag': [0]*num_slots, # conflicts= 1

        ## Take new messages from delays and put it into received messages as well as put it back in to placeholder: 4l(S+1)+6lD(S+1)+2D(S+1)+2(S+1)

        'delay_new_stamp': [[0]*l for _ in range(num_slots+1)], # conflicts= 3
        'delay_new_v_id': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # conflicts= 2
        'delay_new_v_dist': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # conflicts= 2
        'delay_new_v_pi': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # conflicts= 2
        'delay_new_v_gray': [[0]*D for _ in range(num_slots+1)], # conflicts= 2
        'delay_new_current': [[0]*l for _ in range(num_slots+1)], # conflicts= 2
        'delay_new_message_flag': [0]*(num_slots+1), # conflicts= 2
        'shutdown_stamp_delay': [[0]*l for _ in range(num_slots+1)], # conflicts= 1
        'shutdown_v_id_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # conflicts= 1
        'shutdown_v_dist_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # conflicts= 1
        'shutdown_v_pi_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # conflicts= 1
        'shutdown_v_gray_delay': [[0]*D for _ in range(num_slots+1)], # conflicts= 1
        'shutdown_current_delay': [[0]*l for _ in range(num_slots+1)], # conflicts= 1
        'shutdown_message_flag_delay': [0]*(num_slots+1), # conflicts= 1

        ## Check local ids with ids in the message and set update flags for the matching ones; size: 7lD+D+7l(D^2)+(D^2)+D+l+3
        'received_new_message': 0,# conflicts= 1
        'received_v_id': [[0]*l for _ in range(D)], # conflicts= 1
        'u_id2check_with_received_v_ids': [[0]*l for _ in range(D)], # conflicts= 3
        'check_u_id_with_received_v_ids': [[0]*l for _ in range(D)], # conflicts= 4
        'received_v_ids2check_with_v_ids': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 2
        'v_ids2check_with_received_v_ids': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 3
        'check_v_ids_with_received_v_ids': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 1
        'u_equals_received_v_id': [[0]*l for _ in range(D)], # conflicts= 3
        'u_is_received_v_id_counter': [[0]*l for _ in range(D)], # conflicts= 2
        'v_equals_received_v_id': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 3
        'v_is_received_v_id_counter': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 3
        'update_current_counter': [0]*(D+l+2), # conflicts= 1
        'u_id_comparison_counter':  [[0]*(l+1) for _ in range(D)], # conflicts= 1
        'v_id_comparison_counter':  [[[0]*(l+1) for _ in range(D)] for _ in range(D)], # conflicts= 1
        'reset_u_equals_v' :  [[0]*(l) for _ in range(D)], # conflicts= 1
        'reset_v_equals_v':  [[[0]*(l) for _ in range(D)] for _ in range(D)], # conflicts= 1

        ## Update the current node's distance, parent, white and gray flag: 10lD+5D+4l

        'received_v_dist': [[0]*l for _ in range(D)], # conflicts= 2
        'renew_received_v_dist': [[0]*l for _ in range(D)], # conflicts= 1
        'u_dist_update': [[0]*l for _ in range(D)], # conflicts= 1
        'received_v_dist4u': [[0]*l for _ in range(D)], # conflicts= 1
        'u_dist_delay': [[0]*l for _ in range(D)], # conflicts= 2
        'received_v_pi': [[0]*l for _ in range(D)], # conflicts= 2
        'renew_received_v_pi': [[0]*l for _ in range(D)], # conflicts= 1
        'u_pi_update': [[0]*l for _ in range(D)], # conflicts= 1
        'received_v_pi4u': [[0]*l for _ in range(D)], # conflicts= 1
        'u_pi_delay': [[0]*l for _ in range(D)], # conflicts= 1
        'received_v_gray': [0]*D, # conflicts= 2
        'renew_received_v_gray': [0]*D, # conflicts= 1
        'u_gray_update': [0]*D, # conflicts= 1
        'received_v_gray4u': [0]*D, # conflicts= 1
        'u_gray_delay': [0]*D, # conflicts= 2
        'received_stamp': [0]*l, # conflicts= 2
        'update_stamp': [0]*l, # conflicts= 1
        'received_current': [0]*l, # conflicts= 2
        'update_current': [0]*l, # conflicts= 1

        ## Update the adjacent nodes' distance, parent, and white and gray flags : 6l(D^2)+3(D^2)

        'v_dist_update': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 1
        'received_v_dist4v': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 1
        'v_dist_delay': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 2
        'v_pi_update': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 1
        'received_v_pi4v': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 1
        'v_pi_delay': [[[0]*l for _ in range(D)] for _ in range(D)], # conflicts= 2
        'v_gray_update': [[0]*D for _ in range(D)], # conflicts= 1
        'received_v_gray4v': [[0]*D for _ in range(D)], # conflicts= 1
        'v_gray_delay': [[0]*D for _ in range(D)], # conflicts= 2

        ## Copy incoming messages to temporary slots while checking if stamp is new # because this is message we move it to the end : 2lS + 3lDS+ DS+ S
        'u_stamp_slot': [[0]*l for _ in range(num_slots)], # conflicts= 1
        'v_id_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 1
        'v_dist_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 1
        'v_pi_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # conflicts= 1
        'v_gray_slot': [[0]*D for _ in range(num_slots)], # conflicts= 1
        'u_current_slot': [[0]*l for _ in range(num_slots)], # conflicts= 1
        'message_flag_slot': [0]*num_slots, # conflicts= 1


    }
    return nested_dict(3), sample


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


def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)


def get_dataset(l,D): # total number of instructions k'= 18 D^3 l + 6 D^3 + 31 D^2 l + 9 D^2 + 49 D l + 20 D + 51 l + 14
    num_slots = D**2+1
    T, Y = [], []
    # dfs samples
    ## Checking if the node id is equal to current :8l+1
    for i in range(l):

        x, y = get_sample(l,D)
        ##Input
        x['u_id'][i] = 1
        x['current'][i] = 1
        x['compare_current'][i] = 1
        ##Output
        if i == 0:
            y['id_equals'][i] = 1
            y['id_equals_counter'][i] = 1
            y['u_id'][i] = 1
            for k_val in range(D): y['u_id2check_with_received_v_ids'][k_val][i]=1
            for d_val in range(D): y['u_id_asparent'][d_val][i] = 1
        elif i> 0:
            y['id_equals'][i] = 1
            y['u_id'][i] = 1
            for k_val in range(D): y['u_id2check_with_received_v_ids'][k_val][i]=1
            for d_val in range(D): y['u_id_asparent'][d_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_id'][i] = 1
        x['current'][i] = 1
        x['compare_current'][i] = 0
        ##Output
        y['u_id'][i] = 1
        for k_val in range(D): y['u_id2check_with_received_v_ids'][k_val][i]=1
        for d_val in range(D): y['u_id_asparent'][d_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_id'][i] = 1
        x['current'][i] = 0
        x['compare_current'][i] = 1
        ##Output
        y['u_id'][i] = 1
        for k_val in range(D): y['u_id2check_with_received_v_ids'][k_val][i]=1
        for d_val in range(D): y['u_id_asparent'][d_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_id'][i] = 1
        x['current'][i] = 0
        x['compare_current'][i] = 0
        ##Output
        y['u_id'][i] = 1
        for k_val in range(D): y['u_id2check_with_received_v_ids'][k_val][i]= 1
        for d_val in range(D): y['u_id_asparent'][d_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_id'][i] = 0
        x['current'][i] = 0
        x['compare_current'][i] = 1
        ##Output
        if i == 0:
            y['id_equals'][i] = 1
            y['id_equals_counter'][i] = 1
        elif i>0:
            y['id_equals'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['id_equals'][i] = 1
        x['id_equals_counter'][i] = 0
        x['reset_is_equal'][i] = 0
        ##Output
        y['id_equals'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['id_equals'][i] = 1
        x['id_equals_counter'][i] = 1
        x['reset_is_equal'][i] = 0
        ##Output
        if i < l-1:
            y['id_equals_counter'][i+1] = 1
        elif i == l-1:
            for d_val in range(D+1): y['u_is_current'][d_val] = 1
            for i_val in range(l): y['accept_stamp'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(l+1):

        x, y = get_sample(l, D)
        ##Input
        x['comparison_counter'][j] = 1
        ##Output
        if j < l-1:  # j < l (since j is 0-indexed, l becomes l-1)
            y['comparison_counter'][j+1] = 1
        elif j == l-1:
            y['comparison_counter'][j+1] = 1
            for i_val in range(l):
                y['update_stamp'][i_val] = 1
        elif j == l:  # j = l+1 (0-indexed equivalent)
            for i_val in range(l):
                y['reset_is_equal'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Get the message time stamp value and increment it and activate end of loop flags    : 8l+D
    for i in range(l):
        x, y = get_sample(l,D)
        ##Input
        x['stamp'][i] = 1
        x['accept_stamp'][i]=1
        ##Output
        y['u_stamp_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_augend'][i] = 1
        x['u_stamp_addend'][i] = 0
        x['to_copy_stamp_augend'][i] = 0
        ##Output
        y['u_stamp_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_augend'][i] = 0
        x['u_stamp_addend'][i] = 1
        x['to_copy_stamp_augend'][i] = 0
        ##Output
        y['u_stamp_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_augend'][i] = 1
        x['u_stamp_addend'][i] = 1
        x['to_copy_stamp_augend'][i] = 0
        ##Output
        y['u_stamp_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_carry'][i] = 1
        ##Output
        if i < l-1:
            y['u_stamp_addend'][i+1] = 1
        elif i == l-1:
            y['u_stamp_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))


        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_augend'][i] = 1
        x['u_stamp_addend'][i] = 0
        x['to_copy_stamp_augend'][i] = 1
        ##Output
        for s_val in range(num_slots): y['u_stamp_placeholder'][s_val][i] = 1
        for s_val in range(num_slots): y['persistent_last_stamp'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))
    
    for d in range(D):
        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_addend_delay'][d] = 1
        ##Output
        if d < D-1:
            y['u_stamp_addend_delay'][d+1] = 1
        elif d == D-1:
            y['u_stamp_addend'][0] = 1 # =i, i in l
            y['u_stamp_sum_counter'][0] = 1 # =j, j in 2l
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(2*l):
        x, y = get_sample(l,D)
        ##Input
        x['u_stamp_sum_counter'][j] = 1
        ##Output
        if j < 2*l-1:
            y['u_stamp_sum_counter'][j+1] = 1
        elif j == 2*l-1:
            for i_val in range(l): y['to_copy_stamp_augend'][i_val] = 1
            for d_val in range(D): 
                for i_val in range(l): y['upload_v_pi'][d_val][i_val] = 1
            for i_val in range(l): y['upload_u_current'][i_val] = 1
            for d_val in range(D): 
                for i_val in range(l): y['upload_v_dist'][d_val][i_val] = 1
            for d_val in range(D): 
                for i_val in range(l): y['upload_v_id'][d_val][i_val] = 1
            # for d in range(D): y['upload_v_white'][d] = 1 ## temporarily disable because I am not sure we need it uploaded
            for d_val in range(D): y['upload_v_gray'][d_val] = 1
            for i_val in range(l):
                for s_val in range(num_slots): y['overwrite_last_stamp'][s_val][i_val] = 1
            y['upload_message_flag'] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Checking which neighbouring node should be attended and activate flags to update its features : 5D
    for d in range(D+1):
        x, y = get_sample(l,D)
        ##Input
        x['v_turn'][d] = 1
        x['u_is_current'][d] = 1
        ##Output
        if d < D:
            y['attend2v'][d] = 1
            y['v_turn'][d+1] = 1
        elif d == D:
            y['u_stamp_addend'][0] = 1 # =i, i in [l]
            y['u_stamp_sum_counter'][0] = 1 # =j, j in [2l]
            for i_val in range(l): y['set_current_back'][i_val] = 1 # i in [l]
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['v_turn'][d] = 1
        x['u_is_current'][d] = 0
        ##Output
        y['v_turn'][d] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for d in range(D):

        x, y = get_sample(l,D)
        ##Input
        x['attend2v'][d] = 1
        x['v_white'][d] = 1
        x['v_white_overwrite'][d] = 0
        ##Output
        y['v_gray'][d]=1
        for i_val in range(l): y['v_set_pi'][d][i_val]=1
        y['v_dist_addend'][d][0]=1 # =i, i in [l]
        y['v_dist_sum_counter'][d][0]=1 # =j, j in [2l]
        y['u_stamp_addend_delay'][d]=1
        for i_val in range(l): y['set_current'][d][i_val]=1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['attend2v'][d] = 1
        x['v_white'][d] = 0
        x['v_white_overwrite'][d] = 0
        ##Output
        y['u_is_current'][d+1]=1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['attend2v'][d] = 0
        x['v_white'][d] = 1
        x['v_white_overwrite'][d] = 0
        ##Output
        y['v_white'][d]=1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Setting the current index to the next unexplored neighbour or the parent if all is explored: 2l+4lD
    for i in range(l):

        x, y = get_sample(l,D)
        ##Input
        x['set_current_back'][i] = 1
        x['u_pi'][i] = 1
        ##Output
        y['u_current'][i] = 1
        y['u_pi'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['set_current_back'][i] = 0
        x['u_pi'][i] = 1
        ##Output
        y['u_pi'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l,D)
            ##Input
            x['set_current'][d][i] = 1
            x['v_id'][d][i] = 1
            x['upload_v_id'][d][i] = 0
            ##Output
            y['u_current_delay'][d][i] = 1
            y['v_id'][d][i] = 1
            for k_val in range(D): y['v_ids2check_with_received_v_ids'][d][k_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['set_current'][d][i] = 0
            x['v_id'][d][i] = 1
            x['upload_v_id'][d][i] = 0
            ##Output
            y['v_id'][d][i] = 1
            for k_val in range(D): y['v_ids2check_with_received_v_ids'][d][k_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['set_current'][d][i] = 0
            x['v_id'][d][i] = 1
            x['upload_v_id'][d][i] = 1
            ##Output
            y['v_id'][d][i] = 1
            for k_val in range(D): y['v_ids2check_with_received_v_ids'][d][k_val][i] = 1
            for s_val in range(num_slots): y['v_id_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['u_current_delay'][d][i] = 1
            ##Output
            if d < D-1:
                y['u_current_delay'][d+1][i] = 1
            elif d == D-1:
                y['u_current'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Getting this node's distance and incrementing it and assigning to the next node to explore : 2l+9lD
    for i in range(l):

        x, y = get_sample(l,D)
        ##Input
        x['u_dist'][i] = 1
        x['update_v_dist_augend'][i] = 1
        ##Output
        for d_val in range(D): y['v_dist_augend'][d_val][i] = 1
        y['u_dist'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_dist'][i] = 1
        x['update_v_dist_augend'][i] = 0
        ##Output
        y['u_dist'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))


    for i in range(l):
        for d in range(D):

            x, y = get_sample(l,D)
            ##Input
            x['v_dist_augend'][d][i] = 1
            x['v_dist_addend'][d][i] = 0
            x['to_copy_dist_augend'][d][i] = 0
            ##Output
            y['v_dist_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_dist_augend'][d][i] = 0
            x['v_dist_addend'][d][i] = 1
            x['to_copy_dist_augend'][d][i] = 0
            ##Output
            y['v_dist_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_dist_augend'][d][i] = 1
            x['v_dist_addend'][d][i] = 1
            x['to_copy_dist_augend'][d][i] = 0
            ##Output
            y['v_dist_carry'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_dist_carry'][d][i] = 1
            ##Output
            if i < l-1:
                y['v_dist_addend'][d][i+1] = 1
            elif i == l-1:
                y['v_dist_carry'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_dist_augend'][d][i] = 1
            x['v_dist_addend'][d][i] = 0
            x['to_copy_dist_augend'][d][i] = 1
            ##Output
            y['v_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_dist'][d][i] = 1
            x['upload_v_dist'][d][i] = 1
            ##Output
            y['v_dist'][d][i] = 1
            for s_val in range(num_slots): y['v_dist_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_dist'][d][i] = 1
            x['upload_v_dist'][d][i] = 0
            ##Output
            y['v_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for j in range(2*l):
        for d in range(D):
            x, y = get_sample(l,D)
            ##Input
            x['v_dist_sum_counter'][d][j] = 1
            ##Output
            if j < 2*l-1:
                y['v_dist_sum_counter'][d][j+1] = 1
            elif j == 2*l-1:
                for i_val in range(l): y['to_copy_dist_augend'][d][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Assigning the node's id as parent to the next node to explore: 3lD

    for i in range(l):
        for d in range(D):
            x, y = get_sample(l,D)
            ##Input
            x['v_set_pi'][d][i] = 1
            x['u_id_asparent'][d][i] = 1
            ##Output
            y['v_pi'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_pi'][d][i] = 1
            x['upload_v_pi'][d][i] = 0
            ##Output
            y['v_pi'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['v_pi'][d][i] = 1
            x['upload_v_pi'][d][i] = 1
            ##Output
            y['v_pi'][d][i] = 1
            for s_val in range(num_slots): y['v_pi_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Uploading this nodes color as well as the neighbours' colors: 2D+2

    for d in range(D):

        x, y = get_sample(l,D)
        ##Input
        x['v_gray'][d] = 1
        x['upload_v_gray'][d] = 1
        ##Output
        y['v_gray'][d] = 1
        for s_val in range(num_slots): y['v_gray_placeholder'][s_val][d] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['v_gray'][d] = 1
        x['upload_v_gray'][d] = 0
        ##Output
        y['v_gray'][d] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['u_gray'] = 1
    ##Output
    y['u_gray'] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['u_white'] = 1
    x['u_white_overwrite'] = 0
    ##Output
    y['u_white'] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    ## Uploading the next node id to be explored and a flag to declare outgoing message: 1+2l

    x, y = get_sample(l,D)
    ##Input
    x['upload_message_flag'] = 1
    ##Output
    for s in range(num_slots): y['message_flag_placeholder'][s] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    for i in range(l):

        x, y = get_sample(l,D)
        ##Input
        x['u_current'][i] = 1
        x['upload_u_current'][i] = 1
        ##Output
        for s_val in range(num_slots): y['u_current_placeholder'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['u_current'][i] = 1
        x['upload_u_current'][i] = 0
        ##Output
        y['u_current'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Guide the placeholder content to appropriate slots: 3S+2DS+6lDS+4lS

    for  s in  range(num_slots):

        x, y = get_sample(l,D)
        ##Input
        x['determine_slot'][s] = 1
        ##Output
        y['determine_slot'][s] = 1
        for d_val in range(D): y['determine_v_gray_slot'][s][d_val] = 1
        for i_val in range(l):
            for d_val in range(D): y['determine_v_dist_slot'][s][d_val][i_val]=1
        for i_val in range(l):
            for d_val in range(D): y['determine_v_pi_slot'][s][d_val][i_val]=1
        for i_val in range(l):
            for d_val in range(D): y['determine_v_id_slot'][s][d_val][i_val]=1
        for i_val in range(l): y['determine_u_current_slot'][s][i_val]=1
        for i_val in range(l): y['determine_u_stamp_slot'][s][i_val]=1
        y['determine_message_flag_slot'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))


    for d in range(D):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['determine_v_gray_slot'][s][d] = 1
            x['v_gray_placeholder'][s][d] = 0
            ##Output
            y['determine_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['determine_v_gray_slot'][s][d] = 1
            x['v_gray_placeholder'][s][d] = 1
            ##Output
            y['determine_v_gray_slot'][s][d] = 1
            y['v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):
            for s in range(num_slots):

                x, y = get_sample(l,D)
                ##Input
                x['determine_v_dist_slot'][s][d][i] = 1
                x['v_dist_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['determine_v_dist_slot'][s][d][i] = 1
                x['v_dist_placeholder'][s][d][i] = 1
                ##Output
                y['determine_v_dist_slot'][s][d][i] = 1
                y['v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['determine_v_pi_slot'][s][d][i] = 1
                x['v_pi_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['determine_v_pi_slot'][s][d][i] = 1
                x['v_pi_placeholder'][s][d][i] = 1
                ##Output
                y['determine_v_pi_slot'][s][d][i] = 1
                y['v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['determine_v_id_slot'][s][d][i] = 1
                x['v_id_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['determine_v_id_slot'][s][d][i] = 1
                x['v_id_placeholder'][s][d][i] = 1
                ##Output
                y['determine_v_id_slot'][s][d][i] = 1
                y['v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['determine_u_current_slot'][s][i] = 1
            x['u_current_placeholder'][s][i] = 0
            ##Output
            y['determine_u_current_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['determine_u_current_slot'][s][i] = 1
            x['u_current_placeholder'][s][i] = 1
            ##Output
            y['determine_u_current_slot'][s][i] = 1
            y['u_current_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['determine_u_stamp_slot'][s][i] = 1
            x['u_stamp_placeholder'][s][i] = 0
            ##Output
            y['determine_u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['determine_u_stamp_slot'][s][i] = 1
            x['u_stamp_placeholder'][s][i] = 1
            ##Output
            y['determine_u_stamp_slot'][s][i] = 1
            y['u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots):

        x, y = get_sample(l,D)
        ##Input
        x['determine_message_flag_slot'][s] = 1
        x['message_flag_placeholder'][s] = 0
        ##Output
        y['determine_message_flag_slot'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l,D)
        ##Input
        x['determine_message_flag_slot'][s] = 1
        x['message_flag_placeholder'][s] = 1
        ##Output
        y['determine_message_flag_slot'][s] = 1
        y['message_flag_slot'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))


    ## Copy incoming messages to temporary slots while checking if stamp is new: DS+3lDS+2lS+S

    for d in range(D):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['v_gray_slot'][s][d] = 1
            ##Output
            y['temp_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):
            for s in range(num_slots):

                x, y = get_sample(l,D)
                ##Input
                x['v_dist_slot'][s][d][i] = 1
                ##Output
                y['temp_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['v_pi_slot'][s][d][i] = 1
                ##Output
                y['temp_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['v_id_slot'][s][d][i] = 1
                ##Output
                y['temp_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['u_current_slot'][s][i] = 1
            ##Output
            y['temp_u_current_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['u_stamp_slot'][s][i] = 1
            ##Output
            y['temp_u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots):

        x, y = get_sample(l,D)
        ##Input
        x['message_flag_slot'][s] = 1
        ##Output
        y['compare_last_stamp'][s][l-1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Preserve temp data till the new message check is complete; number of instructions: 2DS+6lDS+4lS+S   

    for d in range(D):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['temp_v_gray_slot'][s][d] = 1
            x['to_accept_temp_v_gray'][s][d] = 0
            x['overwrite_temp_v_gray'][s][d] = 0
            ##Output
            y['temp_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['temp_v_gray_slot'][s][d] = 1
            x['to_accept_temp_v_gray'][s][d] = 1
            x['overwrite_temp_v_gray'][s][d] = 0
            ##Output
            y['delay_new_v_gray'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):
            for s in range(num_slots):

                x, y = get_sample(l,D)
                ##Input
                x['temp_v_dist_slot'][s][d][i] = 1
                x['to_accept_temp_v_dist'][s][d][i] = 0
                x['overwrite_temp_v_dist'][s][d][i] = 0
                ##Output
                y['temp_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['temp_v_dist_slot'][s][d][i] = 1
                x['to_accept_temp_v_dist'][s][d][i] = 1
                x['overwrite_temp_v_dist'][s][d][i] = 0
                ##Output
                y['delay_new_v_dist'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['temp_v_pi_slot'][s][d][i] = 1
                x['to_accept_temp_v_pi'][s][d][i] = 0
                x['overwrite_temp_v_pi'][s][d][i] = 0
                ##Output
                y['temp_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['temp_v_pi_slot'][s][d][i] = 1
                x['to_accept_temp_v_pi'][s][d][i] = 1
                x['overwrite_temp_v_pi'][s][d][i] = 0
                ##Output
                y['delay_new_v_pi'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['temp_v_id_slot'][s][d][i] = 1
                x['to_accept_temp_v_id'][s][d][i] = 0
                x['overwrite_temp_v_id'][s][d][i] = 0
                ##Output
                y['temp_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l,D)
                ##Input
                x['temp_v_id_slot'][s][d][i] = 1
                x['to_accept_temp_v_id'][s][d][i] = 1
                x['overwrite_temp_v_id'][s][d][i] = 0
                ##Output
                y['delay_new_v_id'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))


    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['temp_u_current_slot'][s][i] = 1
            x['to_accept_temp_u_current'][s][i] = 0
            x['overwrite_temp_u_current'][s][i] = 0
            ##Output
            y['temp_u_current_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['temp_u_current_slot'][s][i] = 1
            x['to_accept_temp_u_current'][s][i] = 1
            x['overwrite_temp_u_current'][s][i] = 0
            ##Output
            y['delay_new_current'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['persistent_last_stamp'][s][i] = 1
            x['overwrite_last_stamp'][s][i] = 0
            ##Output
            y['persistent_last_stamp'][s][i] = 1
            y['last_stamp'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['persistent_last_stamp'][s][i] = 1
            x['overwrite_last_stamp'][s][i] = 1
            ##Output
            y['last_stamp'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots):

        x, y = get_sample(l, D)
        ##Input
        x['return_message_flag'][s] = 1
        ##Output
        y['delay_new_message_flag'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    #Check if the stamp is new : 10lS

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 1
            x['temp_u_stamp_slot'][s][i] = 1
            x['last_stamp'][s][i] = 1
            x['to_accept_temp_u_stamp'][s][i] = 0
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            if i == 0:
                for i_val in range(l): y['overwrite_temp_u_stamp'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_current'][s][i_val] = 1
                for i_val in range(l):
                    for d in range(D): y['overwrite_temp_v_dist'][s][d][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_pi'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D): y['overwrite_temp_v_gray'][s][d_val] = 1
            elif i > 0:
                y['compare_last_stamp'][s][i-1] = 1
                y['temp_u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 1
            x['temp_u_stamp_slot'][s][i] = 0
            x['last_stamp'][s][i] = 0
            x['to_accept_temp_u_stamp'][s][i] = 0
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            if i == 0:
                for i_val in range(l): y['overwrite_temp_u_stamp'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_current'][s][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_dist'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_pi'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D): y['overwrite_temp_v_gray'][s][d_val] = 1
            elif i > 0:
                y['compare_last_stamp'][s][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 1
            x['temp_u_stamp_slot'][s][i] = 1
            x['last_stamp'][s][i] = 0
            x['to_accept_temp_u_stamp'][s][i] = 0
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            y['new_message'][s][i] = 1
            y['temp_u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_stamp'][s][i] = 1
            x['temp_u_stamp_slot'][s][i] = 0
            x['last_stamp'][s][i] = 1
            x['to_accept_temp_u_stamp'][s][i] = 0
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            y['old_message'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['old_message'][s][i] = 1
            ##Output
            if i > 0:  # i > 1 in 1-indexed becomes i > 0 in 0-indexed
                y['old_message'][s][i-1] = 1
            elif i == 0:  # i = 1 in 1-indexed becomes i = 0 in 0-indexed
                for i_val in range(l): y['overwrite_temp_u_stamp'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_current'][s][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_dist'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_pi'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['overwrite_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D): y['overwrite_temp_v_gray'][s][d_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['new_message'][s][i] = 1
            ##Output
            if i>0:
                y['new_message'][s][i-1] = 1
            elif i==0:
                for i_val in range(l): y['to_accept_temp_u_stamp'][s][i_val] = 1
                for i_val in range(l): y['to_accept_temp_u_current'][s][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['to_accept_temp_v_dist'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['to_accept_temp_v_pi'][s][d_val][i_val] = 1
                for i_val in range(l):
                    for d_val in range(D): y['to_accept_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D): y['to_accept_temp_v_gray'][s][d_val] = 1
                y['return_message_flag'][s] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 0
            x['temp_u_stamp_slot'][s][i] = 1
            x['last_stamp'][s][i] = 1
            x['to_accept_temp_u_stamp'][s][i] = 0
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            y['temp_u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 0
            x['temp_u_stamp_slot'][s][i] = 1
            x['last_stamp'][s][i] = 0
            x['to_accept_temp_u_stamp'][s][i] = 0
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            y['temp_u_stamp_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 0
            x['temp_u_stamp_slot'][s][i] = 1
            x['last_stamp'][s][i] = 0
            x['to_accept_temp_u_stamp'][s][i] = 1
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            y['delay_new_stamp'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,D)
            ##Input
            x['compare_last_stamp'][s][i] = 0
            x['temp_u_stamp_slot'][s][i] = 1
            x['last_stamp'][s][i] = 1
            x['to_accept_temp_u_stamp'][s][i] = 1
            x['overwrite_temp_u_stamp'][s][i] = 0
            ##Output
            y['delay_new_stamp'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Take new messages from delays and put it into received messages as well as put it back in to placeholder: 2l(S+1)+D(S+1)+3lD(S+1)+(S+1)

    for i in range(l):
        for s in range(num_slots+1):

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_stamp'][s][i] = 1
            x['shutdown_stamp_delay'][s][i] = 0
            ##Output
            if s < num_slots :  # num_slots = D^2 + 1
                y['delay_new_stamp'][s+1][i] = 1
            elif s == num_slots:  # s = D^2 + 1
                y['received_stamp'][i] = 1
                for s_val in range(num_slots):
                    y['u_stamp_placeholder'][s_val][i] = 1
                for s_val in range(num_slots): y['persistent_last_stamp'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))


            x, y = get_sample(l, D)
            ## Input
            x['delay_new_current'][s][i] = 1
            x['shutdown_current_delay'][s][i] = 0
            ## Output
            if s < num_slots:  # s < D²+1
                y['delay_new_current'][s+1][i] = 1
            elif s == num_slots:  # s = D²+1
                y['received_current'][i] = 1
                for s_val in range(num_slots): y['u_current_placeholder'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for d in range(D):
        for s in range(num_slots+1):
            x, y = get_sample(l, D)
            ## Input
            x['delay_new_v_gray'][s][d] = 1
            x['shutdown_v_gray_delay'][s][d] = 0
            ## Output
            if s < num_slots:  # s < D²+1 (0-indexed, so D**2 is the last index)
                y['delay_new_v_gray'][s+1][d] = 1
            elif s == num_slots:  # s = D²+1 (0-indexed, D**2 is the D²+1-th element)
                y['received_v_gray'][d] = 1
                for s_val in range(num_slots): y['v_gray_placeholder'][s_val][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):
            for s in range(num_slots+1):

                x, y = get_sample(l, D)
                ## Input
                x['delay_new_v_dist'][s][d][i] = 1
                x['shutdown_v_dist_delay'][s][d][i] = 0
                ## Output
                if s < num_slots:  # s < D²+1
                    y['delay_new_v_dist'][s+1][d][i] = 1
                elif s == num_slots:  # s = D²+1
                    y['received_v_dist'][d][i] = 1
                    for s_val in range(num_slots): y['v_dist_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ## Input
                x['delay_new_v_id'][s][d][i] = 1
                x['shutdown_v_id_delay'][s][d][i] = 0
                ## Output
                if s < num_slots:  # s < D²+1
                    y['delay_new_v_id'][s+1][d][i] = 1
                elif s == num_slots:  # s = D²+1
                    y['received_v_id'][d][i] = 1
                    for s_val in range(num_slots): y['v_id_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ## Input
                x['delay_new_v_pi'][s][d][i] = 1
                x['shutdown_v_pi_delay'][s][d][i] = 0
                ## Output
                if s < num_slots:  # s < D²+1
                    y['delay_new_v_pi'][s+1][d][i] = 1
                elif s == num_slots:  # s = D²+1
                    y['received_v_pi'][d][i] = 1
                    for s_val in range(num_slots): y['v_pi_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for s in range(num_slots+1):

        x, y = get_sample(l, D)
        ##Input
        x['delay_new_message_flag'][s] = 1
        x['shutdown_message_flag_delay'][s] = 0
        ##Output
        if s < num_slots - 1:  # s < D^2+1
            y['delay_new_message_flag'][s+1] = 1
        elif s == num_slots - 1:  # s = D^2+1
            y['delay_new_message_flag'][s+1] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['overwrite_last_stamp'][s_val][i_val] = 1
            for s_val in range(num_slots):  # All s < D^2+2
                y['shutdown_message_flag_delay'][s_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_stamp_delay'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for d_val in range(D):
                    y['shutdown_v_gray_delay'][s_val][d_val] = 1
            for s_val in range(num_slots):
                for d_val in range(D):
                    for i_val in range(l):
                        y['shutdown_v_dist_delay'][s_val][d_val][i_val] = 1
            for s_val in range(num_slots):
                for d_val in range(D):
                    for i_val in range(l):
                        y['shutdown_v_id_delay'][s_val][d_val][i_val] = 1
            for s_val in range(num_slots):
                for d_val in range(D):
                    for i_val in range(l):
                        y['shutdown_v_pi_delay'][s_val][d_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_current_delay'][s_val][i_val] = 1
        elif s == num_slots:  # s = D^2+2
            for s_val in range(num_slots):
                y['message_flag_placeholder'][s_val] = 1
            y['received_new_message'] = 1
            for k_val in range(D):  # Assuming k_max is defined
                for i_val in range(l):
                    y['check_u_id_with_received_v_ids'][k_val][i_val] = 1
            for k_val in range(D):
                y['u_id_comparison_counter'][k_val][0] = 1  # j=1 as specified
        T.append(x)
        Y.append(flatten_sample(y))

    ## Check local ids with ids in the message and set update flags for the matching ones: 3+6lD+2D+5l(D^2)+(D^2)+l

    x, y = get_sample(l, D)
    ##Input
    x['received_new_message'] = 1
    ##Output
    for d_val in range(D):
        for k_val in range(D):  # Assuming k range is D
            for i_val in range(l):
                y['check_v_ids_with_received_v_ids'][d_val][k_val][i_val] = 1
    for d_val in range(D):
        for k_val in range(D):  # Assuming k range is D
            y['v_id_comparison_counter'][d_val][k_val][0] = 1
    y['update_current_counter'][0] = 1 # 0=j in D+l+2
    T.append(x)
    Y.append(flatten_sample(y))

    for i in range(l):
        for k in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['received_v_id'][k][i] = 1
            x['u_id2check_with_received_v_ids'][k][i] = 1
            x['check_u_id_with_received_v_ids'][k][i] = 1
            ##Output
            if i == 0:  #  0-indexing, so i=1 becomes i=0
                y['u_equals_received_v_id'][k][i] = 1
                for d_val in range(D):
                    y['received_v_ids2check_with_v_ids'][d_val][k][i] = 1
                y['u_is_received_v_id_counter'][k][i] = 1
            elif i > 0:  # i > 1 becomes i > 0
                y['u_equals_received_v_id'][k][i] = 1
                for d_val in range(D):
                    y['received_v_ids2check_with_v_ids'][d_val][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_v_id'][k][i] = 0
            x['u_id2check_with_received_v_ids'][k][i] = 0
            x['check_u_id_with_received_v_ids'][k][i] = 1
            ##Output
            if i == 0:
                y['u_equals_received_v_id'][k][i] = 1
                y['u_is_received_v_id_counter'][k][i] = 1
            elif i > 0:
                y['u_equals_received_v_id'][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_v_id'][k][i] = 1
            x['u_id2check_with_received_v_ids'][k][i] = 0
            x['check_u_id_with_received_v_ids'][k][i] = 1
            ##Output
            for d_val in range(D):
                y['received_v_ids2check_with_v_ids'][d_val][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_equals_received_v_id'][k][i] = 1
            x['u_is_received_v_id_counter'][k][i] = 1
            x['reset_u_equals_v'][k][i] = 0
            ##Output
            if i < l - 1:
                y['u_is_received_v_id_counter'][k][i+1] = 1
            elif i == l - 1:
                for i_val in range(l):
                    y['u_dist_update'][k][i_val] = 1
                for i_val in range(l):
                    y['u_pi_update'][k][i_val] = 1
                y['u_gray_update'][k] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_equals_received_v_id'][k][i] = 1
            x['u_is_received_v_id_counter'][k][i] = 0
            x['reset_u_equals_v'][k][i] = 0
            ##Output
            y['u_equals_received_v_id'][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for j in range(l+1):
        for k in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['u_id_comparison_counter'][k][j] = 1
            ##Output
            if j < l:  # j < l+1 (since j is 0-indexed, l+1 becomes l)
                y['u_id_comparison_counter'][k][j+1] = 1
            elif j == l:  # j = l+1 (since j is 0-indexed, l+1 becomes l)
                for i_val in range(l):
                    y['reset_u_equals_v'][k][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for k in range(D):
            for d in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['received_v_ids2check_with_v_ids'][d][k][i] = 1
                x['v_ids2check_with_received_v_ids'][d][k][i] = 1
                x['check_v_ids_with_received_v_ids'][d][k][i] = 1
                ##Output
                if i == 0:
                    y['v_equals_received_v_id'][d][k][i] = 1
                    y['v_is_received_v_id_counter'][d][k][i] = 1
                elif i > 0:
                    y['v_equals_received_v_id'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['received_v_ids2check_with_v_ids'][d][k][i] = 0
                x['v_ids2check_with_received_v_ids'][d][k][i] = 0
                x['check_v_ids_with_received_v_ids'][d][k][i] = 1
                ##Output
                if i == 0:
                    y['v_equals_received_v_id'][d][k][i] = 1
                    y['v_is_received_v_id_counter'][d][k][i] = 1
                elif i > 0:
                    y['v_equals_received_v_id'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_equals_received_v_id'][d][k][i] = 1
                x['v_is_received_v_id_counter'][d][k][i] = 1
                x['reset_v_equals_v'][d][k][i]= 0
                ##Output
                if i < l - 1:
                    y['v_is_received_v_id_counter'][d][k][i+1] = 1
                elif i == l - 1:
                    for i_val in range(l):
                        y['v_dist_update'][d][k][i_val] = 1
                    for i_val in range(l):
                        y['v_pi_update'][d][k][i_val] = 1
                    y['v_gray_update'][d][k] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_equals_received_v_id'][d][k][i] = 1
                x['v_is_received_v_id_counter'][d][k][i] = 0
                x['reset_v_equals_v'][d][k][i]= 0
                ##Output
                y['v_equals_received_v_id'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for j in range(l+1):
        for k in range(D):
            for d in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['v_id_comparison_counter'][d][k][j] = 1
                ##Output
                if j < l:  # j < l+1 (since j is 0-indexed, l+1 becomes l)
                    y['v_id_comparison_counter'][d][k][j+1] = 1
                elif j == l:  # j = l+1 (since j is 0-indexed, l+1 becomes l)
                    for i_val in range(l):
                        y['reset_v_equals_v'][d][k][i_val] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for j in range(l+D+2):

        x, y = get_sample(l, D)
        ##Input
        x['update_current_counter'][j] = 1
        ##Output
        if j < l + D :  # j < l+D+1 (0-indexed: j < l+D)
            y['update_current_counter'][j+1] = 1
        elif j == l + D :  # j = l+D+1 (0-indexed: j = l+D)
            y['update_current_counter'][j+1] = 1
            for i_val in range(l):
                y['update_current'][i_val] = 1
        elif j == l + D + 1:  # j = l+D+2 (0-indexed: j = l+D+1)
            for i_val in range(l):
                y['compare_current'][i_val] = 1
            y['comparison_counter'][0]= 1
            for i_val in range(l):
                for k_val in range(D):
                    y['renew_received_v_dist'][k_val][i_val] = 1
            for i_val in range(l):
                for k_val in range(D):
                    y['renew_received_v_pi'][k_val][i_val] = 1
            for i_val in range(l):
                for k_val in range(D):
                    y['renew_received_v_gray'][k_val] = 1
            for i_val in range(l):
                y['update_v_dist_augend'][i_val] = 1

        T.append(x)
        Y.append(flatten_sample(y))

    ## Update the current node's distance, parent, white and gray flag: 6lD+3D+4l

    for i in range(l):
        for k in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['received_v_dist'][k][i] = 1
            x['renew_received_v_dist'][k][i] = 0
            ##Output
            y['received_v_dist'][k][i] = 1
            y['received_v_dist4u'][k][i] = 1
            for d_val in range(D):
                y['received_v_dist4v'][d_val][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_dist_update'][k][i] = 1
            x['received_v_dist4u'][k][i] = 1
            ##Output
            y['u_dist_delay'][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_dist_delay'][k][i] = 1
            ##Output
            if k < D - 1:
                y['u_dist_delay'][k+1][i] = 1
            elif k == D - 1:
                y['u_dist'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_v_pi'][k][i] = 1
            x['renew_received_v_pi'][k][i] = 0
            ##Output
            y['received_v_pi'][k][i] = 1
            y['received_v_pi4u'][k][i] = 1
            for d_val in range(D):
                y['received_v_pi4v'][d_val][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_pi_update'][k][i] = 1
            x['received_v_pi4u'][k][i] = 1
            ##Output
            y['u_pi_delay'][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_pi_delay'][k][i] = 1
            ##Output
            if k < D - 1:
                y['u_pi_delay'][k+1][i] = 1
            elif k == D - 1:
                y['u_pi'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for k in range(D):

        x, y = get_sample(l, D)
        ##Input
        x['received_v_gray'][k] = 1
        x['renew_received_v_gray'][k] = 0
        ##Output
        y['received_v_gray'][k] = 1
        y['received_v_gray4u'][k] = 1
        for d_val in range(D):
            y['received_v_gray4v'][d_val][k] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_gray_update'][k] = 1
        x['received_v_gray4u'][k] = 1
        ##Output
        y['u_gray_delay'][k] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_gray_delay'][k] = 1
        ##Output
        if k < D - 1:
            y['u_gray_delay'][k+1] = 1
        elif k == D - 1:
            y['u_gray'] = 1
            y['u_white_overwrite'] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['received_stamp'][i] = 1
        x['update_stamp'][i] = 0
        ##Output
        y['received_stamp'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_stamp'][i] = 1
        x['update_stamp'][i] = 1
        ##Output
        y['stamp'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_current'][i] = 1
        x['update_current'][i] = 0
        ##Output
        y['received_current'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_current'][i] = 1
        x['update_current'][i] = 1
        ##Output
        y['current'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Update the adjacent nodes' distance, parent, and white and gray flags: 4l(D^2)+ 2(D^2)

    for i in range(l):
        for k in range(D):
            for d in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['v_dist_update'][d][k][i] = 1
                x['received_v_dist4v'][d][k][i] = 1
                ##Output
                y['v_dist_delay'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_dist_delay'][d][k][i] = 1
                ##Output
                if k < D - 1:  # k < D
                    y['v_dist_delay'][d][k+1][i] = 1
                elif k == D - 1:  # k = D
                    y['v_dist'][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_pi_update'][d][k][i] = 1
                x['received_v_pi4v'][d][k][i] = 1
                ##Output
                y['v_pi_delay'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_pi_delay'][d][k][i] = 1
                ##Output
                if k < D - 1:  # k < D
                    y['v_pi_delay'][d][k+1][i] = 1
                elif k == D - 1:  # k = D
                    y['v_pi'][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for k in range(D):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['v_gray_update'][d][k] = 1
            x['received_v_gray4v'][d][k] = 1
            ##Output
            y['v_gray_delay'][d][k] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_gray_delay'][d][k] = 1
            ##Output
            if k < D - 1:  # k < D
                y['v_gray_delay'][d][k+1] = 1
            elif k == D - 1:  # k = D
                y['v_gray'][d] = 1
                y['v_white_overwrite'][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))







    
    
    return T, np.array(Y)

