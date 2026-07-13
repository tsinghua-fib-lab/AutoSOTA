import numpy as np
from .utils import flatten_sample, nested_dict
import csv

def get_sample(l,D):
    num_slots = D**2+1
    sample = { ## toal size of the vector: 8D^3+11D^2+22D+58l+84Dl+38D^2l+32D^3l+18

        ## Compare pointer and the current nodes priority to trigger the for loop; size: 7l+1
        'u_priority': [0] * (l),  # i ranges from 1 to l+1 # collisions = 4
        'q_pointer': [0] * (l), # collisions= 1
        'compare_u_priority': [0] * (l), # collisions= 1
        'priority_comparison_counter': [0] * (l + 1),  # j ranges from 1 to l+2 # collisions= 2
        'u_matches_pointer': [0] * (l),  # collisions= 3
        'u_matches_counter': [0] * (l), # collisions= 3
        'reset_u_matches': [0] * (l), # collisions= 1

        ## Set the loop parameters up ; size: 3D+1

        'u_matched': 0, # collisions= 1
        'v_turn': [0] * (D),  # d ∈ [D], index 1-based # collisions= 1
        'v_white': [0] * (D), # collisions= 1
        'v_white_overwrite': [0] * (D), # collisions= 1

        ## Assign priority to white nodes and increment it ; size: 10lD+2l

        'accept_last_in_q': [0] * l, # collisions= 1
        'u_last_in_q': [0] * l, # collisions= 1
        'v_set_priority': [[0] * l for _ in range(D)], # collisions= 1
        'v_last_in_q': [[0] * l for _ in range(D)], # collisions= 3
        'v_priority': [[0] * l for _ in range(D)], # collisions= 4
        'upload_v_priority': [[0] * l for _ in range(D)], # collisions= 1
        'v_last_in_q_augend': [[0] * l for _ in range(D)], # collisions= 4 for d=0, 3 otherwise
        'v_last_in_q_addend': [[0] * l for _ in range(D)], # collisions= 2
        'to_copy_last_in_q_augend': [[0] * l for _ in range(D)], # collisions= 1
        'v_last_in_q_carry': [[0] * l for _ in range(D)], # collisions= 2
        'v_last_in_q_sum_counter': [[0] * (2*l) for _ in range(D)],  # j ranges from 0 to 2l # collisions= 3

        ## Increment distance and assign white nodes the new distance; size: 6lD+8l

        'u_dist': [0]*l, # collisions= 3
        'update_v_dist_augend': [0]*l, # collisions= 1
        'v_dist_augend': [0]*l, # collisions= 3
        'v_dist_addend': [0]*l, # collisions= 1 (check for i=0)
        'to_copy_dist_augend': [0]*l, # collisions= 1
        'v_dist_carry': [0]*l, # collisions= 2
        'v_dist_sum_counter': [0]*(2*l),  # j ranges from 0 to 2l # collisions= 1
        'v_set_dist_counter': [[0] * (2*l) for _ in range(D)], # collisions= 1
        'v_set_dist': [[0]*l for _ in range(D)], # collisions= 1
        'v_dist_incremented': [[0]*l for _ in range(D)], # collisions= 2
        'v_dist': [[0]*l for _ in range(D)], # collisions= 4
        'upload_v_dist': [[0]*l for _ in range(D)], # collisions= 1

        ## Set the parent of white nodes equal to current node; size: 4lD+l

        'u_id': [0] * l, # collisions= 1
        'u_id_asparent': [[0] * l for _ in range(D)], # collisions= 1
        'v_set_pi': [[0] * l for _ in range(D)], # collisions= 1
        'v_pi': [[0] * l for _ in range(D)], # collisions= 4
        'upload_v_pi': [[0] * l for _ in range(D)], # collisions= 1

        ## Preserve this nodes colors and parent; size: l+3

        'u_white': 0,  # scalar value # collisions= 1
        'u_white_overwrite': 0,  # scalar value # collisions= 1
        'u_gray': 0,  # scalar value # collisions= 2
        'u_pi': [0]*l, # collisions= 2

        ## Preserve and upload gray; size: 2D

        'v_gray': [0]*D, # collisions= 4
        'upload_v_gray': [0]*D, # collisions= 

        ## Upload message flag; size: 1

        'upload_message_flag': 0, # collisions= 1

        ## Increment pointer and set upload flags; size: 8l

        'pointer_augend': [0] * l, # collisions= 4
        'pointer_addend': [0] * l, # collisions= 1
        'upload_pointer_augend': [0] * l, # collisions= 1
        'pointer_carry': [0] * l, # collisions= 2
        'pointer_sum_counter': [0] * (2*l),  # j goes from 0 to 2l # collisions= 1
        'after_loop_last_in_q': [0] * l, # collisions= 2
        'upload_after_loop_last_in_q': [0] * l, # collisions= 1

        ## Guide the placeholder content to appropriate sending slot; size: 3S+4lS+2DS+8lDS

        'determine_slot': [0] * num_slots, # collisions= 1
        'determine_pointer_slot': [[0] * l for _ in range(num_slots)], # collisions= 3
        'pointer_placeholder': [[0] * l for _ in range(num_slots)], # collisions= 2
        'determine_last_in_q_slot': [[0] * l for _ in range(num_slots)], # collisions= 3
        'last_in_q_placeholder': [[0] * l for _ in range(num_slots)], # collisions= 2
        'determine_v_gray_slot': [[0] * D for _ in range(num_slots)], # collisions= 3
        'v_gray_placeholder': [[0] * D for _ in range(num_slots)], # collisions= 2
        'determine_v_dist_slot': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'v_dist_placeholder': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'determine_v_pi_slot': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'v_pi_placeholder': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'determine_v_id_slot': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'v_id_placeholder': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'determine_v_priority_slot': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'v_priority_placeholder': [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'determine_message_flag_slot': [0] * num_slots, # collisions= 3
        'message_flag_placeholder': [0] * num_slots, # collisions= 2

        ## Preserve temp data till the new message check is complete; size: S+5lS+3DS+12lDS

        'persistent_last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 3
        'overwrite_last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2
        'temp_last_in_q_slot': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_last_in_q': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_last_in_q': [[0]*l for _ in range(num_slots)], # collisions= 3
        'temp_v_gray_slot': [[0]*D for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_v_gray': [[0]*D for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_v_gray': [[0]*D for _ in range(num_slots)], # collisions= 3
        'temp_v_dist_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_v_dist': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_v_dist': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'temp_v_id_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_v_id': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_v_id': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'temp_v_pi_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_v_pi': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_v_pi': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'temp_v_priority_slot': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_v_priority': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_v_priority': [[[0]*l for _ in range(D)] for _ in range(num_slots)], # collisions= 3
        'return_message_flag': [0]*num_slots, # collisions= 1

        ## Check if the pointer is new; size: 7lS

        'compare_last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2 (check for i=l-1)
        'temp_pointer_slot': [[0]*l for _ in range(num_slots)], # collisions= 5
        'last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_pointer': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_pointer': [[0]*l for _ in range(num_slots)], # collisions= 3
        'new_message': [[0]*l for _ in range(num_slots)], # collisions= 2
        'old_message': [[0]*l for _ in range(num_slots)], # collisions= 2
        
        ## Take new messages from delays and put it into received messages as well as put it back in to placeholder; size: 4l(S+1)+2D(S+1)+8lD(S+1)+2(S+1)

        'delay_new_last_in_q': [[0]*l for _ in range(num_slots+1)], # collisions= 2
        'shutdown_last_in_q_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_pointer': [[0]*l for _ in range(num_slots+1)], # collisions= 3
        'shutdown_pointer_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_v_gray': [[0]*D for _ in range(num_slots+1)], # collisions= 2
        'shutdown_v_gray_delay': [[0]*D for _ in range(num_slots+1)], # collisions= 1
        'delay_new_v_priority': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 2
        'shutdown_v_priority_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 1
        'delay_new_v_dist': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 2
        'shutdown_v_dist_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 1
        'delay_new_v_id': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 2
        'shutdown_v_id_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 1
        'delay_new_v_pi': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 2
        'shutdown_v_pi_delay': [[[0]*l for _ in range(D)] for _ in range(num_slots+1)], # collisions= 1
        'delay_new_message_flag': [0] * (num_slots+1), # collisions= 2
        'shutdown_message_flag_delay': [0] * (num_slots+1), # collisions= 1

        ## Check local ids with ids in the message and set update flags for the matching ones ; size: l+2D+9lD+7l(D^2)+(D^2)+3

        'v_id': [[0]*l for _ in range(D)], # collisions= 2
        'upload_v_id': [[0]*l for _ in range(D)], # collisions= 1
        'received_new_message': 0, # collisions= 1
        'received_v_id': [[0]*l for _ in range(D)], # collisions= 1
        'u_id2check_with_received_v_ids': [[0]*l for _ in range(D)], # collisions= 1
        'check_u_id_with_received_v_ids': [[0]*l for _ in range(D)], # collisions= 1
        'received_v_ids2check_with_v_ids': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 2
        'v_ids2check_with_received_v_ids': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 2
        'check_v_ids_with_received_v_ids': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'u_equals_received_v_id': [[0]*l for _ in range(D)], # collisions= 3
        'u_is_received_v_id_counter': [[0]*l for _ in range(D)], # collisions= 3
        'reset_u_equals_v': [[0]*l for _ in range(D)], # collisions= 1
        'v_equals_received_v_id': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 3
        'v_is_received_v_id_counter': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 3
        'reset_v_equals_v': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'u_id_comparison_counter': [[0]*(l+1) for _ in range(D)], # collisions= 1
        'v_id_comparison_counter': [[[0]*(l+1) for _ in range(D)] for _ in range(D)], # collisions= 1
        'update_pointer_counter': [0]*(D+l+2), # collisions= 1

        ## Update the current node's distance, priority parent, white and gray flag; size: 15lD+5D+4l

        'received_v_priority': [[0]*l for _ in range(D)], # collisions= 2
        'renew_received_v_priority': [[0]*l for _ in range(D)], # collisions= 1
        'received_v_priority4u': [[0]*l for _ in range(D)], # collisions= 1
        'u_priority_update': [[0]*l for _ in range(D)], # collisions= 1
        'u_priority_delay': [[0]*l for _ in range(D)], # collisions= 2
        'received_v_dist': [[0]*l for _ in range(D)], # collisions= 2
        'renew_received_v_dist': [[0]*l for _ in range(D)], # collisions= 1
        'received_v_dist4u': [[0]*l for _ in range(D)], # collisions= 1
        'u_dist_update': [[0]*l for _ in range(D)], # collisions= 1
        'u_dist_delay': [[0]*l for _ in range(D)], # collisions= 2
        'received_v_pi': [[0]*l for _ in range(D)], # collisions= 2
        'renew_received_v_pi': [[0]*l for _ in range(D)], # collisions= 1
        'received_v_pi4u': [[0]*l for _ in range(D)], # collisions= 1
        'u_pi_update': [[0]*l for _ in range(D)], # collisions= 1
        'u_pi_delay': [[0]*l for _ in range(D)], # collisions= 2
        'received_v_gray': [0]*D, # collisions= 2
        'renew_received_v_gray': [0]*D, # collisions= 1
        'received_v_gray4u': [0]*D, # collisions= 1
        'u_gray_update': [0]*D, # collisions= 1
        'u_gray_delay': [0]*D, # collisions= 2
        'received_pointer': [0]*l, # collisions= 2
        'update_pointer': [0]*l, # collisions= 1
        'received_last_in_q': [0]*l, # collisions= 2
        'update_last_in_q': [0]*l, # collisions= 1

        ## Update the adjacent nodes' distance, parent, and white and gray flags ; size: 9l(D^2)+3(D^2)

        'v_priority_update': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'received_v_priority4v': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'v_priority_delay': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 2
        'v_dist_update': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'received_v_dist4v': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'v_dist_delay': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 2
        'v_pi_update': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'received_v_pi4v': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 1
        'v_pi_delay': [[[0]*l for _ in range(D)] for _ in range(D)], # collisions= 2
        'v_gray_update': [[0]*D for _ in range(D)], # collisions= 1
        'received_v_gray4v': [[0]*D for _ in range(D)], # collisions= 1
        'v_gray_delay': [[0]*D for _ in range(D)], # collisions= 2

        ## Copy incoming messages to temporary slots while checking if the pointer is new; size: 2lS+DS+4lDS+S

        'pointer_slot' : [[0] * l for _ in range(num_slots)], # collisions= 1
        'last_in_q_slot' : [[0] * l for _ in range(num_slots)], # collisions= 1
        'v_gray_slot' :  [[0] * D for _ in range(num_slots)], # collisions= 1
        'v_dist_slot' : [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'v_pi_slot' : [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'v_id_slot' : [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'v_priority_slot' : [[[0] * l for _ in range(D)] for _ in range(num_slots)], # collisions= 1
        'message_flag_slot' : [0] * num_slots, # collisions= 1
        


    }
    return nested_dict(3), sample


def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)


def get_dataset(l,D): ## total number of instructions is: 6D^3+9D^2+17D+57l+65Dl+33D^2l+24D^3l+15
    num_slots = D**2+1
    T, Y = [], []

    ## Compare pointer and the current nodes priority to trigger the for loop; number of instructions: 7*l+1
    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['u_priority'][i] = 1
        x['q_pointer'][i] = 1
        x['compare_u_priority'][i] = 1
        ##Output
        if i == 0:  # i=1 in 1-indexed, i=0 in 0-indexed
            y['u_matches_pointer'][i] = 1
            y['u_matches_counter'][i] = 1
            y['u_priority'][i] = 1
            y['pointer_augend'][i] = 1
        elif i > 0:  # i>1 in 1-indexed, i>0 in 0-indexed
            y['u_matches_pointer'][i] = 1
            y['u_priority'][i] = 1
            y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_priority'][i] = 0
        x['q_pointer'][i] = 0
        x['compare_u_priority'][i] = 1
        ##Output
        if i == 0:  # i=1 in 1-indexed
            y['u_matches_pointer'][i] = 1
            y['u_matches_counter'][i] = 1
        elif i > 0:  # i>1 in 1-indexed
            y['u_matches_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_priority'][i] = 1
        x['q_pointer'][i] = 0
        x['compare_u_priority'][i] = 1
        ##Output
        y['u_priority'][i] = 1
        y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_priority'][i] = 1
        x['q_pointer'][i] = 0
        x['compare_u_priority'][i] = 0
        ##Output
        y['u_priority'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_matches_pointer'][i] = 1
        x['u_matches_counter'][i] = 1
        x['reset_u_matches'][i] = 0
        ##Output
        if i < l - 1:  # i < l
            y['u_matches_counter'][i+1] = 1
        elif i == l - 1:  # i = l
            y['u_matched'] = 1
            for i_val in range(l): y['accept_last_in_q'][i_val] = 1
            for i_val in range(l): y['update_v_dist_augend'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_matches_pointer'][i] = 1
        x['u_matches_counter'][i] = 0
        x['reset_u_matches'][i] = 0
        ##Output
        y['u_matches_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in  range(l+1):

        x, y = get_sample(l, D)
        ##Input
        x['priority_comparison_counter'][j] = 1
        ##Output
        if j < l - 1:  # j < l
            y['priority_comparison_counter'][j+1] = 1
        elif j == l - 1:  # j = l
            y['priority_comparison_counter'][j+1] = 1
            for i_val in range(l): y['update_last_in_q'][i_val] = 1
        elif j == l:  # j = l+1 (0-indexed j equals l, since l+1 in 1-indexed = l in 0-indexed)
            for i_val in range(l): y['reset_u_matches'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Set the loop parameters up ; number of instructions: 3*D+1

    x, y = get_sample(l, D)
    ##Input
    x['u_matched'] = 1
    ##Output
    y['v_turn'][0] = 1  # d=1 (0-indexed)
    y['v_dist_addend'][0] = 1  # d=1 (0-indexed)
    y['v_dist_sum_counter'][0] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    for d in range(D):
        x, y = get_sample(l, D)
        ##Input
        x['v_turn'][d] = 1
        x['v_white'][d] = 1
        x['v_white_overwrite'][d] = 0
        ##Output
        y['v_gray'][d] = 1
        y['v_last_in_q_sum_counter'][d][0] = 1  # i=1 (0-indexed)
        y['v_last_in_q_addend'][d][0] = 1  # i=1 (0-indexed)
        for i_val in range(l): y['v_set_pi'][d][i_val] = 1
        for i_val in range(l): y['v_set_priority'][d][i_val] = 1
        y['v_set_dist_counter'][d][0] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_turn'][d] = 0
        x['v_white'][d] = 1
        x['v_white_overwrite'][d] = 0
        ##Output
        y['v_white'][d] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_turn'][d] = 1
        x['v_white'][d] = 0
        x['v_white_overwrite'][d] = 0
        ##Output
        y['v_last_in_q_sum_counter'][d][0] = 1  # i=1 (0-indexed)
        T.append(x)
        Y.append(flatten_sample(y))

    ## Assign priority to white nodes and increment it; number of instructions: 11*l*D+l

    for i in range(l):
        # Instruction 1
        x, y = get_sample(l, D)
        ##Input
        x['accept_last_in_q'][i] = 1
        x['u_last_in_q'][i] = 1
        ##Output
        y['v_last_in_q'][0][i] = 1
        y['v_last_in_q_augend'][0][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for d in range(D):
        for i in range(l):

            x, y = get_sample(l, D)
            ##Input
            x['v_set_priority'][d][i] = 1
            x['v_last_in_q'][d][i] = 1
            ##Output
            y['v_priority'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_set_priority'][d][i] = 0
            x['v_last_in_q'][d][i] = 1
            ##Output
            y['v_last_in_q'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_priority'][d][i] = 1
            x['upload_v_priority'][d][i] = 0
            ##Output
            y['v_priority'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_priority'][d][i] = 1
            x['upload_v_priority'][d][i] = 1
            ##Output
            y['v_priority'][d][i] = 1
            for s_val in range(num_slots):
                y['v_priority_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_last_in_q_augend'][d][i] = 1
            x['v_last_in_q_addend'][d][i] = 0
            x['to_copy_last_in_q_augend'][d][i] = 0
            ##Output
            y['v_last_in_q_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_last_in_q_augend'][d][i] = 1
            x['v_last_in_q_addend'][d][i] = 1
            x['to_copy_last_in_q_augend'][d][i] = 0
            ##Output
            y['v_last_in_q_carry'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_last_in_q_augend'][d][i] = 0
            x['v_last_in_q_addend'][d][i] = 1
            x['to_copy_last_in_q_augend'][d][i] = 0
            ##Output
            y['v_last_in_q_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_last_in_q_carry'][d][i] = 1
            ##Output
            if i < l - 1:
                y['v_last_in_q_addend'][d][i+1] = 1
            elif i == l - 1:
                y['v_last_in_q_carry'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_last_in_q_augend'][d][i] = 1
            x['v_last_in_q_addend'][d][i] = 0
            x['to_copy_last_in_q_augend'][d][i] = 1
            ##Output
            if d < D - 1:
                y['v_last_in_q'][d+1][i] = 1
                y['v_last_in_q_augend'][d+1][i] = 1
            elif d == D - 1:
                y['after_loop_last_in_q'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for d in range(D):
        for j in range(2*l):
            x, y = get_sample(l, D)
            ##Input
            x['v_last_in_q_sum_counter'][d][j] = 1
            ##Output
            if j < 2 * l - 1:
                y['v_last_in_q_sum_counter'][d][j+1] = 1
            elif j == 2 * l - 1:
                if d < D - 1:
                    for i_val in range(l):
                        y['to_copy_last_in_q_augend'][d][i_val] = 1
                    y['v_turn'][d+1] = 1
                elif d == D - 1:
                    for i_val in range(l):
                        y['to_copy_last_in_q_augend'][d][i_val] = 1
                    y['pointer_addend'][0] = 1
                    y['pointer_sum_counter'][0] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Increment distance and assign white nodes the new distance; number of instructions: 9*l+6*D*l

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['u_dist'][i] = 1
        x['update_v_dist_augend'][i] = 0
        ##Output
        y['u_dist'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_dist'][i] = 1
        x['update_v_dist_augend'][i] = 1
        ##Output
        y['v_dist_augend'][i] = 1
        y['u_dist'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_dist_augend'][i] = 1
        x['v_dist_addend'][i] = 0
        x['to_copy_dist_augend'][i] = 0
        ##Output
        y['v_dist_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_dist_augend'][i] = 0
        x['v_dist_addend'][i] = 1
        x['to_copy_dist_augend'][i] = 0
        ##Output
        y['v_dist_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_dist_augend'][i] = 1
        x['v_dist_addend'][i] = 1
        x['to_copy_dist_augend'][i] = 0
        ##Output
        y['v_dist_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_dist_carry'][i] = 1
        ##Output
        if i < l - 1:
            y['v_dist_addend'][i+1] = 1
        if i == l - 1:
            y['v_dist_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_dist_augend'][i] = 1
        x['v_dist_addend'][i] = 0
        x['to_copy_dist_augend'][i] = 1
        ##Output
        for d_val in range(D):
            y['v_dist_incremented'][d_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(2*l):

        x, y = get_sample(l, D)
        ##Input
        x['v_dist_sum_counter'][j] = 1
        ##Output
        if j < 2 * l - 1:
            y['v_dist_sum_counter'][j+1] = 1
        elif j == 2 * l - 1:
            for i_val in range(l):
                y['to_copy_dist_augend'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(2*l):
        for d in range(D):
            x, y = get_sample(l, D)
            ##Input
            x['v_set_dist_counter'][d][j] = 1
            ##Output
            if j < 2 * l - 1:
                y['v_set_dist_counter'][d][j+1] = 1
            elif j == 2 * l - 1:
                for i_val in range(l):
                    y['v_set_dist'][d][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for d in range(D):
        for i in range(l):

            x, y = get_sample(l, D)
            ##Input
            x['v_set_dist'][d][i] = 1
            x['v_dist_incremented'][d][i] = 1
            ##Output
            y['v_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_set_dist'][d][i] = 0
            x['v_dist_incremented'][d][i] = 1
            ##Output
            y['v_dist_incremented'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_dist'][d][i] = 1
            x['upload_v_dist'][d][i] = 1
            ##Output
            y['v_dist'][d][i] = 1
            for s_val in range(num_slots):
                y['v_dist_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_dist'][d][i] = 1
            x['upload_v_dist'][d][i] = 0
            ##Output
            y['v_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Set the parent of white nodes equal to current node; number of instructions: l+3*D*l

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['u_id'][i] = 1
        ##Output
        for d_val in range(D): y['u_id_asparent'][d_val][i] = 1
        y['u_id'][i] = 1
        for d_val in range(D): y['u_id2check_with_received_v_ids'][d_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for d in range(D):
        for i in range(l):

            x, y = get_sample(l, D)
            ##Input
            x['v_set_pi'][d][i] = 1
            x['u_id_asparent'][d][i] = 1
            ##Output
            y['v_pi'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_pi'][d][i] = 1
            x['upload_v_pi'][d][i] = 0
            ##Output
            y['v_pi'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_pi'][d][i] = 1
            x['upload_v_pi'][d][i] = 1
            ##Output
            y['v_pi'][d][i] = 1
            for s_val in range(num_slots): y['v_pi_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Preserve this nodes color and parent; number of instructions: l+2

    x, y = get_sample(l, D)
    ##Input
    x['u_white'] = 1
    x['u_white_overwrite'] = 0
    ##Output
    y['u_white'] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['u_gray'] = 1
    ##Output
    y['u_gray'] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    for i in range(l):
        x, y = get_sample(l, D)
        ##Input
        x['u_pi'][i] = 1
        ##Output
        y['u_pi'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Preserve and upload gray; number of instructions:2*D

    for d in range(D):

        x, y = get_sample(l, D)
        ##Input
        x['v_gray'][d] = 1
        x['upload_v_gray'][d] = 0
        ##Output
        y['v_gray'][d] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['v_gray'][d] = 1
        x['upload_v_gray'][d] = 1
        ##Output
        y['v_gray'][d] = 1
        for s_val in range(num_slots):  # num_slots = D^2 + 1
            y['v_gray_placeholder'][s_val][d] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Upload message flag; number of instructions: 1 

    x, y = get_sample(l, D)
    ##Input
    x['upload_message_flag'] = 1
    ##Output
    for s_val in range(num_slots):
        y['message_flag_placeholder'][s_val] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    ## Increment pointer and set upload flags; number of instructions: 9*l

    for i in range(l):
        x, y = get_sample(l, D)
        x['pointer_augend'][i] = 1
        x['pointer_addend'][i] = 0
        x['upload_pointer_augend'][i] = 0
        y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))
        
        x, y = get_sample(l, D)
        x['pointer_augend'][i] = 0
        x['pointer_addend'][i] = 1
        x['upload_pointer_augend'][i] = 0
        y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))
        
        x, y = get_sample(l, D)
        x['pointer_augend'][i] = 1
        x['pointer_addend'][i] = 1
        x['upload_pointer_augend'][i] = 0
        y['pointer_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        x['pointer_carry'][i] = 1
        if i < l - 1:
            y['pointer_addend'][i+1] = 1
        elif i == l - 1:
            y['pointer_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(2*l):  # j ranges from 0 to 2l

        x, y = get_sample(l, D)
        x['pointer_sum_counter'][j] = 1
        if j < 2*l-1:
            y['pointer_sum_counter'][j+1] = 1
        elif j == 2*l-1:
            for i_val in range(l): y['upload_pointer_augend'][i_val] = 1
            for d_val in range(D): y['upload_v_gray'][d_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['upload_v_pi'][d_val][i_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['upload_v_dist'][d_val][i_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['upload_v_id'][d_val][i_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['upload_v_priority'][d_val][i_val] = 1
            for i_val in range(l): y['upload_after_loop_last_in_q'][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l): y['overwrite_last_pointer'][s_val][i_val] = 1
            y['upload_message_flag'] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(l):
        x, y = get_sample(l, D)
        x['pointer_augend'][i] = 1
        x['pointer_addend'][i] = 0
        x['upload_pointer_augend'][i] = 1
        for s_val in range(num_slots):
            y['pointer_placeholder'][s_val][i] = 1
        for s_val in range(num_slots):
            y['persistent_last_pointer'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        x['after_loop_last_in_q'][i] = 1
        x['upload_after_loop_last_in_q'][i] = 1
        for s_val in range(num_slots):
            y['last_in_q_placeholder'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))
        
        x, y = get_sample(l, D)
        x['after_loop_last_in_q'][i] = 1
        x['upload_after_loop_last_in_q'][i] = 0
        y['after_loop_last_in_q'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Guide the placeholder content to appropriate sending slot; number of instructions: 3*S+4*S*l+2*S*D+8*S*D*l

    for s in range(num_slots):

        x, y = get_sample(l, D)
        ##Input
        x['determine_slot'][s] = 1
        ##Output
        y['determine_slot'][s] = 1
        for d_val in range(D): y['determine_v_gray_slot'][s][d_val] = 1
        for d_val in range(D):
            for i_val in range(l): y['determine_v_dist_slot'][s][d_val][i_val] = 1
        for d_val in range(D):
            for i_val in range(l): y['determine_v_pi_slot'][s][d_val][i_val] = 1
        for d_val in range(D):
            for i_val in range(l): y['determine_v_id_slot'][s][d_val][i_val] = 1
        for d_val in range(D):
            for i_val in range(l): y['determine_v_priority_slot'][s][d_val][i_val] = 1
        for i_val in range(l): y['determine_pointer_slot'][s][i_val] = 1
        for i_val in range(l): y['determine_last_in_q_slot'][s][i_val] = 1
        y['determine_message_flag_slot'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for s in range(num_slots):
        for i in range(l):

            x, y = get_sample(l, D)
            ##Input
            x['determine_pointer_slot'][s][i] = 1
            x['pointer_placeholder'][s][i] = 1
            ##Output
            y['pointer_slot'][s][i] = 1
            y['determine_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_pointer_slot'][s][i] = 1
            x['pointer_placeholder'][s][i] = 0
            ##Output
            y['determine_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_last_in_q_slot'][s][i] = 1
            x['last_in_q_placeholder'][s][i] = 1
            ##Output
            y['last_in_q_slot'][s][i] = 1
            y['determine_last_in_q_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_last_in_q_slot'][s][i] = 1
            x['last_in_q_placeholder'][s][i] = 0
            ##Output
            y['determine_last_in_q_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['determine_v_gray_slot'][s][d] = 1
            x['v_gray_placeholder'][s][d] = 1
            ##Output
            y['v_gray_slot'][s][d] = 1
            y['determine_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_v_gray_slot'][s][d] = 1
            x['v_gray_placeholder'][s][d] = 0
            ##Output
            y['determine_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots):
        for d in range(D):
            for i in range(l):

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_dist_slot'][s][d][i] = 1
                x['v_dist_placeholder'][s][d][i] = 1
                ##Output
                y['v_dist_slot'][s][d][i] = 1
                y['determine_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_dist_slot'][s][d][i] = 1
                x['v_dist_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_pi_slot'][s][d][i] = 1
                x['v_pi_placeholder'][s][d][i] = 1
                ##Output
                y['v_pi_slot'][s][d][i] = 1
                y['determine_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_pi_slot'][s][d][i] = 1
                x['v_pi_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_id_slot'][s][d][i] = 1
                x['v_id_placeholder'][s][d][i] = 1
                ##Output
                y['v_id_slot'][s][d][i] = 1
                y['determine_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_id_slot'][s][d][i] = 1
                x['v_id_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_priority_slot'][s][d][i] = 1
                x['v_priority_placeholder'][s][d][i] = 1
                ##Output
                y['v_priority_slot'][s][d][i] = 1
                y['determine_v_priority_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['determine_v_priority_slot'][s][d][i] = 1
                x['v_priority_placeholder'][s][d][i] = 0
                ##Output
                y['determine_v_priority_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for s in range(num_slots):

        x, y = get_sample(l, D)
        ##Input
        x['determine_message_flag_slot'][s] = 1
        x['message_flag_placeholder'][s] = 1
        ##Output
        y['message_flag_slot'][s] = 1
        y['determine_message_flag_slot'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['determine_message_flag_slot'][s] = 1
        x['message_flag_placeholder'][s] = 0
        ##Output
        y['determine_message_flag_slot'][s] = 1
        T.append(x)
        Y.append(flatten_sample(y))


    ## Copy incoming messages to temporary slots while checking if the pointer is new; number of instructions: 2*l*S+D*S+4*l*D*S+S

    for i in range(l):
        for s in range(num_slots):
            x, y = get_sample(l, D)
            ##Input
            x['pointer_slot'][s][i] = 1
            ##Output
            y['temp_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['last_in_q_slot'][s][i] = 1
            ##Output
            y['temp_last_in_q_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for d in range(D):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['v_gray_slot'][s][d] = 1
            ##Output
            y['temp_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):
            for s in range(num_slots):

                x, y = get_sample(l, D)
                ##Input
                x['v_dist_slot'][s][d][i] = 1
                ##Output
                y['temp_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_id_slot'][s][d][i] = 1
                ##Output
                y['temp_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_pi_slot'][s][d][i] = 1
                ##Output
                y['temp_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_priority_slot'][s][d][i] = 1
                ##Output
                y['temp_v_priority_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for s in range(num_slots):

        x, y = get_sample(l, D)
        ##Input
        x['message_flag_slot'][s] = 1
        ##Output
        y['compare_last_pointer'][s][l-1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Preserve temp data till the new message check is complete; number of instructions: 4*l*S+2*D*S+8*l*D*S+S

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['persistent_last_pointer'][s][i] = 1
            x['overwrite_last_pointer'][s][i] = 0
            ##Output
            y['last_pointer'][s][i] = 1
            y['persistent_last_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['persistent_last_pointer'][s][i] = 1
            x['overwrite_last_pointer'][s][i] = 1
            ##Output
            y['last_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_last_in_q_slot'][s][i] = 1
            x['to_accept_temp_last_in_q'][s][i] = 0
            x['overwrite_temp_last_in_q'][s][i] = 0
            ##Output
            y['temp_last_in_q_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_last_in_q_slot'][s][i] = 1
            x['to_accept_temp_last_in_q'][s][i] = 1
            x['overwrite_temp_last_in_q'][s][i] = 0
            ##Output
            y['delay_new_last_in_q'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for d in range(D):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['temp_v_gray_slot'][s][d] = 1
            x['to_accept_temp_v_gray'][s][d] = 0
            x['overwrite_temp_v_gray'][s][d] = 0
            ##Output
            y['temp_v_gray_slot'][s][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
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

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_dist_slot'][s][d][i] = 1
                x['to_accept_temp_v_dist'][s][d][i] = 0
                x['overwrite_temp_v_dist'][s][d][i] = 0
                ##Output
                y['temp_v_dist_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_dist_slot'][s][d][i] = 1
                x['to_accept_temp_v_dist'][s][d][i] = 1
                x['overwrite_temp_v_dist'][s][d][i] = 0
                ##Output
                y['delay_new_v_dist'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_id_slot'][s][d][i] = 1
                x['to_accept_temp_v_id'][s][d][i] = 0
                x['overwrite_temp_v_id'][s][d][i] = 0
                ##Output
                y['temp_v_id_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_id_slot'][s][d][i] = 1
                x['to_accept_temp_v_id'][s][d][i] = 1
                x['overwrite_temp_v_id'][s][d][i] = 0
                ##Output
                y['delay_new_v_id'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_pi_slot'][s][d][i] = 1
                x['to_accept_temp_v_pi'][s][d][i] = 0
                x['overwrite_temp_v_pi'][s][d][i] = 0
                ##Output
                y['temp_v_pi_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_pi_slot'][s][d][i] = 1
                x['to_accept_temp_v_pi'][s][d][i] = 1
                x['overwrite_temp_v_pi'][s][d][i] = 0
                ##Output
                y['delay_new_v_pi'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_priority_slot'][s][d][i] = 1
                x['to_accept_temp_v_priority'][s][d][i] = 0
                x['overwrite_temp_v_priority'][s][d][i] = 0
                ##Output
                y['temp_v_priority_slot'][s][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['temp_v_priority_slot'][s][d][i] = 1
                x['to_accept_temp_v_priority'][s][d][i] = 1
                x['overwrite_temp_v_priority'][s][d][i] = 0
                ##Output
                y['delay_new_v_priority'][s][d][i] = 1
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

    ## Check if the pointer is new; number of instructions: 10*l*S

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['temp_pointer_slot'][s][i] = 1
            x['last_pointer'][s][i] = 1
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            if i == 0:  # i=1 (0-indexed, i=0 corresponds to i=1)
                for i_val in range(l): y['overwrite_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_last_in_q'][s][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_priority'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_pi'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_dist'][s][d_val][i_val] = 1
                for d_val in range(D): y['overwrite_temp_v_gray'][s][d_val] = 1
            elif i > 0:  # i>1
                y['compare_last_pointer'][s][i-1] = 1
                y['temp_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['temp_pointer_slot'][s][i] = 0
            x['last_pointer'][s][i] = 0
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            if i == 0:  # i=1
                for i_val in range(l): y['overwrite_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_last_in_q'][s][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_priority'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_pi'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_dist'][s][d_val][i_val] = 1
                for d_val in range(D): y['overwrite_temp_v_gray'][s][d_val] = 1
            elif i > 0:  # i>1
                y['compare_last_pointer'][s][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['temp_pointer_slot'][s][i] = 1
            x['last_pointer'][s][i] = 0
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['new_message'][s][i] = 1
            y['temp_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['temp_pointer_slot'][s][i] = 0
            x['last_pointer'][s][i] = 1
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
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
                for i_val in range(l): y['overwrite_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_last_in_q'][s][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_priority'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_pi'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['overwrite_temp_v_dist'][s][d_val][i_val] = 1
                for d_val in range(D): y['overwrite_temp_v_gray'][s][d_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['new_message'][s][i] = 1
            ##Output
            if i >0:
                y['new_message'][s][i-1] = 1
            elif i == 0:
                for i_val in range(l): y['to_accept_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['to_accept_temp_last_in_q'][s][i_val] = 1
                for d_val in range(D): y['to_accept_temp_v_gray'][s][d_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['to_accept_temp_v_dist'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['to_accept_temp_v_pi'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['to_accept_temp_v_id'][s][d_val][i_val] = 1
                for d_val in range(D):
                    for i_val in range(l): y['to_accept_temp_v_priority'][s][d_val][i_val] = 1
                y['return_message_flag'][s] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 0
            x['temp_pointer_slot'][s][i] = 1
            x['last_pointer'][s][i] = 1
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['temp_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 0
            x['temp_pointer_slot'][s][i] = 1
            x['last_pointer'][s][i] = 0
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['temp_pointer_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 0
            x['temp_pointer_slot'][s][i] = 1
            x['last_pointer'][s][i] = 1
            x['to_accept_temp_pointer'][s][i] = 1
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['delay_new_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 0
            x['temp_pointer_slot'][s][i] = 1
            x['last_pointer'][s][i] = 0
            x['to_accept_temp_pointer'][s][i] = 1
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['delay_new_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Take new messages from delays and put it into received messages as well as put it back in to placeholder; number of instructions: 2*l*(S+1)+D*(S+1)+4*l*D*(S+1)+(S+1)

    for i in range(l):
        for s in range(num_slots+1):

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_last_in_q'][s][i] = 1
            x['shutdown_last_in_q_delay'][s][i] = 0
            ##Output
            if s < num_slots:
                y['delay_new_last_in_q'][s+1][i] = 1
            elif s == num_slots:
                y['received_last_in_q'][i] = 1
                for s_val in range(num_slots):
                    y['last_in_q_placeholder'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_pointer'][s][i] = 1
            x['shutdown_pointer_delay'][s][i] = 0
            ##Output
            if s < num_slots:
                y['delay_new_pointer'][s+1][i] = 1
            elif s == num_slots:
                y['received_pointer'][i] = 1
                for s_val in range(num_slots):
                    y['pointer_placeholder'][s_val][i] = 1
                for s_val in range(num_slots):
                    y['persistent_last_pointer'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for d in range(D):
        for s in range(num_slots+1):

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_v_gray'][s][d] = 1
            x['shutdown_v_gray_delay'][s][d] = 0
            ##Output
            if s < num_slots:
                y['delay_new_v_gray'][s+1][d] = 1
            elif s == num_slots:
                y['received_v_gray'][d] = 1
                for s_val in range(num_slots):
                    y['v_gray_placeholder'][s_val][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):
            for s in range(num_slots+1):

                x, y = get_sample(l, D)
                ##Input
                x['delay_new_v_priority'][s][d][i] = 1
                x['shutdown_v_priority_delay'][s][d][i] = 0
                ##Output
                if s < num_slots:
                    y['delay_new_v_priority'][s+1][d][i] = 1
                elif s == num_slots:
                    y['received_v_priority'][d][i] = 1
                    for s_val in range(num_slots):
                        y['v_priority_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['delay_new_v_dist'][s][d][i] = 1
                x['shutdown_v_dist_delay'][s][d][i] = 0
                ##Output
                if s < num_slots:
                    y['delay_new_v_dist'][s+1][d][i] = 1
                elif s == num_slots:
                    y['received_v_dist'][d][i] = 1
                    for s_val in range(num_slots):
                        y['v_dist_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['delay_new_v_id'][s][d][i] = 1
                x['shutdown_v_id_delay'][s][d][i] = 0
                ##Output
                if s < num_slots:
                    y['delay_new_v_id'][s+1][d][i] = 1
                elif s == num_slots:
                    y['received_v_id'][d][i] = 1
                    for s_val in range(num_slots):
                        y['v_id_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['delay_new_v_pi'][s][d][i] = 1
                x['shutdown_v_pi_delay'][s][d][i] = 0
                ##Output
                if s < num_slots:
                    y['delay_new_v_pi'][s+1][d][i] = 1
                elif s == num_slots:
                    y['received_v_pi'][d][i] = 1
                    for s_val in range(num_slots):
                        y['v_pi_placeholder'][s_val][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for s in range(num_slots+1):

        x, y = get_sample(l, D)
        ##Input
        x['delay_new_message_flag'][s] = 1
        x['shutdown_message_flag_delay'][s] = 0
        ##Output
        if s < num_slots - 1:
            y['delay_new_message_flag'][s+1] = 1
        elif s == num_slots - 1:
            y['delay_new_message_flag'][s+1] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['overwrite_last_pointer'][s_val][i_val] = 1
            for s_val in range(num_slots):
                y['shutdown_message_flag_delay'][s_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_pointer_delay'][s_val][i_val] = 1
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
                for d_val in range(D):
                    for i_val in range(l):
                        y['shutdown_v_priority_delay'][s_val][d_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_last_in_q_delay'][s_val][i_val] = 1
        elif s == num_slots:
            for s_val in range(num_slots):
                y['message_flag_placeholder'][s_val] = 1
            y['received_new_message'] = 1
            for k_val in range(D):  
                for i_val in range(l):
                    y['check_u_id_with_received_v_ids'][k_val][i_val] = 1
            for k_val in range(D):
                y['u_id_comparison_counter'][k_val][0] = 1  # j=1 in 1-indexing becomes j=0 in 0-indexing

        T.append(x)
        Y.append(flatten_sample(y))

    ## Check local ids with ids in the message and set update flags for the matching ones; number of instructions: 8*l*D+5*l*(D^2)+2*D+(D^2)+l+3

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['v_id'][d][i] = 1
            x['upload_v_id'][d][i] = 0
            ##Output
            y['v_id'][d][i] = 1
            for k_val in range(D):
                y['v_ids2check_with_received_v_ids'][d][k_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_id'][d][i] = 1
            x['upload_v_id'][d][i] = 1
            ##Output
            y['v_id'][d][i] = 1
            for k_val in range(D):
                y['v_ids2check_with_received_v_ids'][d][k_val][i] = 1
            for s_val in range(num_slots):
                y['v_id_placeholder'][s_val][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['received_new_message'] = 1
    ##Output
    for d_val in range(D):
        for k_val in range(D):
            for i_val in range(l):
                y['check_v_ids_with_received_v_ids'][d_val][k_val][i_val] = 1
    y['update_pointer_counter'][0] = 1  # j=1 in 1-indexing, j=0 in 0-indexing
    for d_val in range(D):
        for k_val in range(D):
            y['v_id_comparison_counter'][d_val][k_val][0] = 1  # j=1 in 1-indexing, j=0 in 0-indexing
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
            if i == 0:  # i=1 in 1-indexing, i=0 in 0-indexing
                y['u_equals_received_v_id'][k][i] = 1
                for d_val in range(D):
                    y['received_v_ids2check_with_v_ids'][d_val][k][i] = 1
                y['u_is_received_v_id_counter'][k][i] = 1
            else:  # i>1 in 1-indexing, i>=1 in 0-indexing
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
            if i == 0:  # i=1 in 1-indexing, i=0 in 0-indexing
                y['u_equals_received_v_id'][k][i] = 1
                y['u_is_received_v_id_counter'][k][i] = 1
            else:  # i>1 in 1-indexing, i>=1 in 0-indexing
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

    for i in range(l):
        for d in range(D):
            for k in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['received_v_ids2check_with_v_ids'][d][k][i] = 1
                x['v_ids2check_with_received_v_ids'][d][k][i] = 1
                x['check_v_ids_with_received_v_ids'][d][k][i] = 1
                ##Output
                if i == 0:  # i=1 in 1-indexing, i=0 in 0-indexing
                    y['v_equals_received_v_id'][d][k][i] = 1
                    y['v_is_received_v_id_counter'][d][k][i] = 1
                else:  # i>1 in 1-indexing, i>=1 in 0-indexing
                    y['v_equals_received_v_id'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['received_v_ids2check_with_v_ids'][d][k][i] = 0
                x['v_ids2check_with_received_v_ids'][d][k][i] = 0
                x['check_v_ids_with_received_v_ids'][d][k][i] = 1
                ##Output
                if i == 0:  # i=1 in 1-indexing, i=0 in 0-indexing
                    y['v_equals_received_v_id'][d][k][i] = 1
                    y['v_is_received_v_id_counter'][d][k][i] = 1
                else:  # i>1 in 1-indexing, i>=1 in 0-indexing
                    y['v_equals_received_v_id'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for j in range(l+1):
        for k in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['u_id_comparison_counter'][k][j] = 1
            ##Output
            if j < l:  # j < l+1 in 1-indexing, j < l in 0-indexing
                y['u_id_comparison_counter'][k][j+1] = 1
            elif j == l:  # j = l+1 in 1-indexing, j = l in 0-indexing
                for i_val in range(l):
                    y['reset_u_equals_v'][k][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for i in range(l):
        for k in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['u_equals_received_v_id'][k][i] = 1
            x['u_is_received_v_id_counter'][k][i] = 1
            x['reset_u_equals_v'][k][i] = 0
            ##Output
            if i < l-1:  # i < l in 1-indexing, i < l-1 in 0-indexing
                y['u_is_received_v_id_counter'][k][i+1] = 1
            elif i == l-1:  # i = l in 1-indexing, i = l-1 in 0-indexing
                for i_val in range(l):
                    y['u_dist_update'][k][i_val] = 1
                for i_val in range(l):
                    y['u_pi_update'][k][i_val] = 1
                for i_val in range(l):
                    y['u_priority_update'][k][i_val] = 1
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
            for d in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['v_id_comparison_counter'][d][k][j] = 1
                ##Output
                if j < l:  # j < l+1 in 1-indexing, j < l in 0-indexing
                    y['v_id_comparison_counter'][d][k][j+1] = 1
                elif j == l:  # j = l+1 in 1-indexing, j = l in 0-indexing
                    for i_val in range(l):
                        y['reset_v_equals_v'][d][k][i_val] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for i in range(l):
        for k in range(D):
            for d in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['v_equals_received_v_id'][d][k][i] = 1
                x['v_is_received_v_id_counter'][d][k][i] = 1
                x['reset_v_equals_v'][d][k][i] = 0
                ##Output
                if i < l-1:  # i < l in 1-indexing, i < l-1 in 0-indexing
                    y['v_is_received_v_id_counter'][d][k][i+1] = 1
                elif i == l-1:  # i = l in 1-indexing, i = l-1 in 0-indexing
                    for i_val in range(l):
                        y['v_dist_update'][d][k][i_val] = 1
                    for i_val in range(l):
                        y['v_pi_update'][d][k][i_val] = 1
                    for i_val in range(l):
                        y['v_priority_update'][d][k][i_val] = 1
                    y['v_gray_update'][d][k] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_equals_received_v_id'][d][k][i] = 1
                x['v_is_received_v_id_counter'][d][k][i] = 0
                x['reset_v_equals_v'][d][k][i] = 0
                ##Output
                y['v_equals_received_v_id'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

    for j in range(l+D+2):

        x, y = get_sample(l, D)
        ##Input
        x['update_pointer_counter'][j] = 1
        ##Output
        if j < D+l:  # j < l+D+1 in 1-indexing, j < D+l in 0-indexing
            y['update_pointer_counter'][j+1] = 1
        elif j == D+l:  # j = l+D+1 in 1-indexing, j = D+l in 0-indexing
            for i_val in range(l):
                y['update_pointer'][i_val] = 1
            y['update_pointer_counter'][j+1] = 1
        elif j == D+l+1:  # j = l+D+2 in 1-indexing, j = D+l+1 in 0-indexing
            for i_val in range(l):
                y['compare_u_priority'][i_val] = 1
            y['priority_comparison_counter'][0] = 1  # i=1 in 1-indexing, i=0 in 0-indexing
            for k_val in range(D):
                for i_val in range(l):
                    y['renew_received_v_dist'][k_val][i_val] = 1
            for k_val in range(D):
                for i_val in range(l):
                    y['renew_received_v_pi'][k_val][i_val] = 1
            for k_val in range(D):
                for i_val in range(l):
                    y['renew_received_v_priority'][k_val][i_val] = 1
            for k_val in range(D):
                y['renew_received_v_gray'][k_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Update the current node's distance, priority parent, white and gray flag; number of instructions: 9*l*D+3*D+4*l

    for i in range(l):
        for k in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['received_v_priority'][k][i] = 1
            x['renew_received_v_priority'][k][i] = 0
            ##Output
            y['received_v_priority'][k][i] = 1
            y['received_v_priority4u'][k][i] = 1
            for d_val in range(D):
                y['received_v_priority4v'][d_val][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_priority_update'][k][i] = 1
            x['received_v_priority4u'][k][i] = 1
            ##Output
            y['u_priority_delay'][k][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_priority_delay'][k][i] = 1
            ##Output
            if k < D - 1:
                y['u_priority_delay'][k+1][i] = 1
            elif k == D - 1:
                y['u_priority'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

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
        x['received_pointer'][i] = 1
        x['update_pointer'][i] = 0
        ##Output
        y['received_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_pointer'][i] = 1
        x['update_pointer'][i] = 1
        ##Output
        y['q_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_last_in_q'][i] = 1
        x['update_last_in_q'][i] = 0
        ##Output
        y['received_last_in_q'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_last_in_q'][i] = 1
        x['update_last_in_q'][i] = 1
        ##Output
        y['u_last_in_q'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Update the adjacent nodes' distance, parent, and white and gray flags; number of instructions: 6*l*(D^2)+2*(D^2)

    for i in range(l):
        for k in range(D):
            for d in range(D):

                x, y = get_sample(l, D)
                ##Input
                x['v_priority_update'][d][k][i] = 1
                x['received_v_priority4v'][d][k][i] = 1
                ##Output
                y['v_priority_delay'][d][k][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

                x, y = get_sample(l, D)
                ##Input
                x['v_priority_delay'][d][k][i] = 1
                ##Output
                if k < D - 1:  # k < D (0-indexed: k < D-1)
                    y['v_priority_delay'][d][k+1][i] = 1
                elif k == D - 1:  # k = D (0-indexed: k == D-1)
                    y['v_priority'][d][i] = 1
                T.append(x)
                Y.append(flatten_sample(y))

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
                if k < D - 1:  # k < D (0-indexed: k < D-1)
                    y['v_dist_delay'][d][k+1][i] = 1
                elif k == D - 1:  # k = D (0-indexed: k == D-1)
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
                if k < D - 1:  # k < D (0-indexed: k < D-1)
                    y['v_pi_delay'][d][k+1][i] = 1
                elif k == D - 1:  # k = D (0-indexed: k == D-1)
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
            if k < D - 1:  # k < D (0-indexed: k < D-1)
                y['v_gray_delay'][d][k+1] = 1
            elif k == D - 1:  # k = D (0-indexed: k == D-1)
                y['v_gray'][d] = 1
                y['v_white_overwrite'][d] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    
    
    return T, np.array(Y)

