import numpy as np
from .utils import flatten_sample, nested_dict
import csv

def get_sample(l,D):
    num_slots = D**2+1
    sample = { ## toal size of the vector:  7D^2+113l+38Dl+2D+52D^2l+21

        ## Creating required copies of the value of $|V|$; size:l

        'num_nodes': [0] * l, # collisions= 1

        ## Check if the round is smaller than $|V|$; size:  4l

        
        'compare_round': [0] * l, # collisions= 2
        'n4round': [0] * l, # collisions= 1
        'round': [0] * l, # collisions= 3
        'round_smaller_than_num': [0] * l, # collisions= 2

        ## Check if id matches the pointer; size: 8l+2

        'u_id': [0] * l, # collisions= 4
        'pointer': [0] * l, # collisions= 1
        'compare_u_id': [0] * l, # collisions= 2
        'upload_u_id': [0] * l, # collisions= 1
        'u_matches_pointer': [0] * l, # collisions= 3
        'u_matches_counter': [0] * l, # collisions= 3
        'reset_u_matches': [0] * l, # collisions= 1
        'u_id_comparison_counter': [0] * (l + 2), # collisions= 1

        ## Setting up the neighboring node distances to be aggregated with the weights; size: 5lD

        'v_dist': [[0]*l for _ in range(D)], # collisions= 3
        'fill_candidate_dist_augend': [[0]*l for _ in range(D)], # collisions= 1
        'overwrite_v_dist': [[0]*l for _ in range(D)], # collisions= 1
        'v_inf_backup': [[0]*l for _ in range(D)], # collisions= 2
        'overwrite_v_inf': [[0]*l for _ in range(D)], # collisions= 1

        ## Setting up the edge weights to be aggregated with the neighboring distances; size: 3lD

        'weight': [[0]*l for _ in range(D)], # collisions= 4
        'fill_candidate_dist_addend': [[0]*l for _ in range(D)], # collisions= 1
        'v_inf': [[0]*l for _ in range(D)], # collisions= 1

        ## Setting up the nodes current distances for comparison; size: 6l+4lD

        'u_dist': [0] * l, # collisions= 4
        'fill_incumbent_dist': [0] * l, # collisions= 1
        'upload_u_dist': [0] * l, # collisions= 1
        'backup_incumbent_dist': [[0] * l for _ in range(D)], # collisions= 3
        'keep_incumbent_dist': [[0] * l for _ in range(D)], # collisions= 3
        'reset_backup_incumbent_dist': [[0] * l for _ in range(D)], # collisions= 1
        'overwrite_u_inf_pipe': [[0]*l for _ in range(D)], # collisions= 2
        'u_inf': [0] * l, # collisions= 2
        'overwrite_u_inf': [0] * l, # collisions= 1
        'upload_u_inf': [0] * l, # collisions= 1

        ## Set the loop parameters up; size: 2l+D+4

        'u_matched': 0, # collisions= 1
        'v_turn': [0] * (D+1),  # d from 1 to D (0-indexed: indices 0 to D-1 for d=1 to D) # collisions= 4
        'candidate_dist_sum_counter': [0] * (2*l+2),  # j from 1 to 2l+2 (0-indexed: indices 0 to 2l+1) # collisions= 1
        
        ## Addition of the neighbors' distance with the edge weight; size: 4lD

        'candidate_dist_augend': [[0]*l for _ in range(D)], # collisions= 3
        'candidate_dist_addend': [[0]*l for _ in range(D)], # collisions= 2
        'to_copy_candidate_dist_augend': [[0]*l for _ in range(D)], # collisions= 1
        'candidate_dist_carry': [[0]*l for _ in range(D)], # collisions= 2

        ## Set up the next incumbent distance with candidate dist if the latter is smaller; size: 3lD

        'backup_candidate_dist': [[0]*l for _ in range(D)], # collisions= 2
        'choose_candidate_dist': [[0]*l for _ in range(D)], # collisions= 1
        'reset_backup_candidate_dist': [[0]*l for _ in range(D)], # collisions= 1
        
        ## Check if the the nodes current distance is larger than candidate distances; size: 5lD

        'compare_u_dist': [[0]*l for _ in range(D)], # collisions= 3
        'incumbent_dist': [[0]*l for _ in range(D)], # collisions= 3
        'candidate_dist': [[0]*l for _ in range(D)], # collisions= 2
        'smaller_candidate_dist': [[0]*l for _ in range(D)], # collisions= 2
        'smaller_incumbent_dist': [[0]*l for _ in range(D)], # collisions= 2

        ## Check if pointer has reached the last node; size: 10l+3

        'u_pointer': [0] * l, # collisions= 4
        'n4pointer': [0] * l, # collisions= 1
        'check_pointer': [0] * l, # collisions= 1
        'pointer_matches_n': [0] * l, # collisions= 3
        'pointer_matches_n_counter': [0] * l, # collisions= 3
        'check_pointer_counter': [0] * (l + 1), # collisions= 1
        'decide_pointer': 0, # collisions= 1
        'reset_pointer': 0, # collisions= 1
        'pointer_sum_counter': [0] * (2 * l), # collisions= 1
        'round_sum_counter': [0] * (2 * l), # collisions= 2

        ## Incrementing pointer if it has not reached the last node; size: 4l

        'pointer_augend': [0] * l, # collisions= 4
        'pointer_addend': [0] * l, # collisions= 1
        'to_copy_pointer_augend': [0] * l, # collisions= 1
        'pointer_carry': [0] * l, # collisions= 2

        ## Incrementing round if pointer has reached the last node; size: 4l

        'round_augend': [0] * l, # collisions= 3
        'round_addend': [0] * l, # collisions= 1
        'to_copy_round_augend': [0] * l, # collisions= 1
        'round_carry': [0] * l, # collisions= 2

        ## Upload next pointer and next round; size: 4l+1

        'next_pointer': [0] * l, # collisions= 3
        'upload_next_pointer': [0] * l, # collisions= 1
        'next_round': [0] * l, # collisions= 2
        'upload_next_round': [0] * l, # collisions= 1
        'upload_message_flag': 0, # collisions= 1

        ## Guide the placeholder content to appropriate sending slot; size: 10lS+3S

        'determine_slot': [0] * num_slots, # collisions= 1
        'determine_pointer_slot': [[0] * l for _ in range(num_slots)], # collisions= 3
        'pointer_placeholder': [[0] * l for _ in range(num_slots)], # collisions= 2
        'determine_round_slot': [[0] * l for _ in range(num_slots)], # collisions= 3
        'round_placeholder': [[0] * l for _ in range(num_slots)], # collisions= 2
        'determine_u_dist_slot': [[0] * l for _ in range(num_slots)], # collisions= 3
        'u_dist_placeholder': [[0] * l for _ in range(num_slots)], # collisions= 2
        'determine_u_id_slot': [[0] * l for _ in range(num_slots)], # collisions= 3
        'u_id_placeholder': [[0] * l for _ in range(num_slots)], # collisions= 2
        'determine_u_inf_slot': [[0]*l for _ in range(num_slots)], # collisions= 3
        'u_inf_placeholder': [[0]*l for _ in range(num_slots)], # collisions= 2
        'determine_message_flag_slot': [0] * num_slots, # collisions= 3
        'message_flag_placeholder': [0] * num_slots, # collisions= 2

        ## Preserve temp data till the new message check is complete; size: 19lS + S

        'persistent_last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 3
        'overwrite_last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2
        'persistent_last_round': [[0]*l for _ in range(num_slots)], # collisions= 3
        'overwrite_last_round': [[0]*l for _ in range(num_slots)], # collisions= 2
        'temp_pointer_slot': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_pointer': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_pointer': [[0]*l for _ in range(num_slots)], # collisions= 3
        'temp_round_slot': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_round': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_round': [[0]*l for _ in range(num_slots)], # collisions= 3
        'temp_u_id_slot': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_u_id': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_u_id': [[0]*l for _ in range(num_slots)], # collisions= 3
        'temp_u_dist_slot': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_u_dist': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_u_dist': [[0]*l for _ in range(num_slots)], # collisions= 3
        'temp_u_inf_slot': [[0]*l for _ in range(num_slots)], # collisions= 2
        'to_accept_temp_u_inf': [[0]*l for _ in range(num_slots)], # collisions= 1
        'overwrite_temp_u_inf': [[0]*l for _ in range(num_slots)], # collisions= 3
        'return_message_flag': [0]*num_slots, # collisions= 1

        ## Check if the round in the message is new; size: 5lS

        'compare_last_round': [[0]*l for _ in range(num_slots)], # collisions= 2
        'incoming_round': [[0]*l for _ in range(num_slots)], # collisions= 2
        'last_round': [[0]*l for _ in range(num_slots)], # collisions= 2
        'new_message': [[0]*l for _ in range(num_slots)], # collisions= 3
        'old_message': [[0]*l for _ in range(num_slots)], # collisions= 2

        ## In the case that round is not new, check if the pointer in the message is new; size: 3lS

        'compare_last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2
        'incoming_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2
        'last_pointer': [[0]*l for _ in range(num_slots)], # collisions= 2


        ## Take new messages from delays and put it into received messages as well as put it back in to placeholder; size: 10l(S+1)+2(S+1)

        'delay_new_round': [[0]*l for _ in range(num_slots+1)], # collisions= 2
        'shutdown_round_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_pointer': [[0]*l for _ in range(num_slots+1)], # collisions= 2
        'shutdown_pointer_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_u_id': [[0]*l for _ in range(num_slots+1)], # collisions= 2
        'shutdown_u_id_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_u_dist': [[0]*l for _ in range(num_slots+1)], # collisions= 2
        'shutdown_u_dist_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_u_inf': [[0]*l for _ in range(num_slots+1)], # collisions= 2
        'shutdown_u_inf_delay': [[0]*l for _ in range(num_slots+1)], # collisions= 1
        'delay_new_message_flag': [0] * (num_slots+1), # collisions= 2
        'shutdown_message_flag_delay': [0] * (num_slots+1), # collisions= 1

        ## Check local ids with the id in the message and set update flag for the matching one; size: 8lD+D+l+2

        'v_id': [[0]*l for _ in range(D)], # collisions= 1
        'received_new_message': 0, # collisions= 1
        'received_u_id': [[0]*l for _ in range(D)], # collisions= 1
        'v_id2check': [[0]*l for _ in range(D)], # collisions= 1
        'check_v_id_with_received_u_id': [[0]*l for _ in range(D)], # collisions= 1
        'v_id_comparison_counter': [[0]*(l+1) for _ in range(D)], # collisions= 1
        'v_equals_received_u_id': [[0]*l for _ in range(D)], # collisions= 3
        'v_is_received_u_id_counter': [[0]*l for _ in range(D)], # collisions= 2
        'reset_v_equals_u': [[0]*l for _ in range(D)], # collisions= 1
        'update_round_counter': [0]*(l+1), # collisions= 1

        ## Update the distance information that the node has about its neighbors; size: 6lD

        'received_u_dist': [[0]*l for _ in range(D)], # collisions= 2
        'v_dist_update': [[0]*l for _ in range(D)], # collisions= 1
        'renew_received_u_dist': [[0]*l for _ in range(D)], # collisions= 1
        'received_u_inf': [[0]*l for _ in range(D)], # collisions= 2
        'v_inf_update': [[0]*l for _ in range(D)], # collisions= 1
        'renew_received_u_inf': [[0]*l for _ in range(D)], # collisions= 1

        ## Put the incoming round in the comparison blocks and update; size: 2l

        'received_round': [0]*l, # collisions= 2
        'update_round': [0]*l, # collisions= 1

        ## Setting up round for checking if it is less than $|V|$ for incrementing of pointer has reached the end; size: 3l

        'backup_round': [0]*l, # collisions= 3
        'setup_round_augend': [0]*l, # collisions= 1
        'reset_round': [0]*l, # collisions= 2

        ## Put the incoming pointer in the comparison blocks and update; size: 2l

        'received_pointer': [0] * l, # collisions= 2
        'update_pointer': [0] * l, # collisions= 1

        ## Copy incoming messages to temporary slots while checking if the round or pointer are new; size: 5lS + S
        'pointer_slot':[[0]*l for _ in range(num_slots)], # collisions= 1
        'u_id_slot': [[0]*l for _ in range(num_slots)], # collisions= 1
        'u_inf_slot': [[0]*l for _ in range(num_slots)], # collisions= 1
        'round_slot': [[0]*l for _ in range(num_slots)], # collisions= 1
        'u_dist_slot': [[0]*l for _ in range(num_slots)], # collisions= 1
        'message_flag_slot': [0]*num_slots, # collisions= 1
        



    }
    return nested_dict(3), sample


def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)


def get_dataset(l,D): ## total number of instructions is: 59lD+16D+55l+9(D^2)+31l(D^2)+30l(D^3)+14
    num_slots = D**2+1
    T, Y = [], []

    ## Creating required copies of the value of $|V|$; number of instructions: l
    
    for i in range(l):
        x, y = get_sample(l, D)
        ##Input
        x['num_nodes'][i] = 1
        ##Output
        y['num_nodes'][i] = 1
        y['n4round'][i] = 1
        y['n4pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Check if the round is smaller than $|V|$; number of instructions: 4l

    for i in range(l):
        x, y = get_sample(l, D)
        ## Input
        x['compare_round'][i] = 1
        x['n4round'][i] = 1
        x['round'][i] = 1
        ## Output
        if i > 0:  # i > 1 in 1-indexing becomes i > 0 in 0-indexing
            y['compare_round'][i-1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ## Input
        x['compare_round'][i] = 1
        x['num_nodes'][i] = 0  # num-nodes[i] = 0
        x['round'][i] = 0
        ## Output
        if i > 0:  # i > 1 in 1-indexing becomes i > 0 in 0-indexing
            y['compare_round'][i-1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ## Input
        x['compare_round'][i] = 1
        x['num_nodes'][i] = 1  # num-nodes[i] = 1
        x['round'][i] = 0
        ## Output
        y['round_smaller_than_num'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ## Input
        x['round_smaller_than_num'][i] = 1
        ## Output
        if i > 0:  # i < l in 1-indexing becomes i < l-1 in 0-indexing
            y['round_smaller_than_num'][i-1] = 1
        elif i == 0:  # i = l in 1-indexing becomes i = l-1 in 0-indexing
            for i_val in range(l):
                y['update_pointer'][i_val] = 1
            y['u_id_comparison_counter'][0] = 1  # u-id-comparison-counter[1] in 1-indexing becomes position 1 in 0-indexing
        T.append(x)
        Y.append(flatten_sample(y))

    ## Check if id matches the pointer; number of instructions: 8l+2

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['u_id'][i] = 1
        x['pointer'][i] = 1
        x['compare_u_id'][i] = 1
        x['upload_u_id'][i] = 0
        ##Output
        if i == 0:  # i=1 (0-indexed)
            y['u_matches_pointer'][i] = 1
            y['u_matches_counter'][i] = 1
            y['u_id'][i] = 1
            y['u_pointer'][i] = 1
            y['pointer_augend'][i] = 1
        elif i > 0:  # i>1 (0-indexed)
            y['u_matches_pointer'][i] = 1
            y['u_id'][i] = 1
            y['u_pointer'][i] = 1
            y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_id'][i] = 0
        x['pointer'][i] = 0
        x['compare_u_id'][i] = 1
        x['upload_u_id'][i] = 0
        ##Output
        if i == 0:  # i=1 (0-indexed)
            y['u_matches_pointer'][i] = 1
            y['u_matches_counter'][i] = 1
        elif i > 0:  # i>1 (0-indexed)
            y['u_matches_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_id'][i] = 1
        x['pointer'][i] = 0
        x['compare_u_id'][i] = 1
        x['upload_u_id'][i] = 0
        ##Output
        y['u_id'][i] = 1
        y['u_pointer'][i] = 1
        y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_id'][i] = 1
        x['pointer'][i] = 0
        x['compare_u_id'][i] = 0
        x['upload_u_id'][i] = 0
        ##Output
        y['u_id'][i] = 1
        y['u_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_id'][i] = 1
        x['pointer'][i] = 0
        x['compare_u_id'][i] = 0
        x['upload_u_id'][i] = 1
        ##Output
        y['u_id'][i] = 1
        y['u_pointer'][i] = 1
        for s_val in range(num_slots):
            y['u_id_placeholder'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_matches_pointer'][i] = 1
        x['u_matches_counter'][i] = 1
        x['reset_u_matches'][i] = 0
        ##Output
        if i < l - 1:  # i < l (0-indexed)
            y['u_matches_counter'][i + 1] = 1
        elif i == l - 1:  # i = l (0-indexed)
            y['u_matched'] = 1
            for i_val in range(l):
                y['check_pointer'][i_val] = 1
            y['check_pointer_counter'][0] = 1  # j=1 (0-indexed)
            for i_val in range(l):
                y['setup_round_augend'][i_val] = 1
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

    for j in range(l+2):

        x, y = get_sample(l, D)
        ##Input
        x['u_id_comparison_counter'][j] = 1
        ##Output
        if j < l + 1:  # j < l+2 (0-indexed: j ranges 0 to l+1)
            y['u_id_comparison_counter'][j + 1] = 1
        elif j == l + 1:  # j = l+2 (0-indexed)
            for i_val in range(l):
                y['reset_u_matches'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))


    ## Setting up the neighboring node distances to be aggregated with the weights; number of instructions: 3lD

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['v_dist'][d][i] = 1
            x['fill_candidate_dist_augend'][d][i] = 1
            x['overwrite_v_dist'][d][i] = 0
            ##Output
            y['v_dist'][d][i] = 1
            y['candidate_dist_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_dist'][d][i] = 1
            x['fill_candidate_dist_augend'][d][i] = 0
            x['overwrite_v_dist'][d][i] = 0
            ##Output
            y['v_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_inf_backup'][d][i] = 1
            x['overwrite_v_inf'][d][i] = 0
            ##Output
            y['v_inf_backup'][d][i] = 1
            y['v_inf'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))


    ## Setting up the edge weights to be aggregated with the neighboring distances; number of instructions: 4Dl


    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['weight'][d][i] = 1
            x['fill_candidate_dist_addend'][d][i] = 1
            x['v_inf'][d][i] = 0
            ##Output
            y['candidate_dist_addend'][d][i] = 1
            y['weight'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['weight'][d][i] = 1
            x['fill_candidate_dist_addend'][d][i] = 0
            x['v_inf'][d][i] = 0
            ##Output
            y['weight'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['weight'][d][i] = 1
            x['fill_candidate_dist_addend'][d][i] = 1
            x['v_inf'][d][i] = 1
            ##Output
            y['weight'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['weight'][d][i] = 1
            x['fill_candidate_dist_addend'][d][i] = 0
            x['v_inf'][d][i] = 1
            ##Output
            y['weight'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))


    ## Setting up the nodes current distances for comparison; number of instructions: 5l+3lD

    for i in range(l):
        
        x, y = get_sample(l, D)
        ##Input
        x['u_dist'][i] = 1
        x['fill_incumbent_dist'][i] = 1
        x['upload_u_dist'][i] = 0
        ##Output
        y['incumbent_dist'][0][i] = 1  # d=1 becomes index 0
        y['backup_incumbent_dist'][0][i] = 1  # d=1 becomes index 0
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_dist'][i] = 1
        x['fill_incumbent_dist'][i] = 0  
        x['upload_u_dist'][i] = 0
        ##Output
        y['u_dist'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_dist'][i] = 1
        x['fill_incumbent_dist'][i] = 0  
        x['upload_u_dist'][i] = 1
        ##Output
        y['u_dist'][i] = 1
        for s_val in range(num_slots):
            y['u_dist_placeholder'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_inf'][i] = 1
        x['overwrite_u_inf'][i] = 0
        x['upload_u_inf'][i] = 0
        ##Output
        y['u_inf'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_inf'][i] = 1
        x['overwrite_u_inf'][i] = 0
        x['upload_u_inf'][i] = 1
        ##Output
        y['u_inf'][i] = 1
        for s_val in range(num_slots):
            y['u_inf_placeholder'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))


    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['backup_incumbent_dist'][d][i] = 1
            x['keep_incumbent_dist'][d][i] = 0
            x['reset_backup_incumbent_dist'][d][i] = 0
            ##Output
            y['incumbent_dist'][d][i] = 1
            y['backup_incumbent_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['backup_incumbent_dist'][d][i] = 1
            x['keep_incumbent_dist'][d][i] = 1
            x['reset_backup_incumbent_dist'][d][i] = 0
            ##Output
            if d < D - 1:  # d < D (0-indexed: d < D-1)
                y['backup_incumbent_dist'][d+1][i] = 1
                y['incumbent_dist'][d+1][i] = 1
            elif d == D - 1:  # d = D (0-indexed: d = D-1)
                y['u_dist'][i] = 1  
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['overwrite_u_inf_pipe'][d][i] = 1
            ##Output
            if d < D - 1:  # d < D (0-indexed: d < D-1)
                y['overwrite_u_inf_pipe'][d+1][i] = 1
            elif d == D - 1:  # d = D (0-indexed: d == D-1)
                y['overwrite_u_inf'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))


    ## Set the loop parameters up; number of instructions: 2l+D+4

    x, y = get_sample(l, D)
    ##Input
    x['u_matched'] = 1
    ##Output
    y['v_turn'][0] = 1  # d=1 (0-indexed: index 0)
    T.append(x)
    Y.append(flatten_sample(y))

    for d in range(D+1):

        x, y = get_sample(l, D)
        ##Input
        x['v_turn'][d] = 1  # d is 0-indexed (d=0 represents d=1, d=1 represents d=2, etc.)
        ##Output
        if d == 0:  # d = 1
            for d_val in range(D):
                for i_val in range(l): y['fill_candidate_dist_augend'][d_val][i_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['fill_candidate_dist_addend'][d_val][i_val] = 1
            for i_val in range(l): y['fill_incumbent_dist'][i_val] = 1
            y['candidate_dist_sum_counter'][0] = 1  # j=1 (0-indexed: index 0)
        elif 0 < d < D:  # 1 < d < D+1 (d from 2 to D-1 in 1-indexing)
            y['compare_u_dist'][d][l-1] = 1  # i=l (0-indexed: index l-1)
        elif d == D:  # d = D+1 (0-indexed: index D represents d=D+1)
            for i_val in range(l): y['upload_next_pointer'][i_val] = 1
            for i_val in range(l): y['upload_next_round'][i_val] = 1
            for i_val in range(l): y['upload_u_id'][i_val] = 1
            for i_val in range(l): y['upload_u_dist'][i_val] = 1
            for i_val in range(l): y['upload_u_inf'][i_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['reset_backup_incumbent_dist'][d_val][i_val] = 1
            for d_val in range(D):
                for i_val in range(l): y['reset_backup_candidate_dist'][d_val][i_val] = 1
            y['upload_message_flag'] = 1
            for s_val in range(num_slots):
                for i_val in range(l): y['overwrite_last_pointer'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l): y['overwrite_last_round'][s_val][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(2*l+2):

        x, y = get_sample(l, D)
        ##Input
        x['candidate_dist_sum_counter'][j] = 1  # j is 0-indexed
        ##Output
        if j < 2*l:  # j < 2l+1 (0-indexed: j from 0 to 2l-1 represents j=1 to 2l)
            y['candidate_dist_sum_counter'][j+1] = 1
        elif j == 2*l:  # j = 2l+1 (0-indexed: index 2l represents j=2l+1)
            y['candidate_dist_sum_counter'][j+1] = 1  # j+1 = 2l+2
            for d_val in range(D):
                for i_val in range(l): y['to_copy_candidate_dist_augend'][d_val][i_val] = 1
        elif j == 2*l+1:  # j = 2l+2 (0-indexed: index 2l+1 represents j=2l+2)
            y['compare_u_dist'][0][l-1] = 1  # d=1, i=l (0-indexed: d=0, i=l-1)
        T.append(x)
        Y.append(flatten_sample(y))

    ## Addition of the neighbors' distance with the edge weight; number of instructions: 5lD

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['candidate_dist_augend'][d][i] = 1
            x['candidate_dist_addend'][d][i] = 0
            x['to_copy_candidate_dist_augend'][d][i] = 0
            ##Output
            y['candidate_dist_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['candidate_dist_augend'][d][i] = 1
            x['candidate_dist_addend'][d][i] = 1
            x['to_copy_candidate_dist_augend'][d][i] = 0
            ##Output
            y['candidate_dist_carry'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['candidate_dist_augend'][d][i] = 0
            x['candidate_dist_addend'][d][i] = 1
            x['to_copy_candidate_dist_augend'][d][i] = 0
            ##Output
            y['candidate_dist_augend'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['candidate_dist_carry'][d][i] = 1
            ##Output
            if i < l - 1:  # i < l (0-indexed: i < l-1)
                y['candidate_dist_addend'][d][i+1] = 1
            elif i == l - 1:  # i = l (0-indexed: i = l-1)
                y['candidate_dist_carry'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['candidate_dist_augend'][d][i] = 1
            x['candidate_dist_addend'][d][i] = 0
            x['to_copy_candidate_dist_augend'][d][i] = 1
            ##Output
            y['candidate_dist'][d][i] = 1
            y['backup_candidate_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))


    ## Set up the next incumbent distance with candidate dist if the latter is smaller; number of instructions: 2lD

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['backup_candidate_dist'][d][i] = 1
            x['choose_candidate_dist'][d][i] = 0
            x['reset_backup_candidate_dist'][d][i] = 0
            ##Output
            y['candidate_dist'][d][i] = 1
            y['backup_candidate_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['backup_candidate_dist'][d][i] = 1
            x['choose_candidate_dist'][d][i] = 1
            x['reset_backup_candidate_dist'][d][i] = 0
            ##Output
            if d < D - 1:  # d < D (0-indexed: D-1)
                y['backup_incumbent_dist'][d+1][i] = 1
                y['incumbent_dist'][d+1][i] = 1
            elif d == D - 1:  # d = D (0-indexed: D-1)
                y['u_dist'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))


    ## Check if the the nodes current distance is larger than candidate distances; number of instructions: 6lD

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['compare_u_dist'][d][i] = 1
            x['incumbent_dist'][d][i] = 1
            x['candidate_dist'][d][i] = 1
            ##Output
            if i == 0:  # i=1 in 1-indexing
                for i_val in range(l):
                    y['keep_incumbent_dist'][d][i_val] = 1
                y['v_turn'][d+1] = 1
            else:  # i>1 in 1-indexing (i>=1 in 0-indexing)
                y['compare_u_dist'][d][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_u_dist'][d][i] = 1
            x['incumbent_dist'][d][i] = 0
            x['candidate_dist'][d][i] = 1
            ##Output
            y['smaller_incumbent_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_u_dist'][d][i] = 1
            x['incumbent_dist'][d][i] = 0
            x['candidate_dist'][d][i] = 0
            ##Output
            if i == 0:  # i=1 in 1-indexing
                for i_val in range(l):
                    y['keep_incumbent_dist'][d][i_val] = 1
                y['v_turn'][d+1] = 1
            else:  # i>1 in 1-indexing (i>=1 in 0-indexing)
                y['compare_u_dist'][d][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_u_dist'][d][i] = 1
            x['incumbent_dist'][d][i] = 1
            x['candidate_dist'][d][i] = 0
            ##Output
            y['smaller_candidate_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['smaller_candidate_dist'][d][i] = 1
            ##Output
            if i > 0:  # i < l in 1-indexing
                y['smaller_candidate_dist'][d][i-1] = 1
            elif i == 0:  # i = l in 1-indexing
                for i_val in range(l):
                    y['choose_candidate_dist'][d][i_val] = 1
                y['v_turn'][d+1] = 1
                for i_val in range(l): y['overwrite_u_inf_pipe'][d][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['smaller_incumbent_dist'][d][i] = 1
            ##Output
            if i > 0:  # i < l in 1-indexing
                y['smaller_incumbent_dist'][d][i-1] = 1
            elif i == 0:  # i = l in 1-indexing
                for i_val in range(l):
                    y['keep_incumbent_dist'][d][i_val] = 1
                y['v_turn'][d+1] = 1
            T.append(x)
            Y.append(flatten_sample(y))


    ## Check if pointer has reached the last node; number of instructions: 9l+3

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['u_pointer'][i] = 1
        x['n4pointer'][i] = 1
        x['check_pointer'][i] = 1
        ##Output
        if i == 0:  # i=1 in 1-indexing
            y['pointer_matches_n'][i] = 1
            y['pointer_matches_n_counter'][i] = 1
        else:  # i>1 in 1-indexing (i>=1 in 0-indexing)
            y['pointer_matches_n'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['u_pointer'][i] = 0
        x['n4pointer'][i] = 0
        x['check_pointer'][i] = 1
        ##Output
        if i == 0:  # i=1 in 1-indexing
            y['pointer_matches_n'][i] = 1
            y['pointer_matches_n_counter'][i] = 1
        else:  # i>1 in 1-indexing (i>=1 in 0-indexing)
            y['pointer_matches_n'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['pointer_matches_n'][i] = 1
        x['pointer_matches_n_counter'][i] = 1
        ##Output
        if i < l - 1:  # i < l
            y['pointer_matches_n_counter'][i + 1] = 1
        elif i == l - 1:  # i = l
            y['reset_pointer'] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['pointer_matches_n'][i] = 1
        x['pointer_matches_n_counter'][i] = 0
        ##Output
        y['pointer_matches_n'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for j in range(l+1):

        x, y = get_sample(l, D)
        ##Input
        x['check_pointer_counter'][j] = 1
        ##Output
        if j < l:  # j < l+1 (j ranges 0 to l)
            y['check_pointer_counter'][j + 1] = 1
        elif j == l:  # j = l+1 (j ranges 0 to l, so j=l is the last index)
            y['decide_pointer'] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['decide_pointer'] = 1
    x['reset_pointer'] = 1
    ##Output
    y['next_pointer'][0] = 1
    y['round_addend'][0] = 1
    y['round_sum_counter'][0] = 1  # round-sum-counter[1 (=j, j ∈ [2l])] ← 1
    T.append(x)
    Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['decide_pointer'] = 1
    x['reset_pointer'] = 0
    ##Output
    y['pointer_addend'][0] = 1
    y['pointer_sum_counter'][0] = 1  # pointer-sum-counter[1 (=j, j ∈ [2l])] ← 1
    y['round_sum_counter'][0] = 1  # round-sum-counter[1 (=j, j ∈ [2l])] ← 1
    T.append(x)
    Y.append(flatten_sample(y))

    for j in range(2*l):

        x, y = get_sample(l, D)
        ##Input
        x['pointer_sum_counter'][j] = 1
        ##Output
        if j < 2 * l - 1:  # j < 2l
            y['pointer_sum_counter'][j + 1] = 1
        elif j == 2 * l - 1:  # j = 2l
            for i_val in range(l):  # to-copy-pointer-augend[All (for i)] ← 1
                y['to_copy_pointer_augend'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['round_sum_counter'][j] = 1
        ##Output
        if j < 2 * l - 1:  # j < 2l
            y['round_sum_counter'][j + 1] = 1
        elif j == 2 * l - 1:  # j = 2l
            for i_val in range(l):  # to-copy-round-augend[All (for i)] ← 1
                y['to_copy_round_augend'][i_val] = 1
            for i_val in range(l):  # reset-round[All (for i)] ← 1
                y['reset_round'][i_val] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Incrementing pointer if it has not reached the last node; number of instructions: 5l

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['pointer_augend'][i] = 1
        x['pointer_addend'][i] = 0
        x['to_copy_pointer_augend'][i] = 0
        ##Output
        y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['pointer_augend'][i] = 1
        x['pointer_addend'][i] = 1
        x['to_copy_pointer_augend'][i] = 0
        ##Output
        y['pointer_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['pointer_augend'][i] = 0
        x['pointer_addend'][i] = 1
        x['to_copy_pointer_augend'][i] = 0
        ##Output
        y['pointer_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['pointer_carry'][i] = 1
        ##Output
        if i < l - 1:  # i < l (0-indexed: i < l-1)
            y['pointer_addend'][i+1] = 1
        elif i == l - 1:  # i = l (0-indexed: i = l-1)
            y['pointer_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['pointer_augend'][i] = 1
        x['pointer_addend'][i] = 0
        x['to_copy_pointer_augend'][i] = 1
        ##Output
        y['next_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    
    ## Incrementing round if pointer has reached the last node; number of instructions:5l

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['round_augend'][i] = 1
        x['round_addend'][i] = 0
        x['to_copy_round_augend'][i] = 0
        ##Output
        y['round_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['round_augend'][i] = 1
        x['round_addend'][i] = 1
        x['to_copy_round_augend'][i] = 0
        ##Output
        y['round_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['round_augend'][i] = 0
        x['round_addend'][i] = 1
        x['to_copy_round_augend'][i] = 0
        ##Output
        y['round_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['round_carry'][i] = 1
        ##Output
        if i < l - 1:  # i < l (0-indexed: i < l-1)
            y['round_addend'][i+1] = 1
        elif i == l - 1:  # i = l (0-indexed: i == l-1)
            y['round_carry'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['round_augend'][i] = 1
        x['round_addend'][i] = 0
        x['to_copy_round_augend'][i] = 1
        ##Output
        y['next_round'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Upload next pointer and next round; number of instructions: 4l+1

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['next_pointer'][i] = 1
        x['upload_next_pointer'][i] = 1
        ##Output
        for s_val in range(num_slots):
            y['pointer_placeholder'][s_val][i] = 1
        for s_val in range(num_slots):
            y['persistent_last_pointer'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['next_pointer'][i] = 1
        x['upload_next_pointer'][i] = 0
        ##Output
        y['next_pointer'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['next_round'][i] = 1
        x['upload_next_round'][i] = 1
        ##Output
        for s_val in range(num_slots):
            y['round_placeholder'][s_val][i] = 1
        for s_val in range(num_slots):
            y['persistent_last_round'][s_val][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['next_round'][i] = 1
        x['upload_next_round'][i] = 0
        ##Output
        y['next_round'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    x, y = get_sample(l, D)
    ##Input
    x['upload_message_flag'] = 1
    ##Output
    for s_val in range(num_slots):
        y['message_flag_placeholder'][s_val] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    ## Guide the placeholder content to appropriate sending slot; number of instructions: 11lS+2S

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['determine_slot'][s] = 1
            ##Output
            y['determine_slot'][s] = 1
            for i_val in range(l): y['determine_u_dist_slot'][s][i_val] = 1
            for i_val in range(l): y['determine_u_inf_slot'][s][i_val] = 1
            for i_val in range(l): y['determine_u_id_slot'][s][i_val] = 1
            for i_val in range(l): y['determine_round_slot'][s][i_val] = 1
            for i_val in range(l): y['determine_pointer_slot'][s][i_val] = 1
            y['determine_message_flag_slot'][s] = 1
            T.append(x)
            Y.append(flatten_sample(y))

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
            x['determine_round_slot'][s][i] = 1
            x['round_placeholder'][s][i] = 1
            ##Output
            y['round_slot'][s][i] = 1
            y['determine_round_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_round_slot'][s][i] = 1
            x['round_placeholder'][s][i] = 0
            ##Output
            y['determine_round_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_u_dist_slot'][s][i] = 1
            x['u_dist_placeholder'][s][i] = 1
            ##Output
            y['u_dist_slot'][s][i] = 1
            y['determine_u_dist_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_u_dist_slot'][s][i] = 1
            x['u_dist_placeholder'][s][i] = 0
            ##Output
            y['determine_u_dist_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_u_id_slot'][s][i] = 1
            x['u_id_placeholder'][s][i] = 1
            ##Output
            y['u_id_slot'][s][i] = 1
            y['determine_u_id_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_u_id_slot'][s][i] = 1
            x['u_id_placeholder'][s][i] = 0
            ##Output
            y['determine_u_id_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_u_inf_slot'][s][i] = 1
            x['u_inf_placeholder'][s][i] = 1
            ##Output
            y['u_inf_slot'][s][i] = 1
            y['determine_u_inf_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['determine_u_inf_slot'][s][i] = 1
            x['u_inf_placeholder'][s][i] = 0
            ##Output
            y['determine_u_inf_slot'][s][i] = 1
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



    ## Copy incoming messages to temporary slots while checking if the round or pointer are new; number of instructions: 5lS+S

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['pointer_slot'][s][i] = 1
            ##Output
            y['temp_pointer_slot'][s][i] = 1
            y['incoming_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['round_slot'][s][i] = 1
            ##Output
            y['temp_round_slot'][s][i] = 1
            y['incoming_round'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_id_slot'][s][i] = 1
            ##Output
            y['temp_u_id_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_dist_slot'][s][i] = 1
            ##Output
            y['temp_u_dist_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['u_inf_slot'][s][i] = 1
            ##Output
            y['temp_u_inf_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots):

        x, y = get_sample(l, D)
        ##Input
        x['message_flag_slot'][s] = 1
        ##Output
        y['compare_last_round'][s][l-1] = 1  # l-1 because of 0-indexing (l in pdf = l-1 in python)
        T.append(x)
        Y.append(flatten_sample(y))


    ## Preserve temp data till the new message check is complete; number of instructions: 14lS+S

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
            x['persistent_last_round'][s][i] = 1
            x['overwrite_last_round'][s][i] = 0
            ##Output
            y['last_round'][s][i] = 1
            y['persistent_last_round'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['persistent_last_round'][s][i] = 1
            x['overwrite_last_round'][s][i] = 1
            ##Output
            y['last_round'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_pointer_slot'][s][i] = 1
            x['to_accept_temp_pointer'][s][i] = 0
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['temp_pointer_slot'][s][i] = 1
            y['incoming_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_pointer_slot'][s][i] = 1
            x['to_accept_temp_pointer'][s][i] = 1
            x['overwrite_temp_pointer'][s][i] = 0
            ##Output
            y['delay_new_pointer'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_round_slot'][s][i] = 1
            x['to_accept_temp_round'][s][i] = 0
            x['overwrite_temp_round'][s][i] = 0
            ##Output
            y['temp_round_slot'][s][i] = 1
            y['incoming_round'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_round_slot'][s][i] = 1
            x['to_accept_temp_round'][s][i] = 1
            x['overwrite_temp_round'][s][i] = 0
            ##Output
            y['delay_new_round'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_u_id_slot'][s][i] = 1
            x['to_accept_temp_u_id'][s][i] = 0
            x['overwrite_temp_u_id'][s][i] = 0
            ##Output
            y['temp_u_id_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_u_id_slot'][s][i] = 1
            x['to_accept_temp_u_id'][s][i] = 1
            x['overwrite_temp_u_id'][s][i] = 0
            ##Output
            y['delay_new_u_id'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_u_dist_slot'][s][i] = 1
            x['to_accept_temp_u_dist'][s][i] = 0
            x['overwrite_temp_u_dist'][s][i] = 0
            ##Output
            y['temp_u_dist_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_u_dist_slot'][s][i] = 1
            x['to_accept_temp_u_dist'][s][i] = 1
            x['overwrite_temp_u_dist'][s][i] = 0
            ##Output
            y['delay_new_u_dist'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_u_inf_slot'][s][i] = 1
            x['to_accept_temp_u_inf'][s][i] = 0
            x['overwrite_temp_u_inf'][s][i] = 0
            ##Output
            y['temp_u_inf_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['temp_u_inf_slot'][s][i] = 1
            x['to_accept_temp_u_inf'][s][i] = 1
            x['overwrite_temp_u_inf'][s][i] = 0
            ##Output
            y['delay_new_u_inf'][s][i] = 1
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


    ## Check if the round in the message is new; number of instructions: 6lS

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_round'][s][i] = 1
            x['incoming_round'][s][i] = 1
            x['last_round'][s][i] = 1
            ##Output
            if i == 0:  # i=1 in 1-indexing (i=0 in 0-indexing)
                y['compare_last_pointer'][s][l-1] = 1  # l in 1-indexing (l-1 in 0-indexing)
            elif i > 0:  # i>1 in 1-indexing (i>0 in 0-indexing)
                y['compare_last_round'][s][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_round'][s][i] = 1
            x['incoming_round'][s][i] = 0
            x['last_round'][s][i] = 0
            ##Output
            if i == 0:  # i=1 in 1-indexing (i=0 in 0-indexing)
                y['compare_last_pointer'][s][l-1] = 1  # l in 1-indexing (l-1 in 0-indexing)
            elif i > 0:  # i>1 in 1-indexing (i>0 in 0-indexing)
                y['compare_last_round'][s][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_round'][s][i] = 1
            x['incoming_round'][s][i] = 1
            x['last_round'][s][i] = 0
            ##Output
            y['new_message'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_round'][s][i] = 1
            x['incoming_round'][s][i] = 0
            x['last_round'][s][i] = 1
            ##Output
            y['old_message'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['new_message'][s][i] = 1
            ##Output
            if i > 0:  # i>1 in 1-indexing (i>0 in 0-indexing)
                y['new_message'][s][i-1] = 1
            elif i == 0:  # i=1 in 1-indexing (i=0 in 0-indexing)
                for i_val in range(l): y['to_accept_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['to_accept_temp_round'][s][i_val] = 1
                for i_val in range(l): y['to_accept_temp_u_id'][s][i_val] = 1
                for i_val in range(l): y['to_accept_temp_u_dist'][s][i_val] = 1
                for i_val in range(l): y['to_accept_temp_u_inf'][s][i_val] = 1
                y['return_message_flag'][s] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['old_message'][s][i] = 1
            ##Output
            if i > 0:  # i>1 in 1-indexing (i>0 in 0-indexing)
                y['old_message'][s][i-1] = 1
            elif i == 0:  # i=1 in 1-indexing (i=0 in 0-indexing)
                for i_val in range(l): y['overwrite_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_round'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_id'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_dist'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_inf'][s][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## In the case that round is not new, check if the pointer in the message is new; number of instructions: 3lS

    for i in range(l):
        for s in range(num_slots):

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['incoming_pointer'][s][i] = 1
            x['last_pointer'][s][i] = 1
            ##Output
            if i == 0:  # i = 1 in 1-indexing
                for i_val in range(l): y['overwrite_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_round'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_id'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_dist'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_inf'][s][i_val] = 1
            elif i > 0:  # i > 1 in 1-indexing
                y['compare_last_pointer'][s][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['incoming_pointer'][s][i] = 0
            x['last_pointer'][s][i] = 0
            ##Output
            if i == 0:  # i = 1 in 1-indexing
                for i_val in range(l): y['overwrite_temp_pointer'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_round'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_id'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_dist'][s][i_val] = 1
                for i_val in range(l): y['overwrite_temp_u_inf'][s][i_val] = 1
            elif i > 0:  # i > 1 in 1-indexing
                y['compare_last_pointer'][s][i-1] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['compare_last_pointer'][s][i] = 1
            x['incoming_pointer'][s][i] = 1
            x['last_pointer'][s][i] = 0
            ##Output
            y['new_message'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Take new messages from delays and put it into received messages as well as put it back in to placeholder; number of instructions: 5l(S+1) +(S+1)

    for i in range(l):
        for s in range(num_slots+1):

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_round'][s][i] = 1
            x['shutdown_round_delay'][s][i] = 0
            ##Output
            if s < num_slots :  # num_slots = D^2 + 1
                y['delay_new_round'][s+1][i] = 1
            elif s == num_slots :  # s = D^2 + 1
                y['received_round'][i] = 1
                for s_val in range(num_slots):
                    y['round_placeholder'][s_val][i] = 1
                for s_val in range(num_slots):
                    y['persistent_last_round'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_pointer'][s][i] = 1
            x['shutdown_pointer_delay'][s][i] = 0
            ##Output
            if s < num_slots :  # num_slots = D^2 + 1
                y['delay_new_pointer'][s+1][i] = 1
            elif s == num_slots :  # s = D^2 + 1
                y['received_pointer'][i] = 1
                for s_val in range(num_slots):
                    y['pointer_placeholder'][s_val][i] = 1
                for s_val in range(num_slots):
                    y['persistent_last_pointer'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_u_id'][s][i] = 1
            x['shutdown_u_id_delay'][s][i] = 0
            ##Output
            if s < num_slots :  # num_slots = D^2 + 1
                y['delay_new_u_id'][s+1][i] = 1
            elif s == num_slots :  # s = D^2 + 1
                for d_val in range(D):
                    y['received_u_id'][d_val][i] = 1
                for s_val in range(num_slots):
                    y['u_id_placeholder'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_u_dist'][s][i] = 1
            x['shutdown_u_dist_delay'][s][i] = 0
            ##Output
            if s < num_slots :  # num_slots = D^2 + 1
                y['delay_new_u_dist'][s+1][i] = 1
            elif s == num_slots :  # s = D^2 + 1
                for d_val in range(D):
                    y['received_u_dist'][d_val][i] = 1
                for s_val in range(num_slots):
                    y['u_dist_placeholder'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['delay_new_u_inf'][s][i] = 1
            x['shutdown_u_inf_delay'][s][i] = 0
            ##Output
            if s < num_slots :  # num_slots = D^2 + 1
                y['delay_new_u_inf'][s+1][i] = 1
            elif s == num_slots :  # s = D^2 + 1
                for d_val in range(D):
                    y['received_u_inf'][d_val][i] = 1
                for s_val in range(num_slots):
                    y['u_inf_placeholder'][s_val][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for s in range(num_slots+1):
        x, y = get_sample(l, D)
        ##Input
        x['delay_new_message_flag'][s] = 1
        x['shutdown_message_flag_delay'][s] = 0
        ##Output
        if s < num_slots - 1:  # num_slots = D^2 + 1
            y['delay_new_message_flag'][s+1] = 1
        elif s == num_slots - 1:  # s = D^2 + 1
            y['delay_new_message_flag'][s+1] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['overwrite_last_pointer'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['overwrite_last_round'][s_val][i_val] = 1
            for s_val in range(num_slots):
                y['shutdown_message_flag_delay'][s_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_pointer_delay'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_round_delay'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_u_id_delay'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_u_dist_delay'][s_val][i_val] = 1
            for s_val in range(num_slots):
                for i_val in range(l):
                    y['shutdown_u_inf_delay'][s_val][i_val] = 1
        elif s == num_slots :  # s = D^2 + 2
            for s_val in range(num_slots):
                y['message_flag_placeholder'][s_val] = 1
            y['received_new_message'] = 1
            for d_val in range(D):
                for i_val in range(l):
                    y['check_v_id_with_received_u_id'][d_val][i_val] = 1
            for d_val in range(D):
                y['v_id_comparison_counter'][d_val][0] = 1  # j=1 in 1-indexing becomes j=0 in 0-indexing

        T.append(x)
        Y.append(flatten_sample(y))

    ## Check local ids with the id in the message and set update flag for the matching one; number of instructions: 6lD + D + l + 2

    x, y = get_sample(l, D)
    ##Input
    x['received_new_message'] = 1
    ##Output
    y['update_round_counter'][0] = 1  # j=1 (0-indexed)
    T.append(x)
    Y.append(flatten_sample(y))

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['v_id'][d][i] = 1
            ##Output
            y['v_id'][d][i] = 1
            y['v_id2check'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_u_id'][d][i] = 1
            x['v_id2check'][d][i] = 1
            x['check_v_id_with_received_u_id'][d][i] = 1
            ##Output
            if i == 0:  # i=1 (0-indexed)
                y['v_equals_received_u_id'][d][i] = 1
                y['v_is_received_u_id_counter'][d][i] = 1
            else:  # i>1
                y['v_equals_received_u_id'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_u_id'][d][i] = 0
            x['v_id2check'][d][i] = 0
            x['check_v_id_with_received_u_id'][d][i] = 1
            ##Output
            if i == 0:  # i=1 (0-indexed)
                y['v_equals_received_u_id'][d][i] = 1
                y['v_is_received_u_id_counter'][d][i] = 1
            else:  # i>1
                y['v_equals_received_u_id'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_equals_received_u_id'][d][i] = 1
            x['v_is_received_u_id_counter'][d][i] = 1
            x['reset_v_equals_u'][d][i] = 0
            ##Output
            if i < l-1:  # i < l (0-indexed)
                y['v_is_received_u_id_counter'][d][i+1] = 1
            elif i == l-1:  # i = l (0-indexed)
                for i_val in range(l):
                    y['overwrite_v_inf'][d][i_val] = 1
                for i_val in range(l):
                    y['overwrite_v_dist'][d][i_val] = 1
                for i_val in range(l):
                    y['v_dist_update'][d][i_val] = 1
                for i_val in range(l):
                    y['v_inf_update'][d][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['v_equals_received_u_id'][d][i] = 1
            x['v_is_received_u_id_counter'][d][i] = 0
            x['reset_v_equals_u'][d][i] = 0
            ##Output
            y['v_equals_received_u_id'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for j in range(l+1):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['v_id_comparison_counter'][d][j] = 1
            ##Output
            if j < l:  # j < l+1 (0-indexed: j ranges 0 to l)
                y['v_id_comparison_counter'][d][j+1] = 1
            elif j == l:  # j = l+1 (0-indexed)
                for i_val in range(l):
                    y['reset_v_equals_u'][d][i_val] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    for j in range(l+1):

        x, y = get_sample(l, D)
        ##Input
        x['update_round_counter'][j] = 1
        ##Output
        if j < l-1:  # j < l (0-indexed: j ranges 0 to l)
            y['update_round_counter'][j+1] = 1
        elif j == l-1:  # j = l (0-indexed)
            for i_val in range(l):
                y['update_round'][i_val] = 1
            for i_val in range(l):
                y['reset_round'][i_val] = 1
            y['update_round_counter'][j+1] = 1
        elif j == l:  # j = l+1 (0-indexed)
            y['compare_round'][l-1] = 1  # l (0-indexed)
            for d_val in range(D):
                for i_val in range(l):
                    y['renew_received_u_dist'][d_val][i_val] = 1  
            for d_val in range(D):
                for i_val in range(l):
                    y['renew_received_u_inf'][d_val][i_val] = 1  
        T.append(x)
        Y.append(flatten_sample(y))


    ## Update the distance information that the node has about its neighbors; number of instructions: 4lD

    for i in range(l):
        for d in range(D):

            x, y = get_sample(l, D)
            ##Input
            x['received_u_dist'][d][i] = 1
            x['v_dist_update'][d][i] = 0
            x['renew_received_u_dist'][d][i] = 0
            ##Output
            y['received_u_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_u_dist'][d][i] = 1
            x['v_dist_update'][d][i] = 1
            x['renew_received_u_dist'][d][i] = 0
            ##Output
            y['v_dist'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_u_inf'][d][i] = 1
            x['v_inf_update'][d][i] = 0
            x['renew_received_u_inf'][d][i] = 0
            ##Output
            y['received_u_inf'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l, D)
            ##Input
            x['received_u_inf'][d][i] = 1
            x['v_inf_update'][d][i] = 1
            x['renew_received_u_inf'][d][i] = 0
            ##Output
            y['v_inf_backup'][d][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

    ## Put the incoming round in the comparison blocks and update; number of instructions: 2l

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['received_round'][i] = 1
        x['update_round'][i] = 0
        ##Output
        y['received_round'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_round'][i] = 1
        x['update_round'][i] = 1
        ##Output
        y['backup_round'][i] = 1
        y['round'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Setting up round for checking if it is less than $|V|$ for incrementing of pointer has reached the end; number of instructions: 2l

    for i in range(l):

        x, y = get_sample(l, D)
        ##Input
        x['backup_round'][i] = 1
        x['setup_round_augend'][i] = 1
        x['reset_round'][i] = 0
        ##Output
        y['backup_round'][i] = 1
        y['round'][i] = 1
        y['round_augend'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['backup_round'][i] = 1
        x['setup_round_augend'][i] = 0
        x['reset_round'][i] = 0
        ##Output
        y['backup_round'][i] = 1
        y['round'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    ## Put the incoming pointer in the comparison blocks and update; number of instructions: 3l

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
        y['pointer'][i] = 1
        y['compare_u_id'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l, D)
        ##Input
        x['received_pointer'][i] = 0
        x['update_pointer'][i] = 1
        ##Output
        y['compare_u_id'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))



    
    
    return T, np.array(Y)

