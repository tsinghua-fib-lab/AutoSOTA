# rot_update.py

# function_update/rot_update: inference-only, not differentiable
# torch_function_update/torch_rot_update: differentiable path
 
import torch
import numpy as np

def turn_into_dict(output, all_local_band_indexes, all_distances_to_central):
    B = len(all_local_band_indexes)
    output_dict = []
    for i in range(B):
         output_dict.append({
            "updated function": output[i],
            "local band indexes": all_local_band_indexes[i],
            "distances to central": all_distances_to_central[i]
        })
    return output_dict

def build_Global_dico(all_infos_updated_function):
    '''
    Constructs a global dictionary that aggregates function values and distances for each band point.

    For each local update result, this function populates a dictionary mapping each band point index
    to a list of all function values predicted for it, along with the corresponding distances to 
    their central surface points.

            Parameters:
                    all_infos_updated_function (List[Dict]): List of dictionaries containing:
                        - "updated function": Updated function values (local_size,)
                        - "local band indexes": Corresponding band point indices (local_size,)
                        - "distances to central": Distances to the central point (local_size,)

            Returns:
                    dict: A dictionary {index: [list_of_values, list_of_distances]}, where each index corresponds 
                          to a band point and the lists contain all values and distances accumulated for that point.
    '''
    Global_dico = {}

    for info in all_infos_updated_function:
        updated_function = info["updated function"]
        local_band_indexes = info["local band indexes"]
        distances_to_central = info["distances to central"]

        for val, idx, dist in zip(updated_function, local_band_indexes, distances_to_central):
            if idx not in Global_dico:
                Global_dico[idx] = [[], []]
            Global_dico[idx][0].append(val)
            Global_dico[idx][1].append(dist)

    return Global_dico

def torch_build_Global_dico(all_infos_updated_function):
    '''
    Constructs a global dictionary that aggregates function values and distances for each band point.

    For each local update result, this function populates a dictionary mapping each band point index
    to a list of all function values predicted for it, along with the corresponding distances to 
    their central surface points.

            Parameters:
                    all_infos_updated_function (List[Dict]): List of dictionaries containing:
                        - "updated function": Updated function values (local_size,)
                        - "local band indexes": Corresponding band point indices (local_size,)
                        - "distances to central": Distances to the central point (local_size,)

            Returns:
                    dict: A dictionary {index: [list_of_values, list_of_distances]}, where each index corresponds 
                          to a band point and the lists contain all values and distances accumulated for that point.
    '''
    Global_dico = {}

    for info in all_infos_updated_function:
        updated_function = info["updated function"]
        local_band_indexes = info["local band indexes"]
        distances_to_central = info["distances to central"]

        for val, idx, dist in zip(updated_function, local_band_indexes, distances_to_central):
            idx = int(idx.item()) if torch.is_tensor(idx) else int(idx)   # <-- FIX
            dist = float(dist.item()) if torch.is_tensor(dist) else float(dist)

            if idx not in Global_dico:
                Global_dico[idx] = [[], []]
            Global_dico[idx][0].append(val)
            Global_dico[idx][1].append(dist)

    return Global_dico

def final_updated_function_value(Global_dico, temperature=0.1):
    '''
    Compute the final function values at each band point using weighted averaging,
    where closer predictions contribute more strongly based on inverse-distance softmax weighting.

            Parameters:
                    Global_dico (dict): Dictionary of the form {idx: [list_of_values, list_of_distances]},
                                        where each key corresponds to a band point index, and the values are
                                        lists of predicted values and distances from local centers.
                    temperature (float): Controls the sharpness of the weighting. Lower values emphasize
                                         closer distances more strongly.

            Returns:
                    np.ndarray: Final array of weighted function values for all band points.
    '''
    N = len(Global_dico) # is equal to the number of band points
    final_values = np.zeros(N)

    for idx, (vals, dists) in Global_dico.items():
        vals = np.array(vals)
        dists = np.array(dists)

        weights = np.exp(-dists / temperature)
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            raise ValueError(f"Sum of weights is zero, you might need to increase the temperature to avoid instability.")
        weights /= sum_weights

        result = np.sum(weights * vals)
        final_values[idx] = result
    
    return final_values

def torch_final_updated_function_value(Global_dico, temperature=0.1):
    """
    Torch version of final_updated_function_value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = len(Global_dico)
    final_values = torch.zeros(N, device=device)

    for idx, (vals, dists) in Global_dico.items():
        vals_t = torch.stack([
            v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device)
            for v in vals
        ])

        dists_t = torch.stack([
            d.to(device) if torch.is_tensor(d) else torch.tensor(d, device=device)
            for d in dists
        ])

        weights = torch.softmax(-dists_t / temperature, dim=0)

        result = torch.sum(weights * vals_t)
        final_values[idx] = result

    return final_values

def function_update(neural_weights, band_values, all_local_band_indexes, all_distances_to_central, temperature=0.0423):
    """
    """
    band_local_values = []
    for i in range(len(all_local_band_indexes)):
        idx = all_local_band_indexes[i]
        band_local_values.append(torch.tensor(band_values[idx], dtype=torch.float32).unsqueeze(0)) # (1, local_size)
    band_local_values = torch.stack(band_local_values, dim=0)   # (B, 1, local_size)

    input = band_local_values.unsqueeze(-1).to(neural_weights.device)             
    output = torch.matmul(neural_weights, input).squeeze(-1).squeeze(1).detach().cpu().numpy()  


    all_infos_updated_function = turn_into_dict(output, all_local_band_indexes, all_distances_to_central)

    Global_dico = build_Global_dico(all_infos_updated_function)

    update_band = final_updated_function_value(Global_dico, temperature=temperature)

    return update_band

def torch_function_update(neural_weights, band_values, all_local_band_indexes, all_distances_to_central, temperature=0.0423):
    """
    Torch version of function_update (same structure).
    """
    if not torch.is_tensor(band_values):
        band_values = torch.tensor(band_values, dtype=torch.float32, device=neural_weights.device)
    else:
        band_values = band_values.to(neural_weights.device).float()

    band_local_values = []
    for i in range(len(all_local_band_indexes)):
        idx = all_local_band_indexes[i]
        band_local_values.append(band_values[idx].unsqueeze(0).float()) 

    band_local_values = torch.stack(band_local_values, dim=0) 

    inp = band_local_values.unsqueeze(-1)  

    output = torch.matmul(neural_weights, inp).squeeze(-1).squeeze(1)  

    all_infos_updated_function = turn_into_dict(output, all_local_band_indexes, all_distances_to_central)
    Global_dico = torch_build_Global_dico(all_infos_updated_function)
    update_band = torch_final_updated_function_value(Global_dico, temperature=temperature)

    return update_band

def rot_update(neural_weights, u, all_local_band_indexes, all_distances_to_central, temperature=0.0423):
    values = []
    for rot_num in range(len(neural_weights)):
        update = function_update(
            neural_weights[rot_num], u, all_local_band_indexes, all_distances_to_central, temperature=temperature)
        values.append(update)
    values = np.stack(values, axis=0)

    mean_all = values.mean(axis=0) 
    
    min_vals = values.min(axis=0)
    max_vals = values.max(axis=0)
    sum_vals = values.sum(axis=0) - min_vals - max_vals
    mean_trimmed = sum_vals / (values.shape[0] - 2)
    
    # For the paper we didn't used mean_trimmed
    return mean_all, mean_trimmed

def torch_rot_update(neural_weights, u, all_local_band_indexes, all_distances_to_central, temperature=0.0423):
    """
    Differentiable rotation-ensemble update.
    Returns:
        mean_all: (N,) torch.Tensor
        mean_trimmed: (N,) torch.Tensor (trim min/max over rotations)
    """
    vals = []
    for rot_num in range(len(neural_weights)):
        upd = torch_function_update(
            neural_weights[rot_num],
            u,
            all_local_band_indexes,
            all_distances_to_central,
            temperature=temperature,
        )  
        vals.append(upd)

    values = torch.stack(vals, dim=0) 

    mean_all = values.mean(dim=0)    

    min_vals, _ = values.min(dim=0)   
    max_vals, _ = values.max(dim=0)   
    sum_vals = values.sum(dim=0) - min_vals - max_vals
    mean_trimmed = sum_vals / (values.shape[0] - 2)
    
    # For the paper we didn't used mean_trimmed
    return mean_all, mean_trimmed