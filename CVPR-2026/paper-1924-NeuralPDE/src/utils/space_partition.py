# space_partition.py

import numpy as np

def extract_point_bounding_box(surface_points, band_points, idx_local_band, expansion_factor = 0.5, 
                               priority_factor = 0.5):
    '''
    Extract points that lie within an expanded bounding box defined by band_points.
    Parameters:
    
    band_points (np.ndarray): An array of shape (N, 3).
    idx_local_band (list or np.ndarray): Indices of local band points to define the bounding box.
    surface_points (np.ndarray): An array of shape (M, 3) containing points
    expansion_factor (float): Factor to expand the bounding box.
    priority_factor: Factor for priority. close to 0 means no priority, close to expansion_factor means 
                     that we want poiints far from middle.

    Returns: The masks for surface_points and band_points within the expanded bounding box.
    '''
    local_band_points = band_points[idx_local_band, :]

    min_bounds = np.min(local_band_points, axis=0)
    max_bounds = np.max(local_band_points, axis=0)

    distance = max_bounds - min_bounds
    
    min_bounds_expan = min_bounds - expansion_factor * distance
    max_bounds_expan = max_bounds + expansion_factor * distance

    large_surface_mask = np.all((surface_points >= min_bounds_expan) & (surface_points <= max_bounds_expan), axis=1)
    large_band_mask = np.all((band_points >= min_bounds_expan) & (band_points <= max_bounds_expan), axis=1)

    min_bounds_priority = min_bounds - priority_factor * distance
    max_bounds_priority = max_bounds + priority_factor * distance
    
    large_local_band = band_points[large_band_mask, :]
    small_mask_to_start = np.all((large_local_band <= min_bounds_priority) | (large_local_band >= max_bounds_priority), 
                           axis=1)
        
    mask_to_start = np.zeros(band_points.shape[0], dtype=bool)
    mask_to_start[large_band_mask] = small_mask_to_start

    return large_surface_mask, large_band_mask, mask_to_start

def build_a_local_part(starting_idx, surface_points, band_points, tree_surface_points, tree_band_points, local_size, 
                       expansion_factor, priority_factor):
    # Start by finding the cp on the surface 
    _, idx_cp_surface = tree_surface_points.query(band_points[starting_idx, :], k=1)
    cp_surface = surface_points[idx_cp_surface, :]

    # Extract local band points
    distances_to_central, idx_local_band = tree_band_points.query(cp_surface, k=local_size)

    # Extract surface points in the bounding box of the local band points
    large_surface_mask, large_band_mask, mask_to_start = extract_point_bounding_box(
        surface_points, band_points, idx_local_band, expansion_factor=expansion_factor, 
        priority_factor=priority_factor)

    return idx_local_band, large_surface_mask, large_band_mask, mask_to_start, distances_to_central, idx_cp_surface

class Extractor:
    def __init__(self, surface_points, band_points, tree_surface_points, tree_band_points, 
                 local_size, local_band_indexes, large_surface_masks, large_band_masks, masks_to_start, 
                 expansion_factor, priority_factor, all_distances_to_central, indexes_central_points):
        self.surface_points = surface_points
        self.band_points = band_points
        self.tree_band_points = tree_band_points
        self.local_size = local_size
        self.expansion_factor = expansion_factor
        self.priority_factor = priority_factor
        self.tree_surface_points = tree_surface_points

        # listes à mettre à jour
        self.local_band_indexes = local_band_indexes
        self.large_surface_masks = large_surface_masks
        self.large_band_masks = large_band_masks
        self.masks_to_start = masks_to_start
        self.all_distances_to_central = all_distances_to_central
        self.indexes_central_points = indexes_central_points

    def __call__(self, idx):
        # extraire et calculer
        idx_local_band, large_surface_mask, large_band_mask, mask_to_start, distances_to_central, idx_cp_surface = build_a_local_part(
            idx, self.surface_points, self.band_points, self.tree_surface_points, self.tree_band_points, 
            self.local_size, self.expansion_factor, self.priority_factor)
        
        # mettre à jour les listes
        self.local_band_indexes.append(idx_local_band)
        self.large_surface_masks.append(large_surface_mask)
        self.large_band_masks.append(large_band_mask)
        self.masks_to_start.append(mask_to_start)
        self.all_distances_to_central.append(distances_to_central)
        self.indexes_central_points.append(idx_cp_surface)

def large_mask_wo_local_parts(local_band_index, large_band_mask):
    n = large_band_mask.shape[0]

    mask = np.zeros(n, dtype=bool)
    mask[large_band_mask] = True
    mask[local_band_index] = False

    return mask

def update_mask(list_of_masks, local_band_indexes):
    n = list_of_masks[0].shape[0]

    mask = np.zeros(n, dtype=bool)
    mask[np.concatenate(local_band_indexes)] = True    
    
    return np.logical_or.reduce(list_of_masks)  & ~mask

def update_priority_mask(list_of_masks, mask_to_empty):
    union = np.logical_or.reduce(list_of_masks)
    return union & mask_to_empty

def small_update_priority_mask(priority_mask, mask_to_start, large_band_mask):
    mask_to_remove = large_band_mask & ~mask_to_start
    return priority_mask & ~mask_to_remove

def space_partition(surface_points, band_points, tree_surface_points, tree_band_points, local_size, mask_threshold, 
                    expansion_factor = 0.5, priority_factor = 0.27, max_iters=200000):
    it = 0
    if priority_factor >= expansion_factor:
        print("Warning: priority_factor should be smaller than expansion_factor. Otherwise, the algorithm will not " \
        "defined priority points.")

    print("BE SURE THAT THE CONDITION WITH DELTA X, EPSILON AND LOCAL SIZE IS SATISFIED! " \
    "OTHERWISE THE ALGORITHM MAY NEVER ENDS.")

    local_band_indexes = []
    all_distances_to_central = []

    # Each list correspond to a local part, they give the informations about the  
    # surface points that belong to the bounding box of the local band points.
    large_surface_masks = []
    large_band_masks = [] 
    masks_to_start = []
    indexes_central_points = []

    # Pick a true index in the mask_threshold just to start the algorithm
    if not np.any(mask_threshold):
        raise ValueError("mask_threshold is empty (all False). Cannot start space_partition.")
    starting_idx = np.argmax(mask_threshold)

    # This is done to compute local parts and mask wrt an idx of a point in the band
    # and it adds the results to the lists
    extractor = Extractor(surface_points, band_points, tree_surface_points, tree_band_points, 
                 local_size, local_band_indexes, large_surface_masks, large_band_masks, masks_to_start, 
                 expansion_factor, priority_factor, all_distances_to_central, indexes_central_points)
    
    # Step 1: 
    extractor(starting_idx)

    # We define the mask in the step 1
    # This mask defined the priority points, i.e those far from the local parts
    priority_mask = masks_to_start[-1] & mask_threshold

    # This mask defined the large band without the points already in a local part
    mask_to_empty = update_mask(large_band_masks, local_band_indexes)
    
    while mask_to_empty.any():
        it += 1
        while priority_mask.any():
            new_idx = np.argmax(priority_mask)
            extractor(new_idx)

            # We update the priority mask
            priority_mask = small_update_priority_mask(priority_mask, masks_to_start[-1], large_band_masks[-1])

        while mask_to_empty.any():
            new_idx = np.argmax(mask_to_empty)
            extractor(new_idx)
            mask_to_empty[local_band_indexes[-1]] = False

        # We update the mask_to_empty by adding the new large_band_masks
        mask_to_empty = update_mask(large_band_masks, local_band_indexes)
        # We update the priority_mask by adding the new masks_to_start
        priority_mask = update_priority_mask(masks_to_start, mask_to_empty) & mask_threshold

        if it > max_iters:
            raise RuntimeError("space_partition did not converge: check (dx, epsilon, local_size) coverage condition. Or increase max_iters.")

    return local_band_indexes, large_surface_masks, all_distances_to_central, indexes_central_points