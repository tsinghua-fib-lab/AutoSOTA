"""
Clustering utils code for SAVVY pipeline egocentric tracks / static object tracks merging - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import math
import numpy as np


def calculate_centroid(points):
    """Calculate the centroid of a set of points."""
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    return (round(x_sum / len(points), 2), round(y_sum / len(points), 2))


# Alternative approach: using confidence as weight for centroid calculation
def calculate_weighted_centroid(list_to_merge, temperature=1.0):
    """
    Calculate a single centroid with points weighted by softmax of their confidence scores.
    Temperature controls the sharpness of the softmax distribution.
    
    Args:
        list_to_merge: List of [location, confidence] pairs
        temperature: Controls the softness of the softmax (higher = more uniform weights)
                    Range: 0.1 to 10.0 recommended
    
    Returns:
        A single weighted centroid (x, y)
    """
    # Extract confidence scores
    confidence_scores = np.array([confidence for _, confidence in list_to_merge])
    
    # Apply temperature-scaled softmax to confidence scores
    exp_scores = np.exp(confidence_scores / temperature)  # Temperature scaling
    softmax_weights = exp_scores / np.sum(exp_scores)
    
    weighted_x_sum = 0
    weighted_y_sum = 0
    
    for i, (point, _) in enumerate(list_to_merge):
        x, y = point
        weight = softmax_weights[i]
        weighted_x_sum += x * weight
        weighted_y_sum += y * weight
    
    return (round(weighted_x_sum, 2), round(weighted_y_sum, 2)), np.max(confidence_scores)

def get_possible_locs(cur_angle, cur_distance, ori_center, forward_vec, half_frustum=45, distance_shift=0.5, num_angles=10, num_distances=5, max_valid_distance=5):
    try:
        # base_angle = (-float(cur_angle) + 270) % 360
        base_angle = cur_angle
        angle_range = (base_angle - half_frustum, base_angle + half_frustum)
        if cur_distance is None:
            num_distances = 10 # Sample 5 different distances within the ±1m range
            distance_range = np.linspace(0.1, max_valid_distance+distance_shift, num_distances)
        else:
            cur_distance = float(cur_distance)
            distance_range = np.linspace(max(0.1, cur_distance-distance_shift), cur_distance+distance_shift, num_distances)
        # Create a set of possiblepositions based on the frustum
        possible_positions = set()
        angles = np.linspace(angle_range[0], angle_range[1], num_angles)
        forward_angle = math.degrees(math.atan2(forward_vec[1], forward_vec[0]))
        # forward_angle = (180 + forward_angle) % 360
        # Create points at each combination of angle and distance
        for angle in angles:
            for dist in distance_range:
                # Adjust angle based on our new base angle
                adjusted_angle = math.radians(-angle + forward_angle)
                x = ori_center[0] + dist * math.cos(adjusted_angle)
                y = ori_center[1] + dist * math.sin(adjusted_angle)
                possible_positions.add((round(x, 2), round(y, 2)))
        return np.array(list(possible_positions))
    except:
        return None


def aggregate_final_loc(obj_loc, grid_resolution=0.5, min_traj_num=1):
    final_obj = {}
    for object_name in obj_loc.keys():   
        valid_traj = obj_loc[object_name]
        if len(valid_traj) > min_traj_num:
            # Determine bounds of the grid based on valid trajectory points
            all_points = [point for traj in valid_traj for point in traj]
            min_x = min(p[0] for p in all_points) - 1
            max_x = max(p[0] for p in all_points) + 1
            min_y = min(p[1] for p in all_points) - 1
            max_y = max(p[1] for p in all_points) + 1
            
            # Create grid
            x_grid = np.arange(min_x, max_x, grid_resolution)
            y_grid = np.arange(min_y, max_y, grid_resolution)
            
            # Initialize grid coverage (how many objects cover each cell)
            grid_coverage = {}
            
            # For each object's trajectory, mark grid cells as covered
            for traj_idx, traj_points in enumerate(valid_traj):
                # Convert discrete points to a continuous region
                # For each point in the trajectory, mark grid cells within a certain radius
                radius = grid_resolution  # Adjust based on point density
                
                for point in traj_points:
                    # Mark grid cells around this point
                    x_min_idx = int((point[0] - radius - min_x) / grid_resolution)
                    x_max_idx = int((point[0] + radius - min_x) / grid_resolution) + 1
                    y_min_idx = int((point[1] - radius - min_y) / grid_resolution)
                    y_max_idx = int((point[1] + radius - min_y) / grid_resolution) + 1
                    
                    for x_idx in range(max(0, x_min_idx), min(len(x_grid), x_max_idx)):
                        for y_idx in range(max(0, y_min_idx), min(len(y_grid), y_max_idx)):
                            grid_cell = (x_idx, y_idx)
                            if grid_cell not in grid_coverage:
                                grid_coverage[grid_cell] = set()
                            grid_coverage[grid_cell].add(traj_idx)
            
            # Find cells with maximum coverage
            max_coverage = 0
            max_coverage_cells = []
            
            for cell, coverage in grid_coverage.items():
                if len(coverage) > max_coverage:
                    max_coverage = len(coverage)
                    max_coverage_cells = [cell]
                elif len(coverage) == max_coverage:
                    max_coverage_cells.append(cell)
            
            # If we found cells with coverage from multiple objects
            if max_coverage >= 2 and max_coverage_cells:
                # Convert grid indices back to real coordinates
                # and calculate the centroid of the best overlapped region
                x_sum = sum(x_grid[cell[0]] for cell in max_coverage_cells)
                y_sum = sum(y_grid[cell[1]] for cell in max_coverage_cells)
                centroid = (round(x_sum / len(max_coverage_cells), 2), round(y_sum / len(max_coverage_cells), 2))
                
                # Store the final position
                final_obj[object_name] = centroid
            else:
                x_sum = sum(cell[0] for cell in all_points)
                y_sum = sum(cell[1] for cell in all_points)
                centroid = (round(x_sum / len(all_points), 2), round(y_sum / len(all_points), 2))
                final_obj[object_name] = centroid
    return final_obj


def aggregate_final_loc_simple(valid_traj, grid_resolution=0.5, min_traj_num=1):
    valid_traj = [[np.array(item) for item in valid_traj]]
    if len(valid_traj) > min_traj_num:
        # Determine bounds of the grid based on valid trajectory points
        all_points = [point for traj in valid_traj for point in traj]
        min_x = min(p[0] for p in all_points) - 1
        max_x = max(p[0] for p in all_points) + 1
        min_y = min(p[1] for p in all_points) - 1
        max_y = max(p[1] for p in all_points) + 1
        
        # Create grid
        x_grid = np.arange(min_x, max_x, grid_resolution)
        y_grid = np.arange(min_y, max_y, grid_resolution)
        
        # Initialize grid coverage (how many objects cover each cell)
        grid_coverage = {}
        
        # For each object's trajectory, mark grid cells as covered
        for traj_idx, traj_points in enumerate(valid_traj):
            # Convert discrete points to a continuous region
            # For each point in the trajectory, mark grid cells within a certain radius
            radius = grid_resolution  # Adjust based on point density
            
            for point in traj_points:
                # Mark grid cells around this point
                x_min_idx = int((point[0] - radius - min_x) / grid_resolution)
                x_max_idx = int((point[0] + radius - min_x) / grid_resolution) + 1
                y_min_idx = int((point[1] - radius - min_y) / grid_resolution)
                y_max_idx = int((point[1] + radius - min_y) / grid_resolution) + 1
                
                for x_idx in range(max(0, x_min_idx), min(len(x_grid), x_max_idx)):
                    for y_idx in range(max(0, y_min_idx), min(len(y_grid), y_max_idx)):
                        grid_cell = (x_idx, y_idx)
                        if grid_cell not in grid_coverage:
                            grid_coverage[grid_cell] = set()
                        grid_coverage[grid_cell].add(traj_idx)
        
        # Find cells with maximum coverage
        max_coverage = 0
        max_coverage_cells = []
        
        for cell, coverage in grid_coverage.items():
            if len(coverage) > max_coverage:
                max_coverage = len(coverage)
                max_coverage_cells = [cell]
            elif len(coverage) == max_coverage:
                max_coverage_cells.append(cell)
        
        # If we found cells with coverage from multiple objects
        if max_coverage >= 2 and max_coverage_cells:
            # Convert grid indices back to real coordinates
            # and calculate the centroid of the best overlapped region
            x_sum = sum(x_grid[cell[0]] for cell in max_coverage_cells)
            y_sum = sum(y_grid[cell[1]] for cell in max_coverage_cells)
            centroid = (round(x_sum / len(max_coverage_cells), 2), round(y_sum / len(max_coverage_cells), 2))
        else:
            x_sum = sum(cell[0] for cell in all_points)
            y_sum = sum(cell[1] for cell in all_points)
            centroid = (round(x_sum / len(all_points), 2), round(y_sum / len(all_points), 2))
        return centroid
    
    return valid_traj[0][0]



def calculate_centroid_cluster(points, max_distance=1):
    """
    Calculate the centroid of a set of points, clustering by distance.
    
    Args:
        points: List of points, where each point is [x, y]
        max_distance: Maximum distance for points to be considered in the same cluster
                      If None, a reasonable default will be calculated
    
    Returns:
        numpy array [x, y] representing the centroid of the largest cluster,
        or the centroid of all points if there's no clear majority cluster
    """
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    # Convert points to numpy array if not already
    points_array = np.array(points)
    
    if len(points) <= 1:
        # Handle edge cases
        return np.array(points[0]) if len(points) == 1 else np.array([0, 0])
    
    # Set default max_distance if not provided
    if max_distance is None:
        # Use a heuristic: average of 10% of the max pairwise distance
        x_range = max(p[0] for p in points) - min(p[0] for p in points)
        y_range = max(p[1] for p in points) - min(p[1] for p in points)
        max_distance = 0.1 * np.sqrt(x_range**2 + y_range**2)
    
    # Cluster the points using DBSCAN
    clustering = DBSCAN(eps=max_distance, min_samples=1).fit(points_array)
    labels = clustering.labels_
    
    # Count points in each cluster
    unique_labels = set(labels)
    cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
    
    # Find the largest cluster
    largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
    max_size = cluster_sizes[largest_cluster]
    
    # Check if there are multiple clusters with the same maximum size
    largest_clusters = [label for label, size in cluster_sizes.items() if size == max_size]
    if len(largest_clusters) > 1 or len(unique_labels) == 1:
        # No clear majority cluster or only one cluster exists
        # Return centroid of all points
        x_sum = sum(point[0] for point in points)
        y_sum = sum(point[1] for point in points)
        return np.array([round(x_sum / len(points), 2), round(y_sum / len(points), 2)])
    
    # Calculate centroid of the largest cluster
    cluster_points = points_array[labels == largest_cluster]
    x_sum = np.sum(cluster_points[:, 0])
    y_sum = np.sum(cluster_points[:, 1])
    return np.array([round(x_sum / len(cluster_points), 2), round(y_sum / len(cluster_points), 2)])
