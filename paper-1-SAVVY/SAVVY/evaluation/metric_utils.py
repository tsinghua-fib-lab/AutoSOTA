"""
Metric calculation utils code for SAVVY pipeline evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import math
import numpy as np
import json


import numpy as np

def abs_dist_norm_2d(pred, target):
    """
    Calculate normalized absolute distance between prediction and target for 2D locations.
    
    Args:
        pred (list): Predicted 2D coordinates [x, y]
        target (list): Target 2D coordinates [x, y]
        
    Returns:
        float: Normalized Euclidean distance
    """
    try:
        # Ensure we're working with the first 2 dimensions
        pred = pred[:2]
        target = target[:2]
        
        # Calculate Euclidean distance
        euclidean_dist = np.sqrt(sum((float(p) - float(t))**2 for p, t in zip(pred, target)))
        
        # Normalize by the magnitude of the target vector
        # This gives a relative measure of how far off the prediction is
        target_magnitude = np.sqrt(sum(float(t)**2 for t in target))
        
        # Avoid division by zero
        if target_magnitude == 0:
            return euclidean_dist  # Return unnormalized distance if target is at origin
        
        return euclidean_dist / target_magnitude
    except Exception as e:
        print(f"Error calculating 2D distance: {e}")
        return 0.0
    
    
def abs_dist_norm(flag, pred, target):
    """
    flag overlap or distance
    Calculate normalized absolute distance between prediction and target.
    - For 3D coordinates (lists with 3 numbers): Calculate Euclidean distance
    - For time ranges (lists with 2 numbers): Calculate overlap
    - For single numbers: Calculate normalized absolute difference
    """
    try:
        # Handle 3D coordinates (lists with 3 numbers)
        if flag == "distance":
            if isinstance(pred, list):
                pred = pred[0]
            # Return normalized absolute difference
            return abs(float(pred) - float(target)) / float(target)
            # return abs(math.log(float(pred)) - math.log(float(target))) / (math.log(float(target)))
        else:
            # Convert to float to ensure proper calculation
            p_start, p_end = float(pred[0]), float(pred[1])
            t_start, t_end = float(target[0]), float(target[1])
            intersection = max(0, min(p_end, t_end) - max(p_start, t_start))
            pred_length = p_end - p_start
            target_length = t_end - t_start
            union = pred_length + target_length - intersection
            if union > 0:
                iou = (intersection / union)
            else:
                iou = 0.0
            return 1 - iou
    except:
        return 0.0


def mean_relative_accuracy(flag, pred, target, start, end, interval):
    """
    flag overlap or distance
    Calculate mean relative accuracy over confidence intervals.
    Works with 3D coordinates, time ranges, and single numbers.
    """
    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    
    # Calculate normalized distance based on the type of data
    distance = abs_dist_norm(flag, pred, target)
    
    # Calculate accuracy for each confidence interval
    accuracy = distance <= 1 - conf_intervs
    
    return accuracy.mean()


def _calculate_distance(obj1, obj2):
    """Calculate Euclidean distance between two objects"""
    return np.sqrt((obj1[0] - obj2[0])**2 + 
                        (obj1[1] - obj2[1])**2)


def calculate_azimuth(reference_object_loc, reference_facing_object_loc, target_object_loc):
    # azimuth from -180 to 180
    try:
        if isinstance(reference_facing_object_loc, str):
            reference_facing_object_loc = json.loads(reference_facing_object_loc)
        if isinstance(reference_object_loc, str):
            reference_object_loc = json.loads(reference_object_loc)
        if isinstance(target_object_loc, str):
            target_object_loc = json.loads(target_object_loc)
        facing_vector = [
            float(reference_facing_object_loc[0]) - float(reference_object_loc[0]),
            float(reference_facing_object_loc[1]) - float(reference_object_loc[1])
        ]

        # Step 2: Calculate the vector from reference object to target object
        target_vector = [
            float(target_object_loc[0]) - float(reference_object_loc[0]),
            float(target_object_loc[1]) - float(reference_object_loc[1])
        ]
    except:
        return "format err"
    

    dot_product = facing_vector[0] * target_vector[0] + facing_vector[1] * target_vector[1]
    # Calculate magnitudes
    facing_magnitude = math.sqrt(facing_vector[0]**2 + facing_vector[1]**2)
    target_magnitude = math.sqrt(target_vector[0]**2 + target_vector[1]**2)

    # Calculate cosine of the angle
    if facing_magnitude * target_magnitude == 0:
        cos_angle = 0.0
    else:
        cos_angle = dot_product / (facing_magnitude * target_magnitude)

    cos_angle = max(min(cos_angle, 1.0), -1.0)

    # Calculate the angle in radians
    angle_radians = math.acos(cos_angle)

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    cross_product = facing_vector[0] * target_vector[1] - facing_vector[1] * target_vector[0]
    if cross_product > 0:
        angle_degrees = -angle_degrees
    return angle_degrees


def calculate_azimuth_facevec(reference_object_loc, facing_vector, target_object_loc):
    # azimuth from -180 to 180
    target_vector = [
        float(target_object_loc[0]) - float(reference_object_loc[0]),
        float(target_object_loc[1]) - float(reference_object_loc[1])
    ]

    dot_product = facing_vector[0] * target_vector[0] + facing_vector[1] * target_vector[1]
    # Calculate magnitudes
    facing_magnitude = math.sqrt(facing_vector[0]**2 + facing_vector[1]**2)
    target_magnitude = math.sqrt(target_vector[0]**2 + target_vector[1]**2)

    # Calculate cosine of the angle
    if facing_magnitude * target_magnitude == 0:
        cos_angle = 0.0
    else:
        cos_angle = dot_product / (facing_magnitude * target_magnitude)

    cos_angle = max(min(cos_angle, 1.0), -1.0)

    # Calculate the angle in radians
    angle_radians = math.acos(cos_angle)

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    cross_product = facing_vector[0] * target_vector[1] - facing_vector[1] * target_vector[0]
    if cross_product > 0:
        angle_degrees = -angle_degrees
    return angle_degrees


def calculate_absolute_traj_distance(traj1, traj2, mean_flag=True):
    """
    Calculate absolute distance between two trajectories.
    
    Args:
        traj1: List of positions [[x1, y1], [x2, y2], ...]
        traj2: List of positions [[x1, y1], [x2, y2], ...]
    
    Returns:
        Average Euclidean distance between trajectories
    """
    # Convert to numpy arrays for easier manipulation
    traj1_arr = np.array(traj1)
    traj2_arr = np.array(traj2)
    
    if mean_flag or len(traj1_arr) != len(traj2_arr):
        # Calculate mean positions for both trajectories
        traj1_mean = np.mean(traj1_arr, axis=0)
        traj2_mean = np.mean(traj2_arr, axis=0)
        
        # Calculate Euclidean distance between mean positions
        dist = np.sqrt(np.sum((traj1_mean - traj2_mean) ** 2))
    else:
        distances = []
        for i in range(len(traj1_arr)):
            # Calculate Euclidean distance for this pair of points
            point_dist = np.sqrt(np.sum((traj1_arr[i] - traj2_arr[i]) ** 2))
            distances.append(point_dist)
            
        # Average all point distances
        dist = np.mean(distances)
    return dist

