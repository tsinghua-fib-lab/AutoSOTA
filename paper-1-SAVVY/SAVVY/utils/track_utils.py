"""
Tracking utils code for SAVVY pipeline dynamic tracks merging - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import numpy as np

def get_nearest_times(time_keys, target_time, max_count=4):
    """
    Get the nearest time keys to the target time.
    
    Args:
        time_keys: Iterable of time keys
        target_time: The target time to find nearest neighbors for
        max_count: Maximum number of nearest times to return
        
    Returns:
        List of nearest time keys, sorted by proximity
    """
    time_keys = list(time_keys)  # Convert to list if it's not already
    if not time_keys:
        return []
    
    # Calculate distances and sort by proximity
    time_distances = [(abs(t - target_time), t) for t in time_keys]
    time_distances.sort()  # Sort by distance
    
    # Return the times of the nearest points (up to max_count)
    return [t for _, t in time_distances[:max_count]]


def filter_outliers(current_time, locations, trajectory, object_id, max_distance=None):
    """
    Filter out locations that deviate too much from the existing trajectory.
    
    Args:
        current_time: Current timestamp for prediction
        locations: List of possible location points
        trajectory: The existing trajectory dictionary
        object_id: The object identifier
        max_distance: Optional maximum allowed distance threshold
        
    Returns:
        Filtered list of locations
    """
    # If no trajectory exists for this object, return all locations
    if not trajectory.get(object_id):
        return locations
        
    # Get recent positions to establish trajectory trend, using current_time
    recent_positions = get_recent_positions(trajectory, object_id, current_time)
        
    # Filter based on distance from predicted position
    if recent_positions:
        # Use Kalman filter for prediction at the current time
        predicted_pos = extrapolate_position(recent_positions, current_time, use_kalman=True)
        
        # Use provided max_distance or calculate based on recent positions
        if max_distance is not None:
            max_distance = max(max_distance, calculate_max_allowed_distance(recent_positions))
        else:
            max_distance = calculate_max_allowed_distance(recent_positions)
            
        # # Remove debug breakpoint
        
        return [loc for loc in locations if calculate_distance(loc, predicted_pos) <= max_distance]
        
    return locations

    
    
        
from pykalman import KalmanFilter

def predict_trajectory_point(trajectory, target_time):
    """
    Predict a trajectory point at a given time using Kalman filtering.
    
    Args:
        trajectory (dict): Dictionary with time points as keys and (x, y) tuples as values
        target_time (int): Time point to predict
        
    Returns:
        tuple: Predicted (x, y) coordinates at the target time
    """
    # Extract time points and coordinates from the trajectory
    times = np.array(list(trajectory.keys()))
    positions = np.array(list(trajectory.values()))
    
    # Check if target_time is already in the trajectory
    if target_time in trajectory:
        return trajectory[target_time]
    
    # Initialize Kalman Filter - using a simple constant velocity model
    # State: [x, y, x_velocity, y_velocity]
    transition_matrix = np.array([
        [1, 0, 1, 0],  # x = x + x_velocity
        [0, 1, 0, 1],  # y = y + y_velocity
        [0, 0, 1, 0],  # x_velocity = x_velocity
        [0, 0, 0, 1]   # y_velocity = y_velocity
    ])
    
    observation_matrix = np.array([
        [1, 0, 0, 0],  # We observe x
        [0, 1, 0, 0]   # We observe y
    ])
    
    # Initialize the filter with reasonable values
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=np.array([positions[0][0], positions[0][1], 0, 0]),
        initial_state_covariance=np.eye(4) * 0.1,
        transition_covariance=np.eye(4) * 0.01,
        observation_covariance=np.eye(2) * 0.1
    )
    
    # Train the filter with existing trajectory
    means, covariances = kf.filter(positions)
    
    # If target time is before earliest trajectory point
    if target_time < min(times):
        # Extrapolate backward
        time_diff = target_time - min(times)
        state = means[0]
        # Adjust the state for backward prediction
        predicted_state = np.dot(np.linalg.matrix_power(transition_matrix, abs(time_diff)), state)
        return (predicted_state[0], predicted_state[1])
    
    # If target time is beyond the last trajectory point
    if target_time > max(times):
        # Extrapolate forward
        time_diff = target_time - max(times)
        state = means[-1]
        # Forward prediction
        predicted_state = np.dot(np.linalg.matrix_power(transition_matrix, time_diff), state)
        return (predicted_state[0], predicted_state[1])
    
    # If target time is within trajectory range but not a key
    # Find the two closest time points
    prev_time = max([t for t in times if t < target_time])
    next_time = min([t for t in times if t > target_time])
    
    prev_index = np.where(times == prev_time)[0][0]
    next_index = np.where(times == next_time)[0][0]
    
    # Use the state from the previous time point
    state = means[prev_index]
    
    # Calculate the number of time steps to predict
    time_diff = target_time - prev_time
    
    # Predict the state at target_time
    predicted_state = np.dot(np.linalg.matrix_power(transition_matrix, time_diff), state)
    
    return (predicted_state[0], predicted_state[1])


def get_recent_positions(trajectory, object_id, current_time, window_size=5, time_window=30):
    """
    Get the most recent positions for an object from its trajectory, relative to current_time.
    
    Args:
        trajectory: The trajectory dictionary
        object_id: The object identifier
        current_time: Current timestamp for reference
        window_size: Maximum number of recent positions to consider
        time_window: Maximum time difference (in time units) to consider positions
        
    Returns:
        Dictionary of recent timestamps and positions
    """
    if not trajectory.get(object_id):
        return {}
    # Get all timestamps for this object
    all_timestamps = sorted(trajectory[object_id].keys())
    
    # Filter timestamps that are not too old compared to current_time
    valid_timestamps = [t for t in all_timestamps if abs(current_time - t) <= time_window]
    recent_timestamps = valid_timestamps
    # # If we have no valid timestamps within the time window, use the most recent ones
    # if not valid_timestamps and all_timestamps:
    #     valid_timestamps = all_timestamps[-window_size:]
    
    # # Get the most recent timestamps (up to window_size)
    # recent_timestamps = valid_timestamps[-window_size:] if len(valid_timestamps) > window_size else valid_timestamps
    
    # Return dictionary with recent timestamps and positions
    return {t: trajectory[object_id][t] for t in recent_timestamps}


def extrapolate_position(recent_positions, current_time, use_kalman=True):
    """
    Predict the position at current_time based on recent trajectory.
    
    Args:
        recent_positions: Dictionary of recent timestamps and positions
        current_time: The timestamp for which to predict the position
        use_kalman: Whether to use Kalman filtering
        
    Returns:
        Predicted position (x, y) at current_time
    """
    timestamps = sorted(recent_positions.keys())
    
    # If we have only one position, we can't extrapolate
    if len(timestamps) < 2:
        return recent_positions[timestamps[0]]
    
    # Get the last known timestamp
    last_timestamp = timestamps[-1]
    
    # If current_time is earlier than the last known position, interpolate instead
    if current_time <= last_timestamp:
        # Find the two positions to interpolate between
        for i, t in enumerate(timestamps):
            if t >= current_time:
                if i == 0:
                    # Current time is before first position, use first position
                    return recent_positions[timestamps[0]]
                else:
                    # Interpolate between positions
                    t1, t2 = timestamps[i-1], timestamps[i]
                    pos1, pos2 = recent_positions[t1], recent_positions[t2]
                    
                    # Calculate interpolation factor
                    factor = (current_time - t1) / (t2 - t1) if t2 != t1 else 0
                    
                    # Linear interpolation
                    interp_x = pos1[0] + factor * (pos2[0] - pos1[0])
                    interp_y = pos1[1] + factor * (pos2[1] - pos1[1])
                    
                    return (interp_x, interp_y)
    
    # For extrapolation (current_time > last_timestamp)
    if use_kalman:
        # Create a temporary trajectory dictionary for Kalman filtering
        temp_trajectory = {'temp_object': {}}
        for t in timestamps:
            temp_trajectory['temp_object'][t] = recent_positions[t]
        
        # Apply Kalman filter
        optimized_trajectory = optimize_trajectory_smoothness(temp_trajectory, use_kalman=True)
        
        # Get the filtered positions
        filtered_positions = optimized_trajectory['temp_object']
        
        # Get the last two filtered positions
        last_timestamp = timestamps[-1]
        second_last_timestamp = timestamps[-2] if len(timestamps) >= 2 else None
        
        if second_last_timestamp is not None:
            pos1 = filtered_positions[second_last_timestamp]
            pos2 = filtered_positions[last_timestamp]
            
            # Calculate velocity vector using filtered positions
            time_diff = last_timestamp - second_last_timestamp
            if time_diff > 0:
                velocity_x = (pos2[0] - pos1[0]) / time_diff
                velocity_y = (pos2[1] - pos1[1]) / time_diff
                
                # Extrapolate to current_time
                time_to_predict = current_time - last_timestamp
                predicted_x = pos2[0] + velocity_x * time_to_predict
                predicted_y = pos2[1] + velocity_y * time_to_predict
                
                return (predicted_x, predicted_y)
        
        # Fallback to last known position if can't calculate velocity
        return filtered_positions[last_timestamp]
    else:
        # Simple linear extrapolation based on last two positions
        t1, t2 = timestamps[-2], timestamps[-1]
        pos1, pos2 = recent_positions[t1], recent_positions[t2]
        
        # Calculate velocity vector
        time_diff = t2 - t1
        if time_diff > 0:
            velocity_x = (pos2[0] - pos1[0]) / time_diff
            velocity_y = (pos2[1] - pos1[1]) / time_diff
            
            # Extrapolate to current_time
            time_to_predict = current_time - t2
            predicted_x = pos2[0] + velocity_x * time_to_predict
            predicted_y = pos2[1] + velocity_y * time_to_predict
            
            return (predicted_x, predicted_y)
        else:
            # Fallback to last known position if can't calculate velocity
            return pos2


def optimize_trajectory_smoothness(trajectory, use_kalman=True, filter=True, outlier_threshold=1.0, interpolate_gaps=False):
    """
    Apply global optimization to make the trajectory temporally smooth using Kalman filtering,
    remove outliers, and interpolate missing time points.
    
    Args:
        trajectory: The trajectory dictionary to optimize
        use_kalman: Whether to use Kalman filtering (True) or a simple moving average (False)
        outlier_threshold: Threshold for identifying outliers (standard deviations from prediction)
        interpolate_gaps: Whether to interpolate missing time points
        
    Returns:
        Optimized trajectory dictionary with outliers removed and gaps filled
    """
    if not trajectory:
        return trajectory
    
    # First pass: detect and remove outliers
    if filter:
        trajectory = remove_trajectory_outliers(trajectory, outlier_threshold)
    
    # Second pass: interpolate missing time points if requested
    if interpolate_gaps:
        trajectory = interpolate_missing_timepoints(trajectory)
    
        # Third pass: smooth the cleaned and interpolated trajectory
        if use_kalman:
            return apply_kalman_filter(trajectory)
        else:
            # Fallback to simple moving average if not using Kalman
            return apply_moving_average(trajectory)
    else:
        return trajectory
    

def interpolate_distances(distances, all_timestamps):
    """
    Interpolate missing distance values using Kalman filtering.
    
    Args:
        distances: Dictionary mapping timestamps to distance values
        all_timestamps: List of all timestamps that need distance values
        
    Returns:
        Dictionary with interpolated distance values for all timestamps
    """
    import numpy as np
    
    # If no distances provided, return empty dictionary
    if not distances:
        return {}
    
    # Convert distances to sorted list of (timestamp, distance) tuples
    distance_items = sorted(distances.items())
    
    # If only one distance value, use it for all timestamps
    if len(distance_items) < 2:
        if distance_items:
            return {t: distance_items[0][1] for t in all_timestamps}
        else:
            return {}
    
    # Extract timestamps and distance values
    known_timestamps = [item[0] for item in distance_items]
    known_distances = [item[1] for item in distance_items]
    
    # Initialize Kalman filter for 1D tracking (distance)
    # State: [distance, velocity]
    state_dim = 2
    measurement_dim = 1
    
    # Initial state
    x = np.zeros((state_dim, 1))
    x[0] = known_distances[0]  # Initial distance
    
    # Calculate initial velocity if possible
    if len(known_timestamps) >= 2:
        dt = known_timestamps[1] - known_timestamps[0]
        if dt > 0:
            x[1] = (known_distances[1] - known_distances[0]) / dt  # Initial velocity
    
    # Initial covariance
    P = np.eye(state_dim) * 1.0
    
    # Process noise (how much we expect the state to change between steps)
    Q = np.array([
        [0.01, 0],    # Low noise for distance
        [0, 0.1]      # Higher noise for velocity
    ])
    
    # Measurement noise (how much we trust the measurements)
    R = np.array([[0.1]])
    
    # Measurement matrix (we can only observe distance, not velocity)
    H = np.array([[1, 0]])
    
    # Store filtered distances
    filtered_distances = {known_timestamps[0]: known_distances[0]}
    
    # First pass: forward Kalman filter through known points
    prev_timestamp = known_timestamps[0]
    
    for i in range(1, len(known_timestamps)):
        current_timestamp = known_timestamps[i]
        current_distance = known_distances[i]
        
        # Time step
        dt = current_timestamp - prev_timestamp
        
        # State transition matrix (depends on dt)
        F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Prediction step
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update step with measurement
        z = np.array([[current_distance]])
        y = z - H @ x  # Residual
        S = H @ P @ H.T + R  # Residual covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        x = x + K @ y  # Update state estimate
        P = (np.eye(state_dim) - K @ H) @ P  # Update covariance
        
        # Store filtered distance
        filtered_distances[current_timestamp] = float(x[0])
        
        prev_timestamp = current_timestamp
    
    # Second pass: interpolate and extrapolate for all required timestamps
    interpolated_distances = {}
    
    for t in sorted(all_timestamps):
        # Check if we already have a filtered distance for this timestamp
        if t in filtered_distances:
            interpolated_distances[t] = filtered_distances[t]
            continue
        
        # Find surrounding known timestamps
        earlier_timestamps = [kt for kt in known_timestamps if kt < t]
        later_timestamps = [kt for kt in known_timestamps if kt > t]
        
        if earlier_timestamps and later_timestamps:
            # Interpolation case
            prev_t = max(earlier_timestamps)
            next_t = min(later_timestamps)
            
            # Get filtered distances at surrounding points
            prev_d = filtered_distances[prev_t]
            next_d = filtered_distances[next_t]
            
            # Linear interpolation
            time_range = next_t - prev_t
            if time_range > 0:
                factor = (t - prev_t) / time_range
                interpolated_distances[t] = prev_d + factor * (next_d - prev_d)
            else:
                interpolated_distances[t] = prev_d
        elif earlier_timestamps:
            # Extrapolation after the last known point
            last_t = max(earlier_timestamps)
            second_last_t = max([kt for kt in earlier_timestamps if kt < last_t]) if len(earlier_timestamps) > 1 else None
            
            if second_last_t is not None:
                # Estimate velocity from last two points
                last_d = filtered_distances[last_t]
                second_last_d = filtered_distances[second_last_t]
                time_diff = last_t - second_last_t
                
                if time_diff > 0:
                    velocity = (last_d - second_last_d) / time_diff
                    time_to_predict = t - last_t
                    interpolated_distances[t] = last_d + velocity * time_to_predict
                else:
                    interpolated_distances[t] = last_d
            else:
                # Only one earlier point, use its value
                interpolated_distances[t] = filtered_distances[last_t]
        elif later_timestamps:
            # Extrapolation before the first known point
            first_t = min(later_timestamps)
            second_t = min([kt for kt in later_timestamps if kt > first_t]) if len(later_timestamps) > 1 else None
            
            if second_t is not None:
                # Estimate velocity from first two points
                first_d = filtered_distances[first_t]
                second_d = filtered_distances[second_t]
                time_diff = second_t - first_t
                
                if time_diff > 0:
                    velocity = (second_d - first_d) / time_diff
                    time_to_predict = first_t - t
                    interpolated_distances[t] = first_d - velocity * time_to_predict
                else:
                    interpolated_distances[t] = first_d
            else:
                # Only one later point, use its value
                interpolated_distances[t] = filtered_distances[first_t]
    
    return interpolated_distances


def remove_trajectory_outliers(trajectory, threshold=3.0):
    """
    Remove outlier points from trajectory that don't fit the expected motion pattern.
    
    Args:
        trajectory: The trajectory dictionary
        threshold: Threshold for identifying outliers (standard deviations from prediction)
        
    Returns:
        Cleaned trajectory dictionary with outliers removed
    """
    import numpy as np
    
    if len(trajectory) < 3:
        return trajectory
    # Create a deep copy of the trajectory to avoid modifying the original
    cleaned_trajectory = {}
    timestamps = sorted(trajectory.keys())
    cleaned_positions = {timestamps[0]: trajectory[timestamps[0]]}

    for i in range(1, len(timestamps) - 1):
        current_time = timestamps[i]
        current_pos = trajectory[current_time]
        
        # Get previous and next timestamps/positions
        prev_time = timestamps[i-1]
        next_time = timestamps[i+1]
        prev_pos = trajectory[prev_time]
        next_pos = trajectory[next_time]
        
        # Time differences
        dt_prev = current_time - prev_time
        dt_next = next_time - current_time
        
        # Linear interpolation between previous and next positions
        total_dt = dt_prev + dt_next
        weight_prev = dt_next / total_dt
        weight_next = dt_prev / total_dt
        
        # Predicted position based on linear interpolation
        pred_x = prev_pos[0] * weight_prev + next_pos[0] * weight_next
        pred_y = prev_pos[1] * weight_prev + next_pos[1] * weight_next
        
        # Calculate deviation from prediction
        dx = current_pos[0] - pred_x
        dy = current_pos[1] - pred_y
        deviation = np.sqrt(dx**2 + dy**2)
        
        # Calculate velocity-based expected deviation
        # Higher velocity → higher expected deviation
        velocity_prev = calculate_distance(prev_pos, current_pos) / dt_prev
        velocity_next = calculate_distance(current_pos, next_pos) / dt_next
        avg_velocity = (velocity_prev + velocity_next) / 2
        
        # Scale threshold by velocity
        velocity_factor = max(1.0, min(2.0, avg_velocity))
        adjusted_threshold = threshold * velocity_factor
        
        # Also consider acceleration (change in velocity)
        acceleration = abs(velocity_next - velocity_prev) / total_dt
        
        # Higher acceleration → higher expected deviation
        accel_factor = max(1.0, min(2.0, acceleration * 10 + 1))
        adjusted_threshold *= accel_factor
        
        # Check if the position is an outlier
        # If not, add it to the cleaned positions
        if deviation <= adjusted_threshold:
            cleaned_positions[current_time] = current_pos
    
    # Add the last position
    if len(timestamps) > 0:
        cleaned_positions[timestamps[-1]] = trajectory[timestamps[-1]]
        
    cleaned_trajectory = cleaned_positions
    
    return cleaned_trajectory


def interpolate_missing_timepoints(trajectory, min_time_step=1):
    """
    Interpolate missing time points in a trajectory to ensure consistent sampling.
    
    Args:
        trajectory: The trajectory dictionary
        min_time_step: Minimum time step for interpolation (default: 1 time unit)
        
    Returns:
        Trajectory dictionary with interpolated time points
    """
    interpolated_trajectory = {}
    if len(trajectory) < 2:
        return trajectory
    timestamps = sorted(trajectory.keys())
        
    # Find the appropriate time step
    time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    if time_diffs:
        # Find the most common time difference (mode) that's >= min_time_step
        valid_diffs = [diff for diff in time_diffs if diff >= min_time_step]
        if valid_diffs:
            time_step = min(valid_diffs)
        else:
            time_step = min_time_step
    else:
        time_step = min_time_step
    time_step = min_time_step
    
    # Create interpolated positions dictionary
    interpolated_positions = {}
    
    # Add first position
    interpolated_positions[timestamps[0]] = trajectory[timestamps[0]]
    
    # Process each pair of consecutive positions
    for i in range(len(timestamps) - 1):
        current_time = timestamps[i]
        next_time = timestamps[i+1]
        current_pos = trajectory[current_time]
        next_pos = trajectory[next_time]
        
        # Time difference between consecutive points
        time_diff = next_time - current_time
        
        # If gap is larger than time_step, interpolate
        if time_diff > time_step:
            # Number of steps to interpolate
            num_steps = int(time_diff / time_step)
            
            # Interpolate positions at regular intervals
            for step in range(1, num_steps):
                interp_time = current_time + step * time_step
                
                # Skip if we already reached or passed the next time
                if interp_time >= next_time:
                    continue
                
                # Linear interpolation factor
                factor = (interp_time - current_time) / time_diff
                
                # Interpolated position
                interp_x = current_pos[0] + factor * (next_pos[0] - current_pos[0])
                interp_y = current_pos[1] + factor * (next_pos[1] - current_pos[1])
                
                # Add interpolated position
                interpolated_positions[interp_time] = (interp_x, interp_y)
        
        # Add the next position
        interpolated_positions[next_time] = next_pos
        
    interpolated_trajectory = interpolated_positions

    return interpolated_trajectory


def interpolate_missing_timepoints1d(trajectory, min_time_step=1):
    """
    Interpolate missing time points in a trajectory to ensure consistent sampling.
    
    Args:
        trajectory: The trajectory dictionary
        min_time_step: Minimum time step for interpolation (default: 1 time unit)
        
    Returns:
        Trajectory dictionary with interpolated time points
    """
    interpolated_trajectory = {}
    if len(trajectory) < 2:
        return trajectory
    timestamps = sorted(trajectory.keys())
    # Find the appropriate time step
    time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    if time_diffs:
        # Find the most common time difference (mode) that's >= min_time_step
        valid_diffs = [diff for diff in time_diffs if diff >= min_time_step]
        if valid_diffs:
            time_step = min(valid_diffs)
        else:
            time_step = min_time_step
    else:
        time_step = min_time_step
    time_step = min_time_step
    
    # Create interpolated positions dictionary
    interpolated_positions = {}
    
    # Add first position
    interpolated_positions[timestamps[0]] = trajectory[timestamps[0]]
    
    # Process each pair of consecutive positions
    for i in range(len(timestamps) - 1):
        current_time = timestamps[i]
        next_time = timestamps[i+1]
        current_pos = trajectory[current_time]
        next_pos = trajectory[next_time]
        # if next_pos[0] is None or current_pos[0] is None:
        #     continue
        # Time difference between consecutive points
        time_diff = next_time - current_time
        
        # If gap is larger than time_step, interpolate
        if time_diff > time_step:
            # Number of steps to interpolate
            num_steps = int(time_diff / time_step)
            
            # Interpolate positions at regular intervals
            for step in range(1, num_steps):
                interp_time = current_time + step * time_step
                
                # Skip if we already reached or passed the next time
                if interp_time >= next_time:
                    continue
                
                # Linear interpolation factor
                factor = (interp_time - current_time) / time_diff
                
                # Interpolated position
                interp_x = current_pos + factor * (next_pos - current_pos)
                
                # Add interpolated position
                interpolated_positions[interp_time] = interp_x
        
        # Add the next position
        interpolated_positions[next_time] = next_pos
        
    interpolated_trajectory = interpolated_positions

    return interpolated_trajectory



def apply_kalman_filter(trajectory):
    """
    Apply Kalman filtering to smooth a trajectory.
    
    Args:
        trajectory: Dictionary with timestamps as keys and (x, y) coordinates as values
        
    Returns:
        Smoothed trajectory dictionary
    """
    # Sort times for sequential processing
    times = sorted(trajectory.keys())
    if len(times) < 2:
        return trajectory  # Not enough points to filter
    
    # Initialize Kalman filter parameters
    # State: [x, y, vx, vy] where (x,y) is position and (vx,vy) is velocity
    dt = 1.0  # Time step
    
    # Process noise (how much we expect the state to change between steps)
    process_noise_pos = 0.01  # Position noise
    process_noise_vel = 0.1   # Velocity noise
    
    # Measurement noise (how much we trust the measurements)
    measurement_noise = 0.1
    
    # Initial state and covariance
    x_k = np.array([trajectory[times[0]][0], trajectory[times[0]][1], 0, 0])  # [x, y, vx, vy]
    P_k = np.eye(4)  # Initial covariance
    
    # State transition matrix (physics model)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Process noise covariance
    Q = np.diag([process_noise_pos, process_noise_pos, process_noise_vel, process_noise_vel])
    
    # Measurement matrix (we can only observe position, not velocity)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Measurement noise covariance
    R = np.eye(2) * measurement_noise
    
    # Storage for filtered trajectory
    filtered_trajectory = {times[0]: trajectory[times[0]]}  # Start with the first point
    
    # Apply Kalman filter through all points
    for i in range(1, len(times)):
        # Prediction step
        x_k_minus = F @ x_k
        P_k_minus = F @ P_k @ F.T + Q
        
        # Get measurement
        z_k = np.array([trajectory[times[i]][0], trajectory[times[i]][1]])
        
        # Update step
        y_k = z_k - H @ x_k_minus  # Measurement residual
        S_k = H @ P_k_minus @ H.T + R  # Residual covariance
        K_k = P_k_minus @ H.T @ np.linalg.inv(S_k)  # Kalman gain
        
        # Update state and covariance
        x_k = x_k_minus + K_k @ y_k
        P_k = (np.eye(4) - K_k @ H) @ P_k_minus
        
        # Store filtered position
        filtered_trajectory[times[i]] = (round(x_k[0], 2), round(x_k[1], 2))
    
    return filtered_trajectory



def apply_moving_average(trajectory, window_size=3):
    """
    Apply a simple moving average to smooth a trajectory.
    
    Args:
        trajectory: Dictionary with timestamps as keys and (x, y) coordinates as values
        window_size: Size of the moving average window
        
    Returns:
        Smoothed trajectory dictionary
    """
    times = sorted(trajectory.keys())
    if len(times) < window_size:
        return trajectory  # Not enough points
    
    smoothed_trajectory = {}
    
    # For each point
    for i, time in enumerate(times):
        # Determine window boundaries
        window_start = max(0, i - window_size // 2)
        window_end = min(len(times) - 1, i + window_size // 2)
        window_times = times[window_start:window_end + 1]
        
        # Calculate average position within window
        x_sum = sum(trajectory[t][0] for t in window_times)
        y_sum = sum(trajectory[t][1] for t in window_times)
        avg_x = x_sum / len(window_times)
        avg_y = y_sum / len(window_times)
        
        smoothed_trajectory[time] = (round(avg_x, 2), round(avg_y, 2))
    
    return smoothed_trajectory



def calculate_max_allowed_distance(recent_positions, factor=1.5):
    """
    Calculate maximum allowed distance based on recent movement patterns.
    
    Args:
        recent_positions: Dictionary of recent timestamps and positions
        factor: Multiplier to determine maximum allowed deviation
        
    Returns:
        Maximum allowed distance
    """
    timestamps = sorted(recent_positions.keys())
    
    # If we have fewer than 2 positions, use a default value
    if len(timestamps) < 2:
        return 1.0  # Default max distance
    
    # Calculate average distance between consecutive positions
    total_distance = 0
    count = 0
    
    for i in range(1, len(timestamps)):
        t_prev, t_curr = timestamps[i-1], timestamps[i]
        pos_prev, pos_curr = recent_positions[t_prev], recent_positions[t_curr]
        
        distance = calculate_distance(pos_prev, pos_curr)
        total_distance += distance
        count += 1
    
    avg_distance = total_distance / count if count > 0 else 1.0
    
    # Return a multiple of the average distance
    return avg_distance * factor


def calculate_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Euclidean distance
    """
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5