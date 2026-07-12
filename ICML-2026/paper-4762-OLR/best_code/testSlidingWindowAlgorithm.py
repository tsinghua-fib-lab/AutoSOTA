# Import required libraries for sliding window multi-armed bandit algorithms
import os
import pickle
import numpy as np
import random
import copy
import tqdm

def bucket_exploration(arm_stream, input_memory_size, delta):
    '''
    Explores arms using a bucket-based approach to find epsilon-best arms.
    This algorithm organizes arms into buckets based on their empirical rewards
    and maintains only the most promising arms within the sliding window.
    
    :param arm_stream: The stream of arms to explore
    :param input_memory_size: The size of the memory (number of buckets - 1)
    :param delta: The failure probability threshold
    :return: Array of best rewards at each time step
    '''
    # Add 1 to memory size for bucket indexing
    memory_size = input_memory_size+1

    # Extract basic parameters from the arm stream
    num_arms = arm_stream.n
    window_size = arm_stream.window_size

    # Multi-arm bucketing: store up to K=3 arms per bucket
    # Each bucket is a list of (arm, empirical_reward) tuples, sorted by reward descending
    K = 5

    # Initialize buckets to store arms, each bucket initially empty list
    buckets = [[] for _ in range(memory_size)]

    # Track the best reward found at each time step
    best_rewards = np.zeros(num_arms)

    # Set exploration accuracy parameter (eps = 3/memory_size)
    eps = 3/memory_size

    # Define bucket width (segment size for reward discretization)
    segment = 1/memory_size

    # Calculate number of pulls needed for reliable empirical estimates
    # Based on Hoeffding's inequality for concentration bounds
    num_pulls = int(np.ceil(1/(2*segment*segment) * np.log(num_arms/delta)))

    # Process each arm in the stream sequentially
    for i in range(num_arms):
        # Read the next arm from the stream
        current_arm = arm_stream.read_next_arm()
        # Get current time step
        time_now = arm_stream.t
        # compute the empirical reward of the arm
        empirical_reward = current_arm.batch_pull(num_pulls)
        # find the bucket index
        current_bucket_index = int(np.floor(empirical_reward/segment))
        current_bucket_index = min(current_bucket_index, memory_size-1)
        # put the arm into the bucket, keeping top-K by empirical reward
        buckets[current_bucket_index].append((current_arm, empirical_reward))
        buckets[current_bucket_index].sort(key=lambda x: x[1], reverse=True)
        buckets[current_bucket_index] = buckets[current_bucket_index][:K]

        # expire arms that have fallen outside the sliding window
        for j in range(memory_size):
            buckets[j] = [(arm, er) for arm, er in buckets[j]
                          if time_now - window_size < arm.t]

        # find the best arm: scan high-to-low, collect candidates from
        # top 3 non-empty buckets, pick highest empirical reward
        for j in range(memory_size-1, -1, -1):
            if buckets[j]:
                candidates = list(buckets[j])
                # also collect from the next 2 buckets below
                for offset in range(1, 3):
                    if j - offset >= 0 and buckets[j - offset]:
                        candidates.extend(buckets[j - offset])
                best_candidate = max(candidates, key=lambda x: x[1])
                best_rewards[time_now] = best_candidate[0].p
                break

    return best_rewards

def baseline_exploration_not_clear_expiration(arm_stream, input_memory_size, delta):
    '''
    Explores arms using a baseline approach to find epsilon-best arms.
    This algorithm maintains a fixed set of arms and evaluates their performance
    over time without dynamic bucket allocation.
    
    :param arm_stream: The stream of arms to explore
    :param input_memory_size: The size of the memory (number of buckets - 1)
    :param delta: The failure probability threshold
    :return: Array of best rewards at each time step
    '''

    # Add 1 to memory size for bucket indexing
    memory_size = input_memory_size+1

    # Extract basic parameters from the arm stream
    num_arms = arm_stream.n
    window_size = arm_stream.window_size


    # Initialize buckets to store arms, each bucket initially empty (None)
    buckets = np.array([None]*memory_size)

    # Track the best reward found at each time step
    best_rewards = np.zeros(num_arms)

    # Set exploration accuracy parameter (eps = 3/memory_size)
    eps = 3/memory_size
    segment = 1/memory_size

    # Calculate number of pulls needed for reliable empirical estimates
    # Based on Hoeffding's inequality for concentration bounds
    num_pulls = int(np.ceil(1/(2*segment*segment) * np.log(num_arms/delta)))

    # Generate an arm with p=0 for penalty for the expired arms
    penalty_arm = copy.deepcopy(arm_stream.arm_set[0])
    penalty_arm.p = 0

    # Initialize variables to track the bucket with the lowest empirical reward
    min_empirical_reward = -1
    min_bucket_index = 0

    # Initialize the empirical rewards for each bucket
    empirical_rewards = np.array([-1]*memory_size)

    # Process each arm in the stream sequentially
    for i in range(num_arms):
        # Read the next arm from the stream
        current_arm = arm_stream.read_next_arm()
        # Get current time step
        time_now = arm_stream.t
        # compute the empirical reward of the arm
        empirical_reward = current_arm.batch_pull(num_pulls)
        # Replace this arm to the arm stored in the bucket with the lowest empirical reward if it is better
        if empirical_reward > min_empirical_reward:
            buckets[min_bucket_index] = current_arm
            empirical_rewards[min_bucket_index] = empirical_reward

            # Update the minimum empirical reward and corresponding bucket index
            # min_empirical_reward is the lowest empirical reward among the empirical_rewards, and min_bucket_index is its index
            min_empirical_reward = empirical_rewards[0]
            min_bucket_index = 0
            for j in range(1, memory_size):
                if empirical_rewards[j] < min_empirical_reward:
                    min_empirical_reward = empirical_rewards[j]
                    min_bucket_index = j

        # Find the arm with the highest empirical reward among the buckets
        max_empirical_reward = -1
        max_bucket_index = 0
        for j in range(memory_size):
            if empirical_rewards[j] > max_empirical_reward:
                max_empirical_reward = empirical_rewards[j]
                max_bucket_index = j

        # The best arm we find is buckets[max_bucket_index] if it is not expired, otherwise the penalty arm
        best_arm = buckets[max_bucket_index]
        if best_arm is not None:
            if time_now - window_size > best_arm.t:
                best_arm = penalty_arm
        else:
            best_arm = penalty_arm
        
        best_rewards[time_now] = best_arm.p
    return best_rewards

def baseline_exploration_clear_expiration(arm_stream, input_memory_size, delta):
    '''
    Explores arms using a baseline approach to find epsilon-best arms.
    This algorithm maintains a fixed set of arms and evaluates their performance
    over time without dynamic bucket allocation.
    
    :param arm_stream: The stream of arms to explore
    :param input_memory_size: The size of the memory (number of buckets - 1)
    :param delta: The failure probability threshold
    :return: Array of best rewards at each time step
    '''

    # Add 1 to memory size for bucket indexing
    memory_size = input_memory_size+1

    # Extract basic parameters from the arm stream
    num_arms = arm_stream.n
    window_size = arm_stream.window_size


    # Initialize buckets to store arms, each bucket initially empty (None)
    buckets = np.array([None]*memory_size)

    # Track the best reward found at each time step
    best_rewards = np.zeros(num_arms)

    # Set exploration accuracy parameter (eps = 3/memory_size)
    eps = 3/memory_size
    segment = 1/memory_size

    # Calculate number of pulls needed for reliable empirical estimates
    # Based on Hoeffding's inequality for concentration bounds
    num_pulls = int(np.ceil(1/(2*segment*segment) * np.log(num_arms/delta)))

    # Generate an arm with p=0 for penalty for the expired arms
    penalty_arm = copy.deepcopy(arm_stream.arm_set[0])
    penalty_arm.p = 0

    # Initialize variables to track the bucket with the lowest empirical reward
    min_empirical_reward = -1
    min_bucket_index = 0

    # Initialize the empirical rewards for each bucket
    empirical_rewards = np.array([-1]*memory_size)

    # Process each arm in the stream sequentially
    for i in range(num_arms):
        # Read the next arm from the stream
        current_arm = arm_stream.read_next_arm()
        # Get current time step
        time_now = arm_stream.t
        # compute the empirical reward of the arm
        empirical_reward = current_arm.batch_pull(num_pulls)
        # Replace this arm to the arm stored in the bucket with the lowest empirical reward if it is better
        if empirical_reward > min_empirical_reward:
            buckets[min_bucket_index] = current_arm
            empirical_rewards[min_bucket_index] = empirical_reward

            # Update the minimum empirical reward and corresponding bucket index
            # min_empirical_reward is the lowest empirical reward among the empirical_rewards, and min_bucket_index is its index
            min_empirical_reward = empirical_rewards[0]
            min_bucket_index = 0
            for j in range(1, memory_size):
                if empirical_rewards[j] < min_empirical_reward:
                    min_empirical_reward = empirical_rewards[j]
                    min_bucket_index = j

        # Clear expired arms from buckets and set their empirical rewards to -1
        for j in range(memory_size):
            arm = buckets[j]
            if arm is not None:
                if time_now - window_size > arm.t:
                    buckets[j] = None
                    empirical_rewards[j] = -1

        # Find the arm with the highest empirical reward among the buckets
        max_empirical_reward = -1
        max_bucket_index = 0
        for j in range(memory_size):
            if empirical_rewards[j] > max_empirical_reward:
                max_empirical_reward = empirical_rewards[j]
                max_bucket_index = j

        # The best arm we find is buckets[max_bucket_index] if it is not expired, otherwise the penalty arm
        best_arm = buckets[max_bucket_index]
        if best_arm is not None:
            if time_now - window_size > best_arm.t:
                best_arm = penalty_arm
        else:
            best_arm = penalty_arm
        
        best_rewards[time_now] = best_arm.p
    return best_rewards

def bucket_regret(arm_stream, input_memory_size, delta, budget_per_epoch):
    '''
    Implements bucket-based exploration with regret tracking for sliding window bandits.
    This algorithm extends bucket_exploration to also track cumulative regret while
    finding epsilon-best arms within budget constraints.
    
    :param arm_stream: The stream of arms to explore
    :param input_memory_size: The size of the memory (number of buckets - 1)
    :param delta: The failure probability threshold
    :param budget_per_epoch: Budget constraint per epoch for regret calculation
    :return: The cumulative regret counter
    '''

    # Extract basic parameters
    # Extract basic parameters
    num_arms = arm_stream.n
    window_size = arm_stream.window_size
    memory_size = input_memory_size+1

    # Initialize buckets to store arms, each bucket initially empty
    buckets = np.array([None]*memory_size)

    # Track the best reward found at each time step
    best_rewards = np.zeros(num_arms)

    # Set exploration accuracy parameter
    eps = 3/memory_size

    # Calculate theoretical bound for exploration accuracy
    # Based on budget constraints and statistical requirements
    eps_bound = (np.log(num_arms/delta)/budget_per_epoch)**(1/3)

    # Use the more conservative (larger) segment size for better guarantees
    if eps_bound > eps:
        segment = eps_bound/3
    else:
        segment = eps/3

    # Calculate number of pulls needed based on segment size and confidence
    num_pulls = int(np.ceil(1/(2*segment*segment) * np.log(num_arms/delta)))

    # Process each arm in the stream
    for i in range(num_arms):
        # Read the next arm from the stream
        current_arm = arm_stream.read_next_arm()
        # Get current time step
        time_now = arm_stream.t
        # Pull the arm multiple times to get empirical reward estimate
        empirical_reward = current_arm.batch_pull(num_pulls)
        # Determine which bucket this arm belongs to based on its reward
        current_bucket_index = int(np.floor(empirical_reward/segment))
        # Ensure bucket index doesn't exceed available buckets
        current_bucket_index = min(current_bucket_index, memory_size-1)
        # Place the arm in its corresponding bucket
        buckets[current_bucket_index] = current_arm

        # Remove expired arms that are outside the sliding window
        for j in range(memory_size):
            arm = buckets[j]
            if arm is not None:
                # Check if arm has expired (beyond window size)
                if time_now - window_size >= arm.t:
                    buckets[j] = None

        # Find the best arm by checking the highest-reward bucket that's not empty
        # Iterate from highest to lowest bucket index
        for j in range(memory_size-1, -1, -1):
            if buckets[j] is not None:
                # Store the true reward of the best arm found
                best_rewards[time_now] = buckets[j].p
                break

        # Update regret calculation for both exploration and exploitation phases
        # Regret from exploration (pulling current arm)
        arm_stream.regret_update_epoch(current_arm.p, time_now, min(num_pulls, budget_per_epoch))
        # Regret from exploitation (using best arm found)
        arm_stream.regret_update_epoch(best_rewards[time_now], time_now, max(budget_per_epoch-num_pulls, 0))
    
    return arm_stream.regret_counter

def reservoir_sampling(reservoir, current_time, current_arm, memory_size, window_size):
    '''
    Updates the reservoir using reservoir sampling algorithm for sliding window.
    Maintains a fixed-size sample of arms within the sliding window, ensuring
    each arm has equal probability of being selected for the reservoir.
    
    :param reservoir: Current reservoir of arms (list)
    :param current_time: Current time step
    :param current_arm: New arm to potentially add to reservoir
    :param memory_size: Maximum size of the reservoir
    :param window_size: Size of the sliding window
    :return: Updated reservoir
    '''

    # Remove expired arms that are outside the sliding window
    reservoir = [arm for arm in reservoir if current_time - window_size <= arm.t]

    # If reservoir has space, simply add the new arm
    if len(reservoir) < memory_size:
        reservoir.append(current_arm)
        return reservoir, None
    else:
        # Use reservoir sampling algorithm to decide whether to replace an existing arm
        # Generate random index in range [0, current_time+1]
        index = random.randint(0, current_time+1)
        # If index falls within reservoir size, replace that position
        if index < memory_size:
            reservoir[index] = current_arm
            return reservoir, index
        else:
            # Do not add the new arm to reservoir
            return reservoir, None
    

def ucb(arm_set, budget_per_epoch, delta):
    '''
    Implements Upper Confidence Bound (UCB) algorithm for multi-armed bandits.
    Balances exploration and exploitation by selecting arms with highest upper
    confidence bounds on their estimated rewards.
    
    :param arm_set: Set of available arms to choose from
    :param budget_per_epoch: Total budget (number of pulls) available
    :param delta: Confidence parameter for UCB bounds
    :return: Array containing number of pulls for each arm
    '''
    
    num_arms = len(arm_set)

    # Initialize pull counts (exploration phase)
    num_pulls = np.zeros(num_arms)
    
    # Initialize reward counters for each arm
    count = np.zeros(num_arms)
    
    # Pull each arm once initially to get baseline estimates
    for i in range(num_arms):
        count[i] = arm_set[i].pull()
        num_pulls[i] = num_pulls[i] + 1
        
    # Calculate UCB values: mean reward + confidence radius
    # The confidence radius decreases as we pull an arm more often
    preference = count + np.sqrt(np.log(budget_per_epoch*num_arms/delta)/num_pulls)
    
    # Use remaining budget to pull arms with highest UCB values
    for i in range(budget_per_epoch-num_arms):
        # Select arm with highest upper confidence bound
        current_idx = np.argmax(preference)
        
        # Pull selected arm and update its reward estimate
        count[current_idx] = count[current_idx] + arm_set[current_idx].pull()
        num_pulls[current_idx] = num_pulls[current_idx] + 1

        # Recalculate UCB value of the pulled arm with updated information
        preference[current_idx] = count[current_idx]/num_pulls[current_idx] + np.sqrt(np.log(budget_per_epoch*num_arms/delta)/num_pulls[current_idx])

    return num_pulls, preference

def advanced_ucb(arm_set, budget_per_epoch, delta, preference = None, memory_size = None, cumulative_num_pulls = None, count = None, new_index = -1):
    '''
    Implements Upper Confidence Bound (UCB) algorithm for multi-armed bandits.
    Balances exploration and exploitation by selecting arms with highest upper
    confidence bounds on their estimated rewards.

    :param arm_set: Set of available arms to choose from
    :param budget_per_epoch: Total budget (number of pulls) available
    :param delta: Confidence parameter for UCB bounds
    :param preference: Initial preference values for arms (optional)
    :param memory_size: Size of the memory (optional)
    :param cumulative_num_pulls: Initial cumulative number of pulls for each arm (optional)
    :param count: Initial reward counts for each arm (optional)
    :return: Array containing number of pulls for each arm
    '''

    num_arms = len(arm_set)

    # Initialize cumulative_num_pulls, count, and preference if num_arms is 1
    if num_arms == 1:
        cumulative_num_pulls = np.zeros(num_arms)
        epoch_num_pulls = np.zeros(num_arms)
        count = np.zeros(num_arms)
        preference = np.zeros(num_arms)
        
        count[0] = arm_set[0].batch_pull(budget_per_epoch)*budget_per_epoch
        cumulative_num_pulls[0] = budget_per_epoch
        epoch_num_pulls[0] = budget_per_epoch
        preference[0] = count[0]/cumulative_num_pulls[0] + np.sqrt(np.log(budget_per_epoch*num_arms/delta)/cumulative_num_pulls[0])
        return count, preference, cumulative_num_pulls, epoch_num_pulls

    # Raise error if there is not count, preference, and cumulative_num_pulls provided when num_arms > 1
    if count is None or preference is None or cumulative_num_pulls is None:
        raise ValueError("count, preference, and cumulative_num_pulls must be provided when num_arms > 1")
    # Raise error if their sizes do not match each other
    if len(count) != len(preference) or len(count) != len(cumulative_num_pulls):
        raise ValueError("count, preference, and cumulative_num_pulls must have the same size")    
    # Raise error if there is no memory_size provided when num_arms > 1
    if memory_size is None:
        raise ValueError("memory_size must be provided when num_arms > 1")

    # If len(cumulative_num_pulls) is larger than 1, adjust count, preference, and cumulative_num_pulls accordingly whether their sizes are less than memory_size.
    # If their size is less then memory_size, just append one zero for new arm; else, remove the oldest arm's data and append one zero for new arm if new_index is -1, or replace the new_index arm's data with the new arm's data if new_index is provided, or remain unchanged if new_index is None.
    if len(cumulative_num_pulls) < num_arms:
        cumulative_num_pulls = np.append(cumulative_num_pulls, 0)
        count = np.append(count, 0)
        preference = np.append(preference, 0)
    elif new_index is None:
        pass
    elif new_index == -1:
        cumulative_num_pulls = np.append(cumulative_num_pulls[1:], 0)
        count = np.append(count[1:], 0)
        preference = np.append(preference[1:], 0)
    elif new_index >= 0 and new_index < memory_size:
        cumulative_num_pulls[new_index] = 0
        count[new_index] = 0
        preference[new_index] = 0
    else:
        raise ValueError("new_index must be between 0 and memory_size-1")
    
      
    # Initialize epoch_num_pulls for the arm_set to record number of pulls in this epoch
    epoch_num_pulls = np.zeros(num_arms)

    # Pull new arm once to initialize if new_index is not None
    if new_index is not None:
        count[new_index] = arm_set[new_index].pull()
        cumulative_num_pulls[new_index] = cumulative_num_pulls[new_index] + 1
        epoch_num_pulls[new_index] = epoch_num_pulls[new_index] + 1
        preference[new_index] = count[new_index]/epoch_num_pulls[new_index] + np.sqrt(np.log(budget_per_epoch*num_arms/delta)/cumulative_num_pulls[new_index])

    # Use remaining budget to pull arms with highest UCB values
    for i in range(budget_per_epoch-1):
        # Select arm with highest upper confidence bound
        current_idx = np.argmax(preference)
        
        # Pull selected arm and update its reward estimate
        count[current_idx] = count[current_idx] + arm_set[current_idx].pull()
        cumulative_num_pulls[current_idx] = cumulative_num_pulls[current_idx] + 1
        epoch_num_pulls[current_idx] = epoch_num_pulls[current_idx] + 1

        # Recalculate UCB value of the pulled arm with updated information
        preference[current_idx] = count[current_idx]/cumulative_num_pulls[current_idx] + np.sqrt(np.log(budget_per_epoch*num_arms/delta)/cumulative_num_pulls[current_idx])

    return count, preference, cumulative_num_pulls, epoch_num_pulls


def sliding_window_ucb(arm_stream, budget_per_epoch, delta, input_memory_size):
    '''
    Implements sliding window UCB algorithm that adapts to changing environments.
    Applies UCB algorithm to arms within the sliding window, using either
    full window access or reservoir sampling based on memory constraints.
    
    :param arm_stream: The stream of arms with potential concept drift
    :param budget_per_epoch: Budget constraint per epoch
    :param delta: Confidence parameter for statistical guarantees
    :param input_memory_size: Available memory for storing arms
    :return: Cumulative regret over time
    '''

    # Extract stream parameters
    num_arms = arm_stream.n
    window_size = arm_stream.window_size

    # Check if we have enough memory to store entire window
    if input_memory_size+1 >= window_size:
        # Memory sufficient: use exact sliding window UCB
        # Apply UCB to the most recent window_size arms at each time step
        for i in range(1, num_arms):
            # Define the current window of arms
            latest_time = max(i-window_size+1, 0)
            arm_set = arm_stream.arm_set[latest_time:i+1]
            
            # Apply UCB algorithm to current window
            num_pulls, preference =  ucb(arm_set, budget_per_epoch, delta)

            # Update regret
            for j in range(len(arm_set)):
                arm_stream.regret_update_epoch(arm_set[j].p, i, num_pulls[j])

    else:
        # Memory insufficient: use reservoir sampling approximation
        # Initialize reservoir with the first arm
        reservoir = [arm_stream.read_next_arm()]
        
        # Process remaining arms using reservoir sampling
        for i in range(1, num_arms):
            # Read next arm and update reservoir
            current_arm = arm_stream.read_next_arm()
            reservoir, index = reservoir_sampling(reservoir, i, current_arm, input_memory_size+1, window_size)
            
            # Apply UCB to current reservoir
            num_pulls, preference = ucb(reservoir, budget_per_epoch, delta)

            # Update regret
            for j in range(len(reservoir)):
                arm_stream.regret_update_epoch(reservoir[j].p, i, num_pulls[j])

    return arm_stream.regret_counter

def advanced_sliding_window_ucb(arm_stream, budget_per_epoch, delta, input_memory_size):
    '''
    Implements sliding window UCB algorithm that adapts to changing environments.
    Applies UCB algorithm to arms within the sliding window, using either
    full window access or reservoir sampling based on memory constraints.

    :param arm_stream: The stream of arms with potential concept drift
    :param budget_per_epoch: Budget constraint per epoch
    :param delta: Confidence parameter for statistical guarantees
    :param input_memory_size: Available memory for storing arms
    :return: Cumulative regret over time
    '''

    # Extract stream parameters
    num_arms = arm_stream.n
    window_size = arm_stream.window_size

    # Check if we have enough memory to store entire window
    if input_memory_size+1 >= window_size:
        # Memory sufficient: use exact sliding window UCB
        # Apply UCB to the most recent window_size arms at each time step
        count, preference, cumulative_num_pulls = None, None, None
        for i in range(0, num_arms):
            # Define the current window of arms
            latest_time = max(i-window_size+1, 0)
            arm_set = arm_stream.arm_set[latest_time:i+1]
            
            # Apply UCB algorithm to current window
            count, preference, cumulative_num_pulls, epoch_num_pulls =  advanced_ucb(arm_set, budget_per_epoch, delta, preference=preference, memory_size=input_memory_size+1, cumulative_num_pulls=cumulative_num_pulls, count=count)

            # Update regret
            for j in range(len(arm_set)):
                arm_stream.regret_update_epoch(arm_set[j].p, i, epoch_num_pulls[j])

    else:
        # Memory insufficient: use reservoir sampling approximation
        # Initialize reservoir with the first arm
        reservoir = [arm_stream.read_next_arm()]

        count, preference, cumulative_num_pulls = None, None, None

        
        # Process remaining arms using reservoir sampling
        for i in range(0, num_arms):
            # Read next arm and update reservoir
            if i > 0:
                current_arm = arm_stream.read_next_arm()
                reservoir, index = reservoir_sampling(reservoir, i, current_arm, input_memory_size+1, window_size)
            elif i == 0:
                index = 0

            # Apply UCB algorithm to current window
            count, preference, cumulative_num_pulls, epoch_num_pulls =  advanced_ucb(reservoir, budget_per_epoch, delta, preference=preference, memory_size=input_memory_size+1, cumulative_num_pulls=cumulative_num_pulls, count=count, new_index=index)

            # Update regret
            for j in range(len(reservoir)):
                arm_stream.regret_update_epoch(reservoir[j].p, i, epoch_num_pulls[j])

    return arm_stream.regret_counter


def baseline_regret_clear_expiration(arm_stream, input_memory_size, delta, pulls_per_epoch):
    '''
    Explores arms using a baseline approach to minimize regret.
    This algorithm maintains a fixed set of arms and pulls the best arm among them.

    :param arm_stream: The stream of arms to explore
    :param input_memory_size: The size of the memory (number of buckets - 1)
    :param delta: The failure probability threshold
    :param pulls_per_epoch: The number of pulls allowed per epoch
    :return: Array of best rewards at each time step
    '''

    # Add 1 to memory size for bucket indexing
    memory_size = input_memory_size+1

    # Extract basic parameters from the arm stream
    num_arms = arm_stream.n
    window_size = arm_stream.window_size


    # Initialize buckets to store arms, each bucket initially empty (None)
    buckets = np.array([None]*memory_size)

    # Track the best reward found at each time step
    best_rewards = np.zeros(num_arms)

    # Set exploration accuracy parameter (eps = 3/memory_size)
    eps = 3/memory_size
    segment = 1/memory_size

    # Calculate number of pulls needed for reliable empirical estimates
    # Based on Hoeffding's inequality for concentration bounds
    num_pulls = int(np.ceil(1/(2*segment*segment) * np.log(num_arms/delta)))
    # num_pulls cannot exceed pulls_per_epoch
    num_pulls = min(num_pulls, pulls_per_epoch)

    # Generate an arm with p=0 for penalty for the expired arms
    penalty_arm = copy.deepcopy(arm_stream.arm_set[0])
    penalty_arm.p = 0

    # Initialize variables to track the bucket with the lowest empirical reward
    min_empirical_reward = -1
    min_bucket_index = 0

    # Initialize the empirical rewards for each bucket
    empirical_rewards = np.array([-1]*memory_size)

    # Process each arm in the stream sequentially
    for i in range(num_arms):
        # Read the next arm from the stream
        current_arm = arm_stream.read_next_arm()
        # Get current time step
        time_now = arm_stream.t
        # compute the empirical reward of the arm
        empirical_reward = current_arm.batch_pull(num_pulls)
        # Replace this arm to the arm stored in the bucket with the lowest empirical reward if it is better
        if empirical_reward > min_empirical_reward:
            buckets[min_bucket_index] = current_arm
            empirical_rewards[min_bucket_index] = empirical_reward

            # Update the minimum empirical reward and corresponding bucket index
            # min_empirical_reward is the lowest empirical reward among the empirical_rewards, and min_bucket_index is its index
            min_empirical_reward = empirical_rewards[0]
            min_bucket_index = 0
            for j in range(1, memory_size):
                if empirical_rewards[j] < min_empirical_reward:
                    min_empirical_reward = empirical_rewards[j]
                    min_bucket_index = j

        # Clear expired arms from buckets and set their empirical rewards to -1
        for j in range(memory_size):
            arm = buckets[j]
            if arm is not None:
                if time_now - window_size > arm.t:
                    buckets[j] = None
                    empirical_rewards[j] = -1

        # Find the arm with the highest empirical reward among the buckets
        max_empirical_reward = -1
        max_bucket_index = 0
        for j in range(memory_size):
            if empirical_rewards[j] > max_empirical_reward:
                max_empirical_reward = empirical_rewards[j]
                max_bucket_index = j

        # The best arm we find is buckets[max_bucket_index] if it is not expired, otherwise the penalty arm
        best_arm = buckets[max_bucket_index]
        if best_arm is not None:
            if time_now - window_size > best_arm.t:
                best_arm = penalty_arm
        else:
            best_arm = penalty_arm
        
        best_rewards[time_now] = best_arm.p

    # Compute the whole regret based on the best rewards found
    for i in range(num_arms):
        # At time i, we pull the i-th arm num_pulls times and the best_rewards[i] arm for the remaining pulls_per_epoch - num_pulls times
        arm_stream.regret_update_epoch(arm_stream.arm_set[i].p, i, num_pulls)
        arm_stream.regret_update_epoch(best_rewards[i], i, pulls_per_epoch - num_pulls)
    
    regret = arm_stream.regret_counter

    return regret

def reservoir_sampling_for_everlasting_arm(reservoir, current_time, current_arm, memory_size, window_size, signal, everlasting_arm_reward):
    '''
    Specialized reservoir sampling for scenarios with everlasting arms.
    Tracks whether an everlasting arm (with fixed high reward) has been
    detected and preserved beyond the sliding window.
    
    :param reservoir: Current reservoir of arms
    :param current_time: Current time step
    :param current_arm: New arm to potentially add
    :param memory_size: Maximum reservoir size
    :param window_size: Size of the sliding window
    :param signal: Boolean indicating if everlasting arm was found
    :param everlasting_arm_reward: Reward value of the everlasting arm
    :return: Tuple of (updated_reservoir, signal_status)
    '''

    # If everlasting arm already found, no need to update reservoir
    if signal:
        return reservoir, signal

    # Check if any expiring arms are the everlasting arm
    for i in range(len(reservoir)):
        if current_time - window_size >= reservoir[i].t:
            # If expiring arm has everlasting reward, preserve it
            if reservoir[i].p == everlasting_arm_reward:
                reservoir = reservoir[i]  # Keep only the everlasting arm
                return reservoir, True  # Signal that everlasting arm found
            
    # Remove expired arms (standard reservoir maintenance)
    reservoir = [arm for arm in reservoir if current_time - window_size <= arm.t]

    # Add current arm using standard reservoir sampling
    if len(reservoir) < memory_size:
        reservoir.append(current_arm)
    else:
        # Use reservoir sampling to decide replacement
        index = random.randint(0, current_time+1)
        if index < memory_size:
            reservoir[index] = current_arm
    
    return reservoir, False

def regret_minimization_everlasting(arm_stream, total_budget, input_memory_size):
    '''
    Implements regret minimization for streams containing everlasting arms.
    An everlasting arm is one that maintains optimal reward beyond the sliding window.
    The algorithm aims to detect and exploit such arms for zero regret.
    
    :param arm_stream: Stream of arms potentially containing everlasting arms
    :param total_budget: Total budget available for arm pulls
    :param input_memory_size: Available memory for storing arms
    :return: Final regret value (0 if everlasting arm found, else positive)
    '''

    # Extract stream parameters
    num_arms = arm_stream.n
    window_size = arm_stream.window_size
    best_reward = arm_stream.everlasting_reward

    # If memory exceeds window size, everlasting detection is trivial
    if input_memory_size+1 >= window_size:
        return 0  # Can always find everlasting arm with sufficient memory

    # Initialize reservoir with first arm
    reservoir = [arm_stream.read_next_arm()]
    signal = False  # Flag indicating if everlasting arm detected

    # Process each subsequent arm
    for i in range(1, num_arms):
        current_arm = arm_stream.read_next_arm()
        
        # Update reservoir and check for everlasting arm detection
        reservoir, signal = reservoir_sampling_for_everlasting_arm(
            reservoir, i, current_arm, input_memory_size+1, window_size, signal, best_reward)
        
        # If everlasting arm detected, we achieve zero regret
        if signal:
            return 0
    
    # Check if everlasting arm is in final reservoir
    if best_reward in reservoir:
        return 0  # Zero regret achieved
    else:
        # Everlasting arm not found, incur regret for suboptimal choice
        last_reward = arm_stream.rewards[-1]
        arm_stream.regret_update_everlasting(last_reward, total_budget)
        return arm_stream.regret_counter