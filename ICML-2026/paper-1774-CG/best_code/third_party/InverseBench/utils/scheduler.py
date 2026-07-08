import numpy as np
import copy

r'''
    Scheduler for diffusion sampling following EDM framework.
    schedule (\sigma(t)): linear, sqrt, vp
    timestep (discretization of t): log, poly-n, vp
    scaling: none, vp

    Example:
    VP: Scheduler(num_steps=1000, schedule='vp', timestep='vp', scaling='vp')
    VE: Scheduler(num_steps=1000, schedule='sqrt', timestep='log', scaling='none')
    EDM: Scheduler(num_steps=200, schedule='linear', timestep='poly-7', scaling='none')
    
    Example Usage: See DiffusionSampler in utils/diffusion.py for unconditional diffusion sampling.
    
    Subset Scheduler:
    To create a scheduler with only specific steps (e.g., for faster inference or
    coarse-to-fine sampling), use Scheduler.get_subset_scheduler():
    
        original = Scheduler(num_steps=1000, schedule='linear', timestep='poly-7')
        subset = Scheduler.get_subset_scheduler(original, [0, 100, 200, 300, 500, 700, 999])
    
    The factors (factor_steps, scaling_factor) are recalculated to account for the 
    larger \Delta t between non-consecutive steps.
'''
class Scheduler:
    """
        Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7', scaling='none'):
        """
            Initializes the scheduler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the schedule.
                sigma_max (float): Maximum value of sigma.
                sigma_min (float): Minimum value of sigma.
                sigma_final (float): Final value of sigma, defaults to sigma_min.
                schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
                timestep (str): Type of timestep function ('log' or 'poly-n').
                scaling (str): Type of scaling function ('none' or 'vp').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep
        self.scaling = scaling

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)
        scaling_fn, scaling_derivative_fn = self.get_scaling_fn(scaling)
        if self.schedule == 'vp':
            self.sigma_max = sigma_fn(1) * scaling_fn(1)
            
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])
        scaling_steps = np.array([scaling_fn(t) for t in time_steps])
        # scaling_factor = 1 - \dot s(t)/s(t) * \Delta t
        scaling_factor = np.array(
            [1 -  scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        # factor = 2 s(t)^2 \dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [2 * scaling_fn(time_steps[i])**2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps, self.scaling_factor, self.scaling_steps = sigma_steps, time_steps, factor_steps, scaling_factor, scaling_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
            Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        
        elif schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            sigma_fn = lambda t: np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t) - 1)
            sigma_derivative_fn = lambda t: (beta_d * t + beta_min)*np.exp(beta_d * t**2/2 + beta_min * t) / 2 / sigma_fn(t)
            sigma_inv_fn = lambda sigma: np.sqrt(beta_min**2 + 2*beta_d*np.log(sigma**2 + 1))/beta_d - beta_min/beta_d

        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_scaling_fn(self, schedule):
        if schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            scaling_fn = lambda t: 1/ np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
            scaling_derivative_fn = lambda t: - (beta_d * t + beta_min)/ 2 / np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
        else:
            scaling_fn = lambda t: 1
            scaling_derivative_fn = lambda t: 0
        return scaling_fn, scaling_derivative_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
            Returns the time step function based on the given timestep type.
        """
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        elif timestep == 'vp':
            get_time_step_fn = lambda r: 1 - r * (1 - 1e-3)
        else:
            raise NotImplementedError
        return get_time_step_fn

    @classmethod
    def get_partial_scheduler(cls, scheduler, new_sigma_max):
        """
            Generates a new scheduler with the given sigma_max value.
        """
        new_scheduler = copy.deepcopy(scheduler)
        num_steps = sum([s < new_sigma_max for s in scheduler.sigma_steps]) + 1

        new_scheduler.num_steps = num_steps - 1
        new_scheduler.sigma_max = new_sigma_max
        new_scheduler.sigma_steps = scheduler.sigma_steps[-num_steps:]
        new_scheduler.time_steps = scheduler.time_steps[-num_steps:]
        new_scheduler.factor_steps = scheduler.factor_steps[-num_steps:]
        new_scheduler.scaling_factor = scheduler.scaling_factor[-num_steps:]
        new_scheduler.scaling_steps = scheduler.scaling_steps[-num_steps:]
        return new_scheduler

    @classmethod
    def get_subset_scheduler(cls, scheduler, step_indices):
        """
            Creates a new scheduler that uses only a subset of steps from the original scheduler.
            The factors are recalculated to account for the larger jumps between non-consecutive steps.
            
            Parameters:
                scheduler (Scheduler): The original scheduler with all steps.
                step_indices (list[int]): List of step indices to include (0-indexed). 
                    For example, [0, 100, 200, 300, 400, 500, 550] for a 1000-step scheduler.
                    Must be sorted in ascending order and within range [0, scheduler.num_steps).
            
            Returns:
                Scheduler: A new scheduler with only the specified steps and recalculated factors.
            
            Example:
                original_scheduler = Scheduler(num_steps=1000, schedule='linear', timestep='poly-7')
                subset_scheduler = Scheduler.get_subset_scheduler(original_scheduler, [0, 100, 200, 300, 400, 500, 550])
        """
        step_indices = list(step_indices)
        
        # Validate inputs
        if not step_indices:
            raise ValueError("step_indices cannot be empty")
        if step_indices != sorted(step_indices):
            raise ValueError("step_indices must be sorted in ascending order")
        if step_indices[0] < 0 or step_indices[-1] >= scheduler.num_steps:
            raise ValueError(f"step_indices must be in range [0, {scheduler.num_steps})")
        
        new_scheduler = copy.deepcopy(scheduler)
        new_num_steps = len(step_indices)
        
        # Get the sigma and scaling functions for recalculating factors
        sigma_fn, sigma_derivative_fn, _ = scheduler.get_sigma_fn(scheduler.schedule)
        scaling_fn, scaling_derivative_fn = scheduler.get_scaling_fn(scheduler.scaling if hasattr(scheduler, 'scaling') else 'none')
        
        # Extract time_steps at the selected indices
        # Note: time_steps has num_steps+1 elements (includes the final step)
        # We need the time values at each selected index, plus the final time value
        selected_time_steps = [scheduler.time_steps[i] for i in step_indices]
        # Add the final time step - always use the original scheduler's final time step
        # This ensures we end at the same sigma_final as the original scheduler
        selected_time_steps.append(scheduler.time_steps[-1])
        
        selected_time_steps = np.array(selected_time_steps)
        
        # Recalculate sigma_steps and scaling_steps at selected time points
        selected_sigma_steps = np.array([sigma_fn(t) for t in selected_time_steps])
        selected_scaling_steps = np.array([scaling_fn(t) for t in selected_time_steps])
        
        # Recalculate scaling_factor and factor_steps with new delta t values
        # scaling_factor = 1 - \dot s(t)/s(t) * \Delta t
        new_scaling_factor = np.array([
            1 - scaling_derivative_fn(selected_time_steps[i]) / scaling_fn(selected_time_steps[i]) * 
            (selected_time_steps[i] - selected_time_steps[i + 1])
            for i in range(new_num_steps)
        ])
        
        # factor = 2 s(t)^2 \dot\sigma(t)\sigma(t)\Delta t
        new_factor_steps = np.array([
            2 * scaling_fn(selected_time_steps[i])**2 * 
            sigma_fn(selected_time_steps[i]) * sigma_derivative_fn(selected_time_steps[i]) * 
            (selected_time_steps[i] - selected_time_steps[i + 1])
            for i in range(new_num_steps)
        ])
        
        # Update the new scheduler
        new_scheduler.num_steps = new_num_steps
        new_scheduler.sigma_steps = selected_sigma_steps
        new_scheduler.time_steps = selected_time_steps
        new_scheduler.scaling_steps = selected_scaling_steps
        new_scheduler.scaling_factor = new_scaling_factor
        new_scheduler.factor_steps = [max(f, 0) for f in new_factor_steps]
        new_scheduler.sigma_max = selected_sigma_steps[0]
        
        # Store the original step indices for reference
        new_scheduler.original_step_indices = step_indices
        
        return new_scheduler

    @classmethod
    def get_evenly_spaced_subset_scheduler(cls, scheduler, num_subset_steps, start_step=0):
        """
            Creates a subset scheduler with evenly spaced steps from a starting position to the end.
            
            Parameters:
                scheduler (Scheduler): The original scheduler with all steps.
                num_subset_steps (int): Number of steps to include in the subset.
                start_step (int): The step index to start from (0-indexed). Default is 0.
                    For example, if you're at step 333 out of 1000 and want 20 steps to the end,
                    use start_step=333 and num_subset_steps=20.
            
            Returns:
                Scheduler: A new scheduler with evenly spaced steps from start_step to the end.
            
            Example:
                original_scheduler = Scheduler(num_steps=1000, schedule='linear', timestep='poly-7')
                
                # Full range: 10 steps from 0 to 999
                subset = Scheduler.get_evenly_spaced_subset_scheduler(original_scheduler, 10)
                # Creates steps at indices [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
                
                # Partial range: 20 steps from 333 to 999
                subset = Scheduler.get_evenly_spaced_subset_scheduler(original_scheduler, 20, start_step=333)
                # Creates steps at indices [333, 368, 403, 438, ..., 964, 999]
        """
        if start_step < 0 or start_step >= scheduler.num_steps:
            raise ValueError(f"start_step ({start_step}) must be in range [0, {scheduler.num_steps})")
        
        remaining_steps = scheduler.num_steps - start_step
        if num_subset_steps > remaining_steps:
            raise ValueError(f"num_subset_steps ({num_subset_steps}) cannot exceed remaining steps ({remaining_steps})")
        if num_subset_steps < 1:
            raise ValueError("num_subset_steps must be at least 1")
        
        # Generate evenly spaced indices from start_step to the last step
        step_indices = np.linspace(start_step, scheduler.num_steps - 1, num_subset_steps, dtype=int).tolist()
        
        return cls.get_subset_scheduler(scheduler, step_indices)
