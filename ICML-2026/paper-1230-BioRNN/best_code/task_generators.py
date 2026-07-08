import numpy as np
import torch

def flipflop(dims, dt,
    t_max=50,
    fixation_duration=1,
    stimulus_duration=1,
    decision_delay_duration=5,
    stim_delay_duration_min=5,
    stim_delay_duration_max=25,
    input_amp=1.,
    target_amp=0.5,
    fixate=False,
    choices=None,
    return_ts=False,
    test=False,
    ):
    """ 
    Flipflop task
    """
    dim_in, _, dim_out = dims
    
    if choices is None:
        choices = np.arange(dim_in)
    n_choices = len(choices)

    # Checks
    assert dim_out == dim_in, "Output and input dimensions must agree:    dim_out != dim_in."
    assert np.max(choices) <= (dim_in - 1), "The max choice must agree with input dimension!"

    # Task times
    fixation_duration_discrete = int(fixation_duration / dt)
    stimulus_duration_discrete = int(stimulus_duration / dt)
    decision_delay_duration_discrete = int(decision_delay_duration / dt)
    mean_stim_delay = 0.5 * (stim_delay_duration_min + stim_delay_duration_max)
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)

        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, dim_in))
            target_samp = np.zeros((n_t_max, dim_out))
            mask_samp = np.zeros((n_t_max, dim_out))

            idx_t = fixation_duration_discrete
            if fixate:
                # Mask
                mask_samp[:idx_t] = 1
            
            i_interval = 0
            test_intervals = np.array([16.55, 9.35, 14.80, 11.73, 12.17,  6.50, 13.06, 19.08, 13.19])
            test_choices = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1])
            test_signs = np.array([-1, 1, 1, 1, -1, -1, -1, -1, 1])
            while True:
                # New interval between pulses
                if test and b_idx == 0 and i_interval < len(test_choices):
                    interval = test_intervals[i_interval]
                else:
                    interval = np.random.uniform(stim_delay_duration_min, stim_delay_duration_max)
                # Add the decision delay
                interval += decision_delay_duration
                # New index
                n_t_interval = int(interval / dt)
                idx_tp1 = idx_t + n_t_interval

                # Choose input. 
                if test and b_idx == 0 and i_interval < len(test_choices):
                    choice = test_choices[i_interval]
                    sign = test_signs[i_interval]
                else:
                    choice = np.random.choice(choices)
                    sign = np.random.choice([1, -1])
                
                # Input
                input_samp[idx_t : idx_t + stimulus_duration_discrete, choice] = sign
                # Target
                target_samp[idx_t + decision_delay_duration_discrete : idx_tp1, choice] = sign
                # Mask
                mask_samp[idx_t + decision_delay_duration_discrete : idx_tp1] = 1
                # Update
                idx_t = idx_tp1
                i_interval += 1
                # Break
                if idx_t > n_t_max: break
                    
            # Join
            input_batch[b_idx] = input_samp
            target_batch[b_idx] = target_samp
            mask_batch[b_idx] = mask_samp

        # Scale by input and target amplitude
        input_batch *= input_amp
        target_batch *= target_amp

        return input_batch, target_batch, mask_batch
    
    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task
    
    
########################################################################################
def mante(dims, dt,
          choices=None,
          fixation_duration=3,
          stimulus_duration=20,
          delay_duration=5,
          decision_duration=20,
          input_amp=1.,
          target_amp=0.5,
          context_amp=1., 
          rel_input_std=0.05,
          coherences=None,
          fraction_catch_trails=0.,
          fixate=False,
          return_ts=False,
          test=False,
       ):
    """ 
    Mante task.
    """
    # Dimensions
    dim_in, _, dim_out = dims
    dim_cont = dim_in // 2
    dim_sens = dim_in - dim_cont
    # Checks
    assert dim_in % 2 == 0, "dim_in must be even"
    assert dim_out == 1, "dim_out != 1"

    if choices is None:
        choices = np.arange(dim_cont)
    # Not yet implemented
    assert np.max(choices) <= (dim_cont - 1), "The max choice must agree with input dimension!"
        
    # Task times
    fixation_duration_discrete = int(fixation_duration / dt)
    stim_begin = fixation_duration_discrete
    stimulus_duration_discrete = int(stimulus_duration / dt)
    stim_end = stim_begin + stimulus_duration_discrete
    delay_duration_discrete = int(delay_duration / dt)
    response_begin = stim_end + delay_duration_discrete
    decision_duration_discrete = int(decision_duration / dt)
    n_t_max = fixation_duration_discrete + stimulus_duration_discrete + delay_duration_discrete + \
                     decision_duration_discrete
    t_max = n_t_max * dt

    if coherences is None:
#         coherences = np.array([-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]) / 16.
#         coherences = np.array([-4, -2, -1, 1, 2, 4]) / 4.
        coherences = np.array([-8, -4, -2, -1, 1, 2, 4, 8]) / 8.
    elif type(coherences) == list:
        coherences = np.array(coherences)
        
    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        
        for b_idx in range(batch_size):
            target_samp = np.zeros((n_t_max, dim_out))
            mask_samp = np.zeros((n_t_max, dim_out))
            input_cont_samp = np.zeros((n_t_max, dim_cont))
            
            # Sensory input: noise and coherence
            input_sens_samp = np.random.randn(n_t_max, dim_sens) * rel_input_std / np.sqrt(dt)
            
            # Coherence signal and target
            if b_idx < (1 - fraction_catch_trails) * batch_size:
                # Draw random sensory coherences and context
                if test and b_idx == 0:
                    coh_i = coherences[[-1, 1]]
                    context = 1
                else:
                    coh_i = np.random.choice(coherences, dim_sens)
                    context = np.random.choice(choices)
                # Set input, context, target
                input_sens_samp[stim_begin:stim_end] += coh_i
                input_cont_samp[stim_begin:stim_end, context] = 1.
                target_samp[response_begin:] = (-1)**int(coh_i[context] < 0)
            # Mask
            mask_samp[response_begin:] = 1
            if fixate:
                mask_samp[:stim_end] = 1
                
            # Scale by input and target amplitude
            input_cont_samp *= context_amp
            input_sens_samp *= input_amp
            target_samp *= target_amp

            # Join
            input_batch[b_idx, :, :dim_cont] = input_cont_samp
            input_batch[b_idx, :, dim_cont:] = input_sens_samp
            target_batch[b_idx] = target_samp
            mask_batch[b_idx] = mask_samp
            
        return input_batch, target_batch, mask_batch
    
    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task
        
        
########################################################################################
def romo(dims, dt,
         fixation_duration=3,
         stimulus_duration=1,
         decision_delay_duration=5,
         decision_duration=10,
         stim_delay_duration_min=2,
         stim_delay_duration_max=8,
         input_amp_min=0.5,
         input_amp_max=1.5,
         min_input_diff=0.2,
         target_amp=0.5,
         fixate=True,
         return_ts=False,
         test=False,
         original_variant=False,
       ):
    """ 
    Romo task.
    """
    # Dimensions
    dim_in, _, dim_out = dims
    # Checks
    assert dim_in == 1, "dim_in != 1."
    if original_variant:
        assert dim_out == 1, "dim_out != 1."
    else:
        assert dim_out == 2, "dim_out != 2."

    # Task times
    fixation_duration_discrete = int(fixation_duration / dt)
    stimulus_duration_discrete = int(stimulus_duration / dt)
    decision_delay_duration_discrete = int(decision_delay_duration / dt)
    decision_duration_discrete = int(decision_duration / dt)
    stim_0_begin = fixation_duration_discrete
    stim_0_end = stim_0_begin + stimulus_duration_discrete
    # After stim_1, there is a random delay...
    t_max = (fixation_duration + 2 * stimulus_duration + stim_delay_duration_max
             + decision_delay_duration + decision_duration)
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        
        for b_idx in range(batch_size):
            input_samp = np.zeros((n_t_max, dim_in))
            target_samp = np.zeros((n_t_max, dim_out))
            mask_samp = np.zeros((n_t_max, dim_out))
            
            # Intervals
            if test:
                # Fixed stimulus delay
                # Mean
                stim_delay = 0.5 * (stim_delay_duration_min + stim_delay_duration_max)
            else:
                stim_delay = np.random.uniform(stim_delay_duration_min, stim_delay_duration_max)
            # Set indices
            stim_delay_discrete = int(stim_delay / dt)
            stim_1_begin = stim_0_end + stim_delay_discrete
            stim_1_end = stim_1_begin + stimulus_duration_discrete
            response_begin = stim_1_end + decision_delay_duration_discrete
            response_end = response_begin + decision_duration_discrete
                
            # Random input amplitudes
            if test:
                input_amps = [0.6, 1.1]
                # Fixed input amplitudes
            else:
                while True:
                    input_amps = np.random.uniform(input_amp_min, input_amp_max, size=2)
                    input_diff = np.abs(input_amps[0] - input_amps[1])
                    if input_diff >= min_input_diff:
                        break
            
            larger_input = np.argmax(input_amps)
                
            # Set input, target
            input_samp[stim_0_begin:stim_0_end] = input_amps[0]
            input_samp[stim_1_begin:stim_1_end] = input_amps[1]
            if original_variant:
                target_sign = (-1)**larger_input
                target_samp[response_begin:response_end] = target_sign * target_amp 
            else:
                target_samp[response_begin:response_end, larger_input] = target_amp 
            # Mask
            mask_samp[response_begin:response_end] = 1
            if fixate:
                # Set target output to zero until the decision delay
                mask_samp[:stim_1_end] = 1

            # Join
            input_batch[b_idx] = input_samp
            target_batch[b_idx] = target_samp
            mask_batch[b_idx] = mask_samp
            
        return input_batch, target_batch, mask_batch
    
    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task
    
    
########################################################################################
def input_driven_fp(dims, dt,
             t_max=15, 
             fixation_duration=1,
             decision_duration=1,
             target_amp=1.0, 
             input_amp=1.0, 
             return_ts=False,
            ):
    """ 
    Input-driven fixed point for linear network. 
    """
    dim_in, _, dim_out = dims
    
    # Not yet implemented
    assert dim_in == 1, "dim_in != 1"
    assert dim_out == 1, "dim_out != 1"

    # Task times
    fixation_duration_discrete = int(fixation_duration / dt)
    decision_duration_discrete = int(decision_duration / dt)
    n_t_max = int(t_max / dt)
    
    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        
        # Input: constant
        input_batch[:, fixation_duration_discrete:] = input_amp
        
        # Target: only the last point in time
        target_batch[:, -decision_duration_discrete:] = target_amp
        mask_batch[:, -decision_duration_discrete:] = 1.

        return input_batch, target_batch, mask_batch
    
    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task

    
########################################################################################
def auto_cos(dims, dt,
             t_max=10,
             period=5, 
             n_periods_loss=2,
             t_no_loss=0,
             target_amp=1., 
             return_ts=False,
            ):
    """ 
    Autonomous cosine generation. 
    """
    dim_in, _, dim_out = dims
    
    # Checks
    assert dim_in == 1, "Only one input!"
    assert dim_out == dim_in, "Output and input dimensions must agree:    dim_out != dim_in."

    # Times
    n_t_max = int(t_max / dt)
    # no-loss length in indices
    n_t_no_loss = int(t_no_loss / dt)
    # Length of loss: one full cycle
    t_loss = n_periods_loss * period
    n_t_loss = int(t_loss / dt)
    assert t_max >= (t_no_loss + t_loss)
    
    # Target signal
    ts = np.arange(0, t_max, dt)
    freq = 1 / period
    signal = np.cos(2 * np.pi * ts * freq)
    
    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)

        # Input
        # No input! -> use initial condition only!
#         input_batch[:, :n_t_pulse] = 1
        # Target
        target_batch[:] = signal[None, :, None]
        # Mask
        mask_batch[:, n_t_no_loss : n_t_no_loss + n_t_loss] = 1

        # Scale by target amplitude
        input_batch *= target_amp
        target_batch *= target_amp
        
        return input_batch, target_batch, mask_batch
    
    if return_ts:
        return task, ts
    else:
        return task


def perceptual_rt(dims, dt, choices=None, fixation_duration=100, stimulus_duration=800, no_reward=300, # decision_duration=300,
                  target_high=1.2, target_low=0.2, rel_input_std=0.05, coherences=[1, 2, 4, 8, 16],
                  fraction_catch_trails=0., return_ts=False, test=False, fixation_duration_test=300,
                  stimulus_duration_test=1500, ):
    """
    perceptual_rt task.
    """
    # Dimensions
    dim_in, _,  dim_out = dims
    SCALE = 3.2
    # Checks
    assert dim_in == 3, "RT task input dimension must be 3 (2 decision directions + 1 start signal)"
    assert dim_out == 2, "RT task output dimension must be 2 (two decision options)"

    if choices is None:
        choices = np.arange(dim_out)
    # Not yet implemented
    assert np.max(choices) <= (dim_out - 1), "The max choice must agree with input dimension!"

    # Task times
    fixation_duration_discrete = int(fixation_duration_test / dt) if test else int(fixation_duration / dt)
    stim_begin = fixation_duration_discrete
    stimulus_duration_discrete = int(stimulus_duration_test / dt) if test else int(stimulus_duration / dt)
    stim_end = stim_begin + stimulus_duration_discrete
    no_reward_discrete = int(no_reward / dt)
    response_begin = fixation_duration_discrete + no_reward_discrete

    n_t_max = fixation_duration_discrete + stimulus_duration_discrete
    t_max = n_t_max * dt

    if coherences is None:
        coherences = np.array([1, 2, 4, 8, 16])
    elif type(coherences) == list:
        coherences = np.array(coherences)


    def scale(coh):
        return (1 + SCALE * coh / 100) / 2

    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)

        for b_idx in range(batch_size):
            target_samp = np.zeros((n_t_max, dim_out))
            mask_samp = np.zeros((n_t_max, dim_out))
            input_samp = np.zeros((n_t_max, dim_in-1))
            input_pulse = np.zeros((n_t_max, 1))

            # Sensory input: noise and coherence
            input_noise_samp = np.random.randn(n_t_max, dim_in-1) * rel_input_std / np.sqrt(dt)

            # Coherence signal and target
            if b_idx < (1 - fraction_catch_trails) * batch_size:
                # Draw random sensory coherences and context
                if test and b_idx == 0:
                    coh_i = coherences[[-1, 1]]
                else:
                    coh_i = np.random.choice(coherences, dim_in-1)
                choice = np.random.choice(choices)

                # Set input, context, target
                input_samp[stim_begin:stim_end,choice] += scale(+coh_i[0])
                input_samp[stim_begin:stim_end, 1-choice] += scale(-coh_i[1])
                input_pulse[stim_begin:stim_begin+1] += 1
                input_samp += input_noise_samp

                target_samp[:stim_begin] = target_low
                target_samp[response_begin:,choice] =  target_high
                target_samp[response_begin:,1-choice] = target_low
            # Mask
            mask_samp[:stim_begin] = 1
            mask_samp[response_begin:] = 1

            # Join
            input_batch[b_idx, :, :dim_in-1] = input_samp
            input_batch[b_idx, :, dim_in-1:] = input_pulse

            target_batch[b_idx] = target_samp
            mask_batch[b_idx] = mask_samp

        dtype = torch.FloatTensor
        input_batch = torch.from_numpy(input_batch).type(dtype)
        target_batch = torch.from_numpy(target_batch).type(dtype)
        mask_batch = torch.from_numpy(mask_batch).type(dtype)

        return input_batch, target_batch, mask_batch

    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task


def perceptual_vs(dims, dt, choices=None, fixation_duration=100, stimulus_duration=800, no_reward=300,
                  # decision_duration=300,
                  target_high=1.2, target_low=0.2, rel_input_std=0.05, coherences=[1, 2, 4, 8, 16],
                  fraction_catch_trails=0., return_ts=False, test=False, fixation_duration_test=300,
                  stimulus_duration_test=1500, ):
    """
    perceptual_rt task.
    """
    # Dimensions
    dim_in, _, dim_out = dims
    SCALE = 3.2
    # Checks
    assert dim_in == 3, "VS Task Input Dimension must be 3 (2 decision directions + 1 starting signal)"
    assert dim_out == 2, "VS Task Output Dimension Must Be 2 (Two Decision Options)"

    if choices is None:
        choices = np.arange(dim_out)
    # Not yet implemented
    assert np.max(choices) <= (dim_out - 1), "The max choice must agree with input dimension!"

    # Task times
    fixation_duration_discrete = int(fixation_duration_test / dt) if test else int(fixation_duration / dt)
    stim_begin = fixation_duration_discrete
    stimulus_duration_discrete = int(stimulus_duration_test / dt) if test else int(stimulus_duration / dt)
    stim_end = stim_begin + stimulus_duration_discrete
    no_reward_discrete = int(no_reward / dt)
    response_begin = fixation_duration_discrete + no_reward_discrete

    n_t_max = fixation_duration_discrete + stimulus_duration_discrete
    t_max = n_t_max * dt

    if coherences is None:
        coherences = np.array([1, 2, 4, 8, 16])
    elif type(coherences) == list:
        coherences = np.array(coherences)

    def scale(coh):
        return (1 + SCALE * coh / 100) / 2

    def task(batch_size):
        # Input and target sequences
        input_batch = np.zeros((batch_size, n_t_max, dim_in), dtype=np.float32)
        target_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)
        mask_batch = np.zeros((batch_size, n_t_max, dim_out), dtype=np.float32)

        for b_idx in range(batch_size):
            target_samp = np.zeros((n_t_max, dim_out))
            mask_samp = np.zeros((n_t_max, dim_out))
            input_samp = np.zeros((n_t_max, dim_in - 1))
            input_pulse = np.zeros((n_t_max, 1))

            # Sensory input: noise and coherence
            input_noise_samp = np.random.randn(n_t_max, dim_in - 1) * rel_input_std / np.sqrt(dt)

            # Coherence signal and target
            if b_idx < (1 - fraction_catch_trails) * batch_size:
                # Draw random sensory coherences and context
                if test and b_idx == 0:
                    coh_i = coherences[[-1, 1]]
                else:
                    coh_i = np.random.choice(coherences, dim_in - 1)
                choice = np.random.choice(choices)

                # Set input, context, target
                input_samp[stim_begin:stim_end, choice] += scale(+coh_i[0])
                input_samp[stim_begin:stim_end, 1 - choice] += scale(-coh_i[1])
                input_pulse[stim_begin:stim_begin + 1] += 1
                input_samp += input_noise_samp

                target_samp[:stim_begin] = target_low
                target_samp[response_begin:, choice] = target_high
                target_samp[response_begin:, 1 - choice] = target_low
            # Mask
            mask_samp[:stim_begin] = 1
            mask_samp[response_begin:] = 1

            # Join
            input_batch[b_idx, :, :dim_in - 1] = input_samp
            input_batch[b_idx, :, dim_in - 1:] = input_pulse

            target_batch[b_idx] = target_samp
            mask_batch[b_idx] = mask_samp

        dtype = torch.FloatTensor
        input_batch = torch.from_numpy(input_batch).type(dtype)
        target_batch = torch.from_numpy(target_batch).type(dtype)
        mask_batch = torch.from_numpy(mask_batch).type(dtype)

        return input_batch, target_batch, mask_batch

    if return_ts:
        # Array of times
        ts = np.arange(0, t_max, dt)
        return task, ts
    else:
        return task
