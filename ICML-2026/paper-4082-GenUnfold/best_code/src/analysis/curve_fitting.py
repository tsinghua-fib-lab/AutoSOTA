# src/analysis/curve_fitting.py

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import logging
from typing import Tuple, Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Worm-Like Chain (WLC) model definition
# Force (F) as a function of Extension (x)
# F(x) = (kB*T/p) * [ 1/(4*(1 - x/Lc)^2) - 1/4 + x/Lc ]
# kB: Boltzmann constant (J/K)
# T: Temperature (K)
# p: Persistence Length (nm)
# Lc: Contour Length (nm)

# Physical constants (example values - use consistent units with your data)
KB = 1.380649e-23 # J/K
# Convert J/K to pN*nm/K for common SMFS units (1 J = 1e12 pN*nm)
KB_PN_NM = KB * 1e21 # pN*nm/K

def wlc_model(x: np.ndarray, p: float, Lc: float, temperature_K: float) -> np.ndarray:
    """
    The Worm-Like Chain (WLC) model equation for force as a function of extension.

    Args:
        x (np.ndarray): Array of extension values (in units consistent with Lc and p).
                         Assumes x is in the range [0, Lc).
        p (float): Persistence Length (in units consistent with x and Lc).
        Lc (float): Contour Length (in units consistent with x and p).
        temperature_K (float): Temperature in Kelvin.

    Returns:
        np.ndarray: Array of force values predicted by the WLC model.
    """
    # Ensure x is not exactly Lc to avoid division by zero
    # In practice, fitting is done on x/Lc ratios < 1
    x_norm = x / Lc

    # Handle potential division by zero or values > 1
    # x_norm should be < 1
    valid_mask = (x_norm >= 0) & (x_norm < 1.0)
    force = np.full_like(x, np.nan) # Fill with NaN where invalid

    # Apply WLC model only to valid range
    if np.any(valid_mask):
        valid_x_norm = x_norm[valid_mask]
        force[valid_mask] = (KB_PN_NM * temperature_K / p) * (
            1.0 / (4.0 * (1.0 - valid_x_norm)**2) - 1.0/4.0 + valid_x_norm
        )

    # The WLC model predicts infinite force at x=Lc, which is not physical.
    # Often, experimental data doesn't reach this point.
    # If x > Lc, the model is not valid. We return NaN for simplicity.
    # In some fitting contexts, values slightly > Lc might be handled or excluded.
    # Ensure x is in the range [0, Lc) for valid WLC.

    return force


def fit_wlc_segment(
    segment_extension: np.ndarray,
    segment_force: np.ndarray,
    temperature_K: float,
    p_guess: float = 0.9, # Initial guess for persistence length (nm)
    Lc_guess: float = None, # Initial guess for contour length (nm)
    p_bounds: Tuple[float, float] = (0.1, 5.0), # Bounds for persistence length
    Lc_bounds_min_factor: float = 1.0, # Lower bound for Lc = max(extension) * factor
    Lc_bounds_max_factor: float = 2.0, # Upper bound for Lc = max(extension) * factor
    force_unit_conversion: float = 1.0 # Factor to convert force units to pN if needed
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray] | Tuple[None, None, None]:
    """
    Fits the WLC model to a single segment of an F-E curve.
    This function is designed to work on (extension, force) pairs,
    but adapting it to standardized curves requires careful consideration
    of the extension axis scaling.

    Args:
        segment_extension (np.ndarray): Array of extension values for the segment.
        segment_force (np.ndarray): Array of force values for the segment.
        temperature_K (float): Temperature in Kelvin.
        p_guess (float): Initial guess for persistence length (nm).
        Lc_guess (float, optional): Initial guess for contour length (nm).
                                    If None, defaults to a value based on max extension.
        p_bounds (Tuple[float, float]): Bounds for persistence length (min, max).
        Lc_bounds_min_factor (float): Factor multiplied by max extension to set
                                      the lower bound for Lc.
        Lc_bounds_max_factor (float): Factor multiplied by max extension to set
                                      the upper bound for Lc.
        force_unit_conversion (float): Factor to convert input force units to pN
                                       if your data is not in pN. The WLC model
                                       uses pN for kB_PN_NM.

    Returns:
        Tuple[Dict[str, float], Dict[str, float], np.ndarray] | Tuple[None, None, None]:
            - fitted_params: Dictionary of fitted parameters {'p': p_fit, 'Lc': Lc_fit}.
            - param_errors: Dictionary of estimated standard errors for fitted parameters.
            - fitted_curve: Numpy array of force values predicted by the fitted WLC model
                            at the input segment_extension points.
            Returns (None, None, None) if fitting fails.
    """
    if len(segment_extension) != len(segment_force) or len(segment_extension) < 5: # Need enough points for fitting
        logging.warning("Segment too short or mismatch in length for WLC fitting.")
        return None, None, None
    if np.max(segment_extension) == 0:
         logging.warning("Maximum extension is zero in the segment. Cannot fit WLC.")
         return None, None, None

    # Apply force unit conversion to match kB_PN_NM units
    segment_force_pN = segment_force * force_unit_conversion

    # Define the function to fit (WLC model, fixing temperature)
    # curve_fit expects f(x, *params)
    def wlc_fit_func(x, p, Lc):
        return wlc_model(x, p, Lc, temperature_K=temperature_K)

    # Set initial guess for Lc if not provided
    if Lc_guess is None:
         Lc_guess = np.max(segment_extension) * 1.1 # Common heuristic

    # Set bounds for Lc based on max extension in the segment
    max_ext = np.max(segment_extension)
    Lc_bounds = (max_ext * Lc_bounds_min_factor, max_ext * Lc_bounds_max_factor)

    # Combine bounds for p and Lc
    bounds = ([p_bounds[0], Lc_bounds[0]], [p_bounds[1], Lc_bounds[1]])
    initial_guess = [p_guess, Lc_guess]

    try:
        # Perform the curve fitting
        # Use sigma to weight points? Absolute_sigma=True for actual standard errors
        params, pcov = curve_fit(
            wlc_fit_func,
            segment_extension,
            segment_force_pN,
            p0=initial_guess,
            bounds=bounds,
            # method='dogbox' # Sometimes helps with bounds
            absolute_sigma=True # Use if errors in segment_force are known
        )

        # Extract fitted parameters and their estimated standard errors
        fitted_params = {'p': params[0], 'Lc': params[1]}
        param_errors = {}
        if pcov is not None and np.all(np.isfinite(pcov)):
             # Estimated standard errors are the square root of the diagonal of the covariance matrix
            perr = np.sqrt(np.diag(pcov))
            param_errors = {'p': perr[0], 'Lc': perr[1]}
        else:
            logging.warning("Covariance matrix is not finite. Could not estimate parameter errors.")
            param_errors = {'p': np.nan, 'Lc': np.nan}


        # Generate the fitted WLC curve using the fitted parameters
        fitted_curve_pN = wlc_model(segment_extension, params[0], params[1], temperature_K)
        # Convert back to original force units if conversion was applied
        fitted_curve = fitted_curve_pN / force_unit_conversion


        logging.debug(f"WLC fitting successful. Fitted params: p={fitted_params['p']:.2f}, Lc={fitted_params['Lc']:.2f}")
        return fitted_params, param_errors, fitted_curve

    except RuntimeError as e:
        logging.warning(f"WLC fitting failed: {e}")
        return None, None, None
    except ValueError as e:
        logging.warning(f"WLC fitting failed due to ValueError (e.g., invalid bounds or data): {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during WLC fitting: {e}")
        return None, None, None

def fit_wlc_to_unfolding_segments(
    extension_curve: np.ndarray, # Original extension data corresponding to the force curve
    force_curve: np.ndarray,
    peak_indices: np.ndarray,
    temperature_K: float,
    # Add WLC fitting parameters and bounds here
    p_guess: float = 0.9,
    p_bounds: Tuple[float, float] = (0.1, 5.0),
    Lc_bounds_min_factor: float = 1.0,
    Lc_bounds_max_factor: float = 2.0,
    force_unit_conversion: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Identifies segments between unfolding peaks and attempts to fit the WLC model
    to each segment.

    Args:
        extension_curve (np.ndarray): The original extension data points corresponding
                                      to the force_curve. Crucial for WLC fitting.
        force_curve (np.ndarray): The Force-Extension curve force values.
        peak_indices (np.ndarray): Indices of the detected peaks in the force_curve.
        temperature_K (float): Temperature in Kelvin.
        p_guess, p_bounds, Lc_bounds_min_factor, Lc_bounds_max_factor, force_unit_conversion:
            Parameters for fit_wlc_segment.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              contains the fitting results for a segment, including
                              fitted parameters, errors, and the index of the preceding peak (or start).
                              Returns empty list if no segments can be fitted.
    """
    fitting_results = []

    # Identify segments. Segments are between a start point (0 or previous peak)
    # and an end point (next peak or end of curve).
    # The force drops after a peak, so the stretching segment starts AFTER the peak index.

    # Indices defining the start of each segment
    segment_starts = np.sort(peak_indices) + 1 # Start after each peak
    segment_starts = np.insert(segment_starts, 0, 0) # Add the start of the curve (index 0)

    # Indices defining the end of each segment
    segment_ends = np.append(np.sort(peak_indices), len(force_curve) - 1) # Each peak is an end, plus the end of the curve

    logging.info(f"Attempting WLC fitting on {len(segment_starts)} segments based on {len(peak_indices)} peaks.")

    for i in range(len(segment_starts)):
        start_idx = segment_starts[i]
        end_idx = segment_ends[i] # The peak index marking the end of stretching (before the drop)

        # Ensure the segment is valid and has enough points
        # Need at least 2 points for a segment, but realistically more for stable WLC fit
        if end_idx > start_idx + 4: # Require at least 5 points in the segment
            segment_extension_data = extension_curve[start_idx : end_idx + 1] # Include the end point
            segment_force_data = force_curve[start_idx : end_idx + 1]

            logging.debug(f"Fitting segment from index {start_idx} to {end_idx}.")

            # Attempt to fit WLC to this segment
            fitted_params, param_errors, fitted_curve_segment = fit_wlc_segment(
                segment_extension_data,
                segment_force_data,
                temperature_K=temperature_K,
                p_guess=p_guess,
                p_bounds=p_bounds,
                Lc_bounds_min_factor=Lc_bounds_min_factor,
                Lc_bounds_max_factor=Lc_bounds_max_factor,
                force_unit_conversion=force_unit_conversion
            )

            if fitted_params is not None:
                # Store the fitting results
                result = {
                    'segment_start_idx': start_idx,
                    'segment_end_idx': end_idx,
                    'fitted_params': fitted_params,
                    'param_errors': param_errors,
                    # Optional: Store the fitted curve segment
                    # 'fitted_curve_segment': fitted_curve_segment
                }
                # Calculate Delta Lc for the segment (change in contour length)
                # This requires knowing the contour length of the *previous* segment or the folded state.
                # For the first segment (folded protein), Lc is the contour length of the folded protein + linkers.
                # For subsequent segments, Delta Lc = Lc_this_segment - Lc_previous_segment.
                # This is complex and requires tracking Lc across segments.
                # A simpler approach might be to assume Lc of the folded state and calculate Delta Lc
                # as Lc_fitted - Lc_folded. Or fit Lc and Delta Lc simultaneously.

                # Placeholder for Delta Lc calculation
                # Delta Lc often corresponds to the length of the unfolded domain.
                # For simplicity, let's store the fitted Lc. Calculating meaningful Delta Lc
                # requires more context (previous state Lc).
                result['fitted_Lc'] = fitted_params['Lc']
                result['fitted_p'] = fitted_params['p']

                fitting_results.append(result)
            else:
                logging.debug(f"WLC fitting failed for segment {start_idx}-{end_idx}.")


    logging.info(f"Finished WLC fitting. Successfully fitted {len(fitting_results)} segments.")
    return fitting_results


# Example Usage
if __name__ == "__main__":
    print("--- Testing curve_fitting.py ---")

    # Create a dummy F-E curve with peaks (same as in mechanical_properties.py example)
    fe_len = 500
    ext_points = np.arange(fe_len) # Dummy extension points (indices)
    dummy_curve_force = np.zeros(fe_len)

    # Simulate some stretching and unfolding events
    # Peak 1 (WLC-like stretch)
    p1_Lc = 50 # nm
    p1_p = 0.9 # nm
    temp_K = 300 # K
    wlc_ext1 = np.linspace(1, 45, 100) # Extension in nm
    wlc_force1 = wlc_model(wlc_ext1, p1_p, p1_Lc, temp_K)
    peak1_idx = 100 # Index in the 500 point dummy curve
    dummy_curve_force[:peak1_idx] = wlc_force1[:peak1_idx] # Use first 100 points of WLC

    dummy_curve_force[peak1_idx:peak1_idx+50] = np.max(dummy_curve_force[:peak1_idx]) - np.linspace(0, np.max(dummy_curve_force[:peak1_idx])*0.8, 50) # Drop

    # Peak 2 (WLC-like stretch after unfolding 1 domain)
    # Assume contour length increased by ~30 nm (typical domain size)
    p2_Lc = p1_Lc + 30 # nm
    p2_p = 0.9 # nm (assume same persistence length for unfolded chain)
    wlc_ext2 = np.linspace(40, 75, 150) # Extension in nm
    wlc_force2 = wlc_model(wlc_ext2, p2_p, p2_Lc, temp_K)
    peak2_idx = 200 # Index in dummy curve
    start_idx2 = peak1_idx + 50 # Start after drop
    dummy_curve_force[start_idx2:peak2_idx] = wlc_force2[:(peak2_idx - start_idx2)] # Use points from WLC

    dummy_curve_force[peak2_idx:peak2_idx+70] = np.max(dummy_curve_force[start_idx2:peak2_idx]) - np.linspace(0, np.max(dummy_curve_force[start_idx2:peak2_idx])*0.9, 70) # Drop

    # Peak 3 (WLC-like stretch after unfolding 2 domains)
    p3_Lc = p2_Lc + 30 # nm
    p3_p = 0.9 # nm
    wlc_ext3 = np.linspace(70, 100, fe_len - (peak2_idx + 70)) # Extension in nm
    wlc_force3 = wlc_model(wlc_ext3, p3_p, p3_Lc, temp_K)
    start_idx3 = peak2_idx + 70
    dummy_curve_force[start_idx3:] = wlc_force3 # Use remaining points

    # Need a realistic extension axis corresponding to the forces
    # This is where the challenge with standardized curves arises.
    # Let's assume the original extension varied from 0 to ~100 nm over 500 points
    # A simplified approach: assume linear scaling of the index to extension
    physical_max_ext = 150 # nm (example total extension range)
    dummy_extension_physical = np.linspace(0, physical_max_ext, fe_len)


    # Add some noise
    dummy_curve_force += np.random.randn(fe_len) * 5


    # Find peaks first (needed to define segments)
    from mechanical_properties import find_force_peaks
    peak_indices, peak_properties = find_force_peaks(
        dummy_curve_force,
        height=20,
        distance=30,
        prominence=10,
        smoothing_window=5
    )
    print(f"Found {len(peak_indices)} peaks for WLC fitting segments.")

    # --- Test fit_wlc_to_unfolding_segments ---
    print("\n--- Testing fit_wlc_to_unfolding_segments ---")
    # Provide the physical extension data and the force data
    fitting_results = fit_wlc_to_unfolding_segments(
        dummy_extension_physical, # Use the physical extension data
        dummy_curve_force,
        peak_indices,
        temperature_K=temp_K,
        p_guess=0.8, # Slightly different initial guess
        p_bounds=(0.1, 3.0),
        Lc_bounds_min_factor=1.0,
        Lc_bounds_max_factor=3.0
    )

    print(f"Number of segments successfully fitted: {len(fitting_results)}")
    for i, result in enumerate(fitting_results):
        print(f"Segment {i+1} (Indices {result['segment_start_idx']}-{result['segment_end_idx']}):")
        print(f"  Fitted p: {result['fitted_p']:.2f}, Fitted Lc: {result['fitted_Lc']:.2f}")
        print(f"  Parameter errors (p, Lc): {result['param_errors'].get('p', np.nan):.2f}, {result['param_errors'].get('Lc', np.nan):.2f}")


    # Plotting the segments and fits (requires matplotlib)
    plt.figure(figsize=(12, 8))
    plt.plot(dummy_extension_physical, dummy_curve_force, label='Dummy F-E Curve (Physical Extension)')
    plt.plot(dummy_extension_physical[peak_indices], dummy_curve_force[peak_indices], "x", label="Detected Peaks")

    colors = ['red', 'green', 'purple', 'orange', 'brown'] # Colors for segments
    for i, result in enumerate(fitting_results):
        start_idx = result['segment_start_idx']
        end_idx = result['segment_end_idx']
        segment_extension = dummy_extension_physical[start_idx : end_idx + 1]

        # Generate the fitted WLC curve for plotting
        fitted_p = result['fitted_params']['p']
        fitted_Lc = result['fitted_params']['Lc']
        fitted_curve_segment = wlc_model(segment_extension, fitted_p, fitted_Lc, temp_K)

        plt.plot(segment_extension, dummy_curve_force[start_idx : end_idx + 1], '.', color=colors[i % len(colors)], label=f'Segment {i+1} Data')
        plt.plot(segment_extension, fitted_curve_segment, '--', color=colors[i % len(colors)], label=f'Segment {i+1} WLC Fit (p={fitted_p:.1f}, Lc={fitted_Lc:.1f})')
        plt.vlines(dummy_extension_physical[end_idx], np.min(dummy_curve_force), dummy_curve_force[end_idx], color=colors[i % len(colors)], linestyle=':', linewidth=1) # Mark end of segment (before drop)


    plt.xlabel("Extension (nm)")
    plt.ylabel("Force (pN)") # Assuming force is in pN
    plt.title("Dummy F-E Curve with WLC Fits to Segments")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Ensure force axis starts at 0
    plt.xlim(left=0) # Ensure extension axis starts at 0
    plt.show()