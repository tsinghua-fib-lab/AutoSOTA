# src/analysis/mechanical_properties.py

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian
from scipy.ndimage import uniform_filter1d
import logging
from typing import List, Dict, Any, Tuple

# Assuming the standardized F-E curve is a vector of force values
# corresponding to equally spaced points along a standardized extension axis (e.g., [0, 1]).
# To calculate unfolding energy, we need the "extension step" corresponding
# to the spacing between points in the standardized extension.
# If standardizing maps original extension [min_ext, max_ext] to [0, 1],
# and the curve has length L, the standardized extension step is 1.0 / (L - 1).
# To get a physical work value, you'd ideally need the mapping back to the original extension scale.
# For simplicity here, we calculate "standardized work" or assume a unit extension step.
# A more accurate energy would require storing/inferring the original extension scaling.

def calculate_unfolding_energy(
    force_curve: np.ndarray,
    extension_step: float = 1.0 # The actual extension step in physical units (e.g., nm)
) -> float:
    """
    Calculates the unfolding energy (work done) by numerically integrating
    the Force-Extension curve.

    Args:
        force_curve (np.ndarray): A 1D numpy array of force values from the F-E curve.
                                  Assumes forces correspond to equally spaced extension points.
        extension_step (float): The physical spacing between consecutive points
                                along the extension axis (e.g., in nm).
                                If using a standardized extension axis [0, 1], this would
                                be 1.0 / (length - 1) * effective_max_extension.
                                Providing the actual physical step is crucial for meaningful energy.

    Returns:
        float: The calculated unfolding energy (work done).
               Units: (Force Units) * (Extension Units).
    """
    if len(force_curve) < 2:
        logging.warning("F-E curve too short to calculate energy.")
        return 0.0

    # Numerical integration using the trapezoidal rule
    # Assumes equal spacing, so sum of forces * extension_step
    # If extension_step is 1.0, this is just the sum of forces.
    energy = np.sum(force_curve) * extension_step

    return energy

def calculate_max_force(force_curve: np.ndarray,
                        peak_indices: np.ndarray = None,
                        detachment: bool = False,
                        **kwargs
                        ) -> float:
    """
    Finds the global maximum force in the F-E curve.

    Args:
        force_curve (np.ndarray): A 1D numpy array of force values from the F-E curve.
        peak_indices (np.ndarray): A 1D numpy array of force peaks detected by the F-E curve.
        detachment (bool): If detachment is true, the last peak index will be omitted.

    Returns:
        float: The maximum force value.

    """
    if peak_indices is None:
        peak_indices = np.array([], dtype=int)

    if detachment:
        peak_indices = peak_indices[:-1]

    if peak_indices.size < 1:
        return np.max(force_curve)

    return np.max(force_curve[peak_indices])

def find_force_peaks(
    force_curve: np.ndarray,
    height: float = None, # Required height of peaks
    threshold: float = None, # Required vertical distance to its neighboring samples
    distance: float = None, # Required minimum horizontal distance between neighboring peaks
    prominence: float = None, # Required prominence of peaks
    width: float = None, # Required width of peaks
    wlen: int = None, # Window length for prominence calculation
    rel_height: float = 0.5, # For prominence calculation
    # Add parameters for smoothing if needed
    smoothing_window: int = None, # Window size for moving average smoothing
    smoothing_sigma: float = None # Sigma for Gaussian smoothing
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Finds peaks (unfolding events) in the Force-Extension curve using scipy.signal.find_peaks.

    Args:
        force_curve (np.ndarray): A 1D numpy array of force values from the F-E curve.
        height, threshold, distance, prominence, width, wlen, rel_height:
            Parameters for scipy.signal.find_peaks. Refer to scipy documentation.
            Setting appropriate values is crucial for accurate peak detection.
        smoothing_window (int, optional): If specified, applies a moving average
                                         filter before peak finding. Defaults to None.
        smoothing_sigma (float, optional): If specified, applies a Gaussian filter
                                          before peak finding. Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]:
            - peak_indices: A 1D numpy array of the indices where peaks are found.
            - peak_properties: A dictionary containing properties of the peaks
                               (e.g., 'peak_heights', 'left_bases', etc.)
                               matching the output of scipy.signal.find_peaks.
                               :param threshold:
                               :param force_curve:
                               :param height:
    """
    if len(force_curve) == 0:
        logging.warning("F-E curve is empty. Cannot find peaks.")
        return np.array([], dtype=int), {}

    processed_curve = force_curve.copy()

    # Apply smoothing if specified
    if smoothing_window is not None:
        if smoothing_window > 1:
            processed_curve = uniform_filter1d(processed_curve, size=smoothing_window)
            logging.debug(f"Applied moving average smoothing with window size {smoothing_window}.")
        else:
            logging.warning("Smoothing window must be > 1 for moving average.")

    if smoothing_sigma is not None:
        if smoothing_sigma > 0:
             # Apply Gaussian smoothing. sigma corresponds to the standard deviation of the Gaussian kernel.
             # order=0 means Gaussian smoothing (not derivative)
             # truncate determines how far the filter extends from the center
             truncate = 4.0 # Common value
             processed_curve = gaussian(processed_curve, sigma=smoothing_sigma, truncate=truncate, mode='nearest')
             logging.debug(f"Applied Gaussian smoothing with sigma {smoothing_sigma}.")
        else:
            logging.warning("Smoothing sigma must be > 0 for Gaussian smoothing.")


    # Use scipy.signal.find_peaks
    peak_indices, peak_properties = find_peaks(
        processed_curve,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height
    )

    logging.debug(f"Found {len(peak_indices)} peaks.")

    # Add peak heights to properties (find_peaks doesn't add it by default)
    peak_properties['peak_heights'] = force_curve[peak_indices]


    return peak_indices, peak_properties

def analyze_unfolding_pathway(
    force_curve: np.ndarray,
    peak_indices: np.ndarray,
    peak_properties: Dict[str, np.ndarray]
    # Add parameters related to extension mapping if needed for spacing analysis
) -> Dict[str, Any]:
    """
    Analyzes the detected peaks to infer aspects of the unfolding pathway.

    Args:
        force_curve (np.ndarray): The original 1D numpy array of force values.
        peak_indices (np.ndarray): Indices of the detected peaks.
        peak_properties (Dict[str, np.ndarray]): Dictionary of peak properties from find_peaks.

    Returns:
        Dict[str, Any]: A dictionary containing inferred pathway characteristics.
                       Examples: number of peaks, peak forces, peak spacing, etc.
                       (Note: Peak spacing interpretation requires knowledge of the extension axis).
    """
    analysis_results = {}

    analysis_results['num_peaks'] = len(peak_indices)

    if analysis_results['num_peaks'] > 0:
        # Unfolding forces are the force values at the peak indices
        analysis_results['unfolding_forces'] = peak_properties.get('peak_heights', force_curve[peak_indices]).tolist()

        # Peak spacing (in data points)
        if len(peak_indices) > 1:
            analysis_results['peak_spacing_indices'] = np.diff(peak_indices).tolist()
            # To convert to physical extension spacing, multiply by the physical extension_step
            # analysis_results['peak_spacing_extension'] = np.diff(peak_indices) * physical_extension_step


        # Other properties from peak_properties can be included
        # analysis_results['peak_prominences'] = peak_properties.get('prominences', None)
        # analysis_results['peak_widths'] = peak_properties.get('widths', None)

    else:
        analysis_results['unfolding_forces'] = []
        analysis_results['peak_spacing_indices'] = []
        # analysis_results['peak_prominences'] = []
        # analysis_results['peak_widths'] = []


    return analysis_results


# Example Usage
if __name__ == "__main__":
    print("--- Testing mechanical_properties.py ---")

    # Create a dummy F-E curve with a few peaks
    fe_len = 500
    dummy_curve = np.zeros(fe_len)
    ext_points = np.arange(fe_len) # Dummy extension points (indices)

    # Simulate some stretching and unfolding events
    # Peak 1
    peak1_idx = 100
    peak1_force = 80
    dummy_curve[:peak1_idx] = np.linspace(0, peak1_force, peak1_idx)
    dummy_curve[peak1_idx:peak1_idx+50] = peak1_force - np.linspace(0, peak1_force*0.8, 50) # Drop

    # Peak 2
    peak2_idx = 200
    peak2_force = 120
    dummy_curve[peak1_idx+50:peak2_idx] = np.linspace(dummy_curve[peak1_idx+49], peak2_force, peak2_idx - (peak1_idx+50))
    dummy_curve[peak2_idx:peak2_idx+70] = peak2_force - np.linspace(0, peak2_force*0.9, 70) # Drop

    # Peak 3 (smaller)
    peak3_idx = 300
    peak3_force = 50
    dummy_curve[peak2_idx+70:peak3_idx] = np.linspace(dummy_curve[peak2_idx+69], peak3_force, peak3_idx - (peak2_idx+70))
    dummy_curve[peak3_idx:] = peak3_force - np.linspace(0, peak3_force*0.5, fe_len - peak3_idx) # Decay to end

    # Add some noise
    dummy_curve += np.random.randn(fe_len) * 8

    dummy_curve = np.load('../../scripts/checkpoints/DiffusionModelTrainer/true_curves.npy')[11].reshape(130)

    import matplotlib.pyplot as plt


    plt.figure(figsize=(10, 6))
    plt.plot(dummy_curve, label='Dummy F-E Curve')

    # --- Test calculate_unfolding_energy ---
    print("\n--- Testing calculate_unfolding_energy ---")
    # Assume a dummy extension step (e.g., 0.1 nm per data point)
    dummy_extension_step = 0.1
    energy = calculate_unfolding_energy(dummy_curve)
    print(f"Calculated unfolding energy (with step {dummy_extension_step}): {energy:.2f} (Force Units * {dummy_extension_step} nm)")



    # --- Test find_force_peaks ---
    print("\n--- Testing find_force_peaks ---")
    # Need to set reasonable parameters based on dummy curve characteristics
    # Look for peaks with minimum height > 30, minimum distance 50 data points, prominence > 15
    peak_indices, peak_properties = find_force_peaks(
        dummy_curve,
        height=0, # Minimum height
        distance=10, # Minimum horizontal distance
        prominence=0.05, # Minimum prominence
        smoothing_window=1 # Apply smoothing
        # smoothing_sigma=3 # Alternative smoothing
    )
    print(peak_properties.keys())

    print(f"Detected peak indices: {peak_indices}")
    print(f"Detected peak properties (heights): {peak_properties.get('peak_heights')}")

    # --- Test calculate_max_force ---
    print("\n--- Testing calculate_max_force ---")
    max_f = calculate_max_force(dummy_curve, peak_indices)
    print(f"Calculated maximum force: {max_f:.2f}")

    # Plot detected peaks
    plt.plot(peak_indices, dummy_curve[peak_indices], "x", label="Detected Peaks")
    plt.vlines(x=peak_indices, ymin=dummy_curve[peak_indices] - peak_properties.get('prominences', 0),
               ymax=dummy_curve[peak_indices], color="C1")
    if 'right_bases' in peak_properties:
        plt.hlines(y=peak_properties['peak_heights'], xmin=peak_properties['right_bases'],
               xmax=peak_properties['left_bases'], color="C1") # This might plot incorrectly, bases are indices


    # --- Test analyze_unfolding_pathway ---
    print("\n--- Testing analyze_unfolding_pathway ---")
    pathway_analysis = analyze_unfolding_pathway(dummy_curve, peak_indices, peak_properties)
    print("Unfolding pathway analysis results:")
    for key, value in pathway_analysis.items():
        print(f"  {key}: {value}")


    plt.xlabel("Extension (Data Points)") # Or scaled extension
    plt.ylabel("Force")
    plt.title("Dummy F-E Curve and Detected Peaks")
    plt.legend()
    plt.grid(True)
    plt.show()