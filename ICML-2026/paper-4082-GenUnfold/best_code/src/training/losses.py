# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NoisePredictionLoss(nn.Module):
    """
    Mean Squared Error (MSE) loss between the predicted noise and the true added noise.
    This is the standard loss for diffusion models that predict the noise.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MSE loss.

        Args:
            predicted_noise (torch.Tensor): The noise predicted by the denoising model.
                                            Shape (batch_size, fe_curve_length, fe_curve_channels).
            true_noise (torch.Tensor): The actual noise added during the forward diffusion process.
                                       Shape (batch_size, fe_curve_length, fe_curve_channels).

        Returns:
            torch.Tensor: The calculated MSE loss.
        """
        if predicted_noise.shape != true_noise.shape:
            logging.error(f"Noise prediction loss input shape mismatch: {predicted_noise.shape} vs {true_noise.shape}")
            raise ValueError("Predicted noise and true noise shapes must match.")

        return self.mse(predicted_noise, true_noise)

class MechanicalPropertyLoss(nn.Module):
    """
    Optional loss function that penalizes differences in extracted mechanical
    properties between the generated (or partially denoised) curve and the true curve.
    This requires differentiating through the mechanical property extraction.
    """
    def __init__(self, property_weights: Dict[str, float] = None):
        """
        Initializes the MechanicalPropertyLoss.

        Args:
            property_weights (Dict[str, float], optional): Dictionary mapping mechanical
                                                          property names ('unfolding_energy', 'max_force')
                                                          to their respective weights in the total loss.
                                                          Defaults to None (no properties used).
        """
        super().__init__()
        self.property_weights = property_weights if property_weights is not None else {}
        self.mse = nn.MSELoss()


    def extract_mechanical_properties(self, fe_curve: torch.Tensor, keys: List[str]) -> Dict[str, torch.Tensor]:
        """
        Placeholder function to extract differentiable mechanical properties
        from a single F-E curve tensor.

        Args:
            fe_curve (torch.Tensor): A single F-E curve tensor. Shape (fe_curve_length, fe_curve_channels).
                                     Assumes a single channel (force).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of extracted property tensors.
                                     Each value should be a scalar tensor.
        """
        # This is a CRITICAL placeholder. Implementing differentiable
        # extraction of properties like peak force or integration
        # from a tensor is complex and depends on your data representation
        # and definition of these properties.

        # Example: Simple (and likely non-differentiable directly in typical implementations)
        # Max Force: torch.max(fe_curve)
        # Unfolding Energy (simple integration): torch.sum(fe_curve) * extension_step # Need extension step
        properties = {}
        if 'mean' in keys:
            properties['mean'] = torch.mean(fe_curve, dim=2)
        if 'max' in keys:
            properties['max'] = torch.max(fe_curve, dim=2).values
        if 'var' in keys:
            properties['var'] = torch.var(fe_curve, dim=2)

        return properties


    def forward(self, generated_curve: torch.Tensor, true_curve: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mechanical property loss.

        Args:
            generated_curve (torch.Tensor): The generated or partially denoised F-E curve.
                                            Shape (batch_size, fe_curve_length, fe_curve_channels).
            true_curve (torch.Tensor): The true F-E curve (x_0).
                                       Shape (batch_size, fe_curve_length, fe_curve_channels).

        Returns:
            torch.Tensor: The calculated weighted sum of property losses.
        """
        if generated_curve.shape != true_curve.shape:
            logging.error(f"Mechanical property loss input shape mismatch: {generated_curve.shape} vs {true_curve.shape}")
            raise ValueError("Generated curve and true curve shapes must match.")

        total_loss = torch.tensor(0.0, device=generated_curve.device)

        gen_props = self.extract_mechanical_properties(generated_curve, self.property_weights.keys())
        true_props = self.extract_mechanical_properties(true_curve, self.property_weights.keys())  # Requires true_curve to be x_0

        for prop_name, weight in self.property_weights.items():
            if prop_name in gen_props and prop_name in true_props:
                gen_val = gen_props[prop_name]
                true_val = true_props[prop_name]

                # Ensure they are tensors and calculate MSE
                if isinstance(gen_val, torch.Tensor) and isinstance(true_val, torch.Tensor):
                    prop_loss = self.mse(gen_val, true_val)
                    total_loss += weight * prop_loss
                else:
                    logging.warning(
                        f"Property '{prop_name}' extraction did not return tensors. ")
            else:
                logging.warning(f"Property '{prop_name}' not found in extracted properties.")

        # Average loss over the batch
        num_samples = true_curve.shape[0]
        if num_samples > 0:
            total_loss /= num_samples

        return total_loss

class GeneratedCurveMatchingLoss(nn.Module):
    """
    Optional loss function that penalizes differences directly between the
    final generated curve (x_0_hat) and the true curve (x_0).
    Could use MSE, L1, or other distance metrics.
    """
    def __init__(self, loss_type: str = 'mse'):
        """
        Initializes the GeneratedCurveMatchingLoss.

        Args:
            loss_type (str): Type of loss to use ('mse', 'l1'). Defaults to 'mse'.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose from 'mse', 'l1'.")

        logging.warning("GeneratedCurveMatchingLoss is typically applied to the final output (x_0_hat).")
        logging.warning("Applying it directly during the noise prediction training requires predicting x_0 or a similar quantity.")
        logging.warning("In a standard noise-predicting diffusion model, the primary loss is MSE on noise.")


    def forward(self, predicted_x0: torch.Tensor, true_x0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the direct curve matching loss.

        Args:
            predicted_x0 (torch.Tensor): The predicted denoised data (x_0_hat).
                                        Shape (batch_size, fe_curve_length, fe_curve_channels).
                                        This might need to be derived from the noise prediction.
            true_x0 (torch.Tensor): The actual true data (x_0).
                                    Shape (batch_size, fe_curve_length, fe_curve_channels).

        Returns:
            torch.Tensor: The calculated curve matching loss.
        """
        if predicted_x0.shape != true_x0.shape:
            logging.error(f"Curve matching loss input shape mismatch: {predicted_x0.shape} vs {true_x0.shape}")
            raise ValueError("Predicted x0 and true x0 shapes must match.")

        return self.criterion(predicted_x0, true_x0)


class GeneratedPeakMatchingLoss(nn.Module):
    """
    Optional loss function that penalizes differences directly between the
    force peak in final generated curve (x_0_hat) and that in the true curve (x_0).
    Could use MSE, L1, or other distance metrics.
    """
    def __init__(self, loss_type: str = 'mse'):
        """
        Initializes the GeneratedCurveMatchingLoss.

        Args:
            loss_type (str): Type of loss to use ('mse', 'l1'). Defaults to 'mse'.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose from 'mse', 'l1'.")

    def _get_peak_mask(self, curve: torch.Tensor, top_n: int = 4) -> torch.Tensor:
        """
        Generates a boolean mask around the top N peaks of a curve.
        """
        N, C, L = curve.shape

        # Find the N largest values in the curve
        # .topk() is a better way to get the top values and indices
        top_values, top_indices = curve.abs().topk(top_n, dim=-1, largest=True, sorted=False)  # (N, 1, 4)

        # Create a zero mask
        mask = torch.zeros_like(curve, dtype=torch.bool, device=curve.device)

        # For each peak index, set a window of size peak_window_size to True
        for i in range(N):
            for j in range(top_n):
                peak_idx = top_indices[i, 0, j].item()
                start_idx = max(0, peak_idx - self.peak_window_size // 2)
                end_idx = min(L, peak_idx + self.peak_window_size // 2 + 1)
                mask[i, 0, start_idx:end_idx] = True
        return mask


    def forward(self, predicted_x0: torch.Tensor, true_x0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the direct curve matching loss.

        Args:
            predicted_x0 (torch.Tensor): The predicted denoised data (x_0_hat).
                                        Shape (batch_size, fe_curve_length, fe_curve_channels).
                                        This might need to be derived from the noise prediction.
            true_x0 (torch.Tensor): The actual true data (x_0).
                                    Shape (batch_size, fe_curve_length, fe_curve_channels).

        Returns:
            torch.Tensor: The calculated curve matching loss.
        """
        if predicted_x0.shape != true_x0.shape:
            logging.error(f"Curve matching loss input shape mismatch: {predicted_x0.shape} vs {true_x0.shape}")
            raise ValueError("Predicted x0 and true x0 shapes must match.")

        with torch.no_grad():  # This part is not in the computation graph
            true_peak_mask = self._get_peak_mask(true_x0, top_n=4)

        predicted_peak_region = predicted_x0[true_peak_mask]
        true_peak_region = true_x0[true_peak_mask]

        return self.criterion(predicted_peak_region, true_peak_region)


# Example Usage
if __name__ == "__main__":
    print("--- Testing losses.py ---")

    batch_size = 4
    fe_curve_length = 200
    fe_curve_channels = 1

    # Dummy tensors
    predicted_noise = torch.randn(batch_size, fe_curve_length, fe_curve_channels, requires_grad=True)
    true_noise = torch.randn(batch_size, fe_curve_length, fe_curve_channels)
    dummy_x0 = torch.randn(batch_size, fe_curve_length, fe_curve_channels) # Dummy clean data
    # Simulate a partially denoised curve (e.g., x_t)
    dummy_xt = dummy_x0 + torch.randn_like(dummy_x0) * 0.5 # Add some noise

    # Test NoisePredictionLoss
    print("\n--- Testing NoisePredictionLoss ---")
    noise_loss_fn = NoisePredictionLoss()
    noise_loss = noise_loss_fn(predicted_noise, true_noise)
    print(f"Noise prediction loss: {noise_loss.item():.4f}")
    try:
        noise_loss.backward()
        print("Backward pass successful for NoisePredictionLoss.")
        print("Predicted noise gradients:", predicted_noise.grad.norm().item())
    except Exception as e:
        print(f"Backward pass failed: {e}")


    # Test MechanicalPropertyLoss (Note: Requires differentiable extraction)
    print("\n--- Testing MechanicalPropertyLoss ---")
    # Note: The dummy `extract_mechanical_properties` is likely not fully differentiable.
    # This test is conceptual.
    property_weights = {'unfolding_energy': 0.5, 'max_force': 0.5}
    mech_prop_loss_fn = MechanicalPropertyLoss(property_weights=property_weights)
    # In a real scenario, you'd apply this to the final generated curve or predicted x_0
    # For this test, let's compare a noisy version against the original x_0
    # Note: Gradients might not flow correctly due to placeholder extraction
    try:
         # Need requires_grad=True for the input curve if calculating properties
         dummy_xt_requires_grad = dummy_xt.clone().detach().requires_grad_(True)
         mech_prop_loss = mech_prop_loss_fn(dummy_xt_requires_grad, dummy_x0)
         print(f"Mechanical property loss (dummy): {mech_prop_loss.item():.4f}")
         mech_prop_loss.backward()
         print("Backward pass attempted for MechanicalPropertyLoss (check grad values).")
         print("Dummy x_t gradients:", dummy_xt_requires_grad.grad.norm().item()) # May be zero if extraction is not differentiable
    except Exception as e:
        print(f"An error occurred during MechanicalPropertyLoss test: {e}")


    # Test GeneratedCurveMatchingLoss
    print("\n--- Testing GeneratedCurveMatchingLoss ---")
    curve_match_loss_fn_mse = GeneratedCurveMatchingLoss(loss_type='mse')
    curve_match_loss_fn_l1 = GeneratedCurveMatchingLoss(loss_type='l1')

    # In training, you'd likely predict x_0_hat from noise and compare to x_0
    # Let's use dummy_xt as predicted_x0 for the test (conceptually)
    dummy_predicted_x0 = dummy_xt.clone().detach().requires_grad_(True)

    curve_match_loss_mse = curve_match_loss_fn_mse(dummy_predicted_x0, dummy_x0)
    print(f"Generated curve matching loss (MSE): {curve_match_loss_mse.item():.4f}")
    try:
        curve_match_loss_mse.backward()
        print("Backward pass successful for GeneratedCurveMatchingLoss (MSE).")
        print("Dummy predicted_x0 gradients:", dummy_predicted_x0.grad.norm().item())
    except Exception as e:
        print(f"Backward pass failed: {e}")

    # Need new tensor for L1 backward test
    dummy_predicted_x0_l1 = dummy_xt.clone().detach().requires_grad_(True)
    curve_match_loss_l1 = curve_match_loss_fn_l1(dummy_predicted_x0_l1, dummy_x0)
    print(f"Generated curve matching loss (L1): {curve_match_loss_l1.item():.4f}")
    try:
         curve_match_loss_l1.backward()
         print("Backward pass successful for GeneratedCurveMatchingLoss (L1).")
         print("Dummy predicted_x0 (L1) gradients:", dummy_predicted_x0_l1.grad.norm().item())
    except Exception as e:
         print(f"Backward pass failed: {e}")