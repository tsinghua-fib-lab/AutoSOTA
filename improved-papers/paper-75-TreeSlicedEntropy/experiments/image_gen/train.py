import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Keep for legend
import numpy as np
import os
import math
import random
from datetime import datetime
import time

# --- Add POT import ---
import ot # Python Optimal Transport library

from model import Autoencoder, Classifier
from tsw import TSW, PartialTSW, generate_trees_frames
from baselines import sliced_wasserstein, sopt, spot, pawl, unbalanced_sliced_ot, sliced_unbalanced_ot

# --- Model Loading Functions (Unchanged) ---
def load_autoencoder(path, latent_dim, device):
    model = Autoencoder(latent_dim=latent_dim).to(device)
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded Autoencoder weights from {path}")
        except Exception as e:
            print(f"Error loading Autoencoder weights from {path}: {e}")
            print("Using initialized Autoencoder for pre-trained AE role.")
    else:
        print(f"Pre-trained AE path '{path}' not found or not provided. Using initialized Autoencoder for pre-trained AE role.")
    model.eval()
    return model

def load_classifier(path, num_classes, device):
    model = Classifier(num_classes=num_classes).to(device)
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded Classifier weights from {path}")
        except Exception as e:
            print(f"Error loading Classifier weights from {path}: {e}")
            print("Using initialized Classifier for pre-trained Classifier role.")
    else:
        print(f"Pre-trained Classifier path '{path}' not found or not provided. Using initialized Classifier for pre-trained Classifier role.")
    model.eval()
    return model

def load_model_state(model, path, device): # This is for models trained BY THIS SCRIPT
    if path and os.path.exists(path):
        print(f"Attempting to load model state from {path}...")
        try:
            model.to(device)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print("Model state loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model state from {path}: {e}")
            try:
                print("Trying fallback: loading state_dict to CPU first...")
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.to(device); model.eval()
                print("Fallback loading of model state successful.")
                return True
            except Exception as e2:
                 print(f"Fallback loading of model state failed: {e2}")
                 return False
    else:
        print(f"Info: Model state file not found at '{path}' (or path not provided).")
        return False

# --- Dataset Preparation (Unchanged) ---
class FilteredMNIST(Dataset):
    def __init__(self, mnist_dataset, target_digits, percentages, seed=42):
        self.mnist_dataset = mnist_dataset; self.target_digits = target_digits
        self.percentages = percentages; self.seed = seed
        if not (len(target_digits) == len(percentages)): raise ValueError("target_digits and percentages must have the same length.")
        if not (abs(sum(percentages) - 100.0) < 1e-6):
            if not (len(percentages) == 1 and percentages[0] == 100.0): raise ValueError(f"Percentages must sum to 100. Got: {sum(percentages)}")
        rng = random.Random(seed); indices_by_digit = {digit: [] for digit in range(10)}
        try:
            dataset_targets = mnist_dataset.targets
            if isinstance(dataset_targets, torch.Tensor): dataset_targets = dataset_targets.tolist()
        except AttributeError: dataset_targets = [label for _, label in mnist_dataset]
        for i in range(len(mnist_dataset)):
            label = dataset_targets[i]
            if isinstance(label, torch.Tensor): label = label.item()
            indices_by_digit[label].append(i)
        primary_digit_idx = np.argmax(percentages) if percentages else -1
        estimated_total_size = 0
        if primary_digit_idx != -1 and percentages[primary_digit_idx] > 0 :
            primary_digit = target_digits[primary_digit_idx]; available_primary = len(indices_by_digit[primary_digit])
            estimated_total_size = int(available_primary / (percentages[primary_digit_idx] / 100.0))
        final_indices = []
        for digit, percentage in zip(target_digits, percentages):
            num_samples_needed = int(round(estimated_total_size * (percentage / 100.0)))
            available_indices = indices_by_digit[digit]; rng.shuffle(available_indices)
            num_samples_to_take = min(len(available_indices), num_samples_needed)
            if len(available_indices) < num_samples_needed and len(available_indices) > 0 :
                print(f"Warning (FilteredMNIST): Needed {num_samples_needed} for digit {digit}, only {len(available_indices)} available. Taking {num_samples_to_take}.")
            selected_indices = available_indices[:num_samples_to_take]; final_indices.extend(selected_indices)
        rng.shuffle(final_indices); self.subset = Subset(mnist_dataset, final_indices)
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx): return self.subset[idx]


# --- Function to construct target latent dataset (Unchanged) ---
def construct_target_latent_dataset(autoencoder, classifier, num_target_samples, percent_0_target,
                                    latent_dim, device, batch_size_sampling=512, max_sample_multiplier=10):
    print(f"Constructing synthetic target latent dataset of size {num_target_samples}...")
    print(f"Targeting {percent_0_target:.1f}% for digit 0, {100-percent_0_target:.1f}% for digit 1.")
    autoencoder.eval(); classifier.eval()
    collected_latents_0, collected_labels_0 = [], []
    collected_latents_1, collected_labels_1 = [], []
    num_needed_0 = int(round(num_target_samples * (percent_0_target / 100.0)))
    num_needed_1 = num_target_samples - num_needed_0
    print(f"Need {num_needed_0} latents for digit 0, {num_needed_1} latents for digit 1.")
    max_candidates_to_screen = num_target_samples * max_sample_multiplier; candidates_screened = 0
    with torch.no_grad():
        while (len(collected_latents_0) < num_needed_0 or len(collected_latents_1) < num_needed_1) and \
              candidates_screened < max_candidates_to_screen:
            z_candidates = (2 * torch.rand(batch_size_sampling, latent_dim) - 1).to(device) # Ensure latents are in [-1,1] if AE Tanh
            candidates_screened += batch_size_sampling
            decoded_images = autoencoder.decode(z_candidates)
            predictions = classifier(decoded_images).argmax(dim=1)
            for i in range(z_candidates.size(0)):
                pred_label = predictions[i].item()
                if pred_label == 0 and len(collected_latents_0) < num_needed_0:
                    collected_latents_0.append(z_candidates[i].unsqueeze(0))
                    collected_labels_0.append(torch.tensor([0], dtype=torch.long, device=device))
                elif pred_label == 1 and len(collected_latents_1) < num_needed_1:
                    collected_latents_1.append(z_candidates[i].unsqueeze(0))
                    collected_labels_1.append(torch.tensor([1], dtype=torch.long, device=device))
            if candidates_screened % (batch_size_sampling * 20) == 0:
                print(f"  Screened {candidates_screened}/{max_candidates_to_screen}. Coll 0s: {len(collected_latents_0)}/{num_needed_0}, Coll 1s: {len(collected_latents_1)}/{num_needed_1}")
            if len(collected_latents_0) >= num_needed_0 and len(collected_latents_1) >= num_needed_1:
                print("Sufficient samples collected for both classes."); break
    print(f"Finished screening. Total candidates screened: {candidates_screened}")
    if len(collected_latents_0) < num_needed_0: print(f"Warning: Collected only {len(collected_latents_0)}/{num_needed_0} for digit 0.")
    if len(collected_latents_1) < num_needed_1: print(f"Warning: Collected only {len(collected_latents_1)}/{num_needed_1} for digit 1.")
    final_latents_0 = torch.cat(collected_latents_0[:num_needed_0], dim=0) if collected_latents_0 else torch.empty(0, latent_dim, device=device)
    final_labels_0 = torch.cat(collected_labels_0[:num_needed_0], dim=0) if collected_labels_0 else torch.empty(0, dtype=torch.long, device=device)
    final_latents_1 = torch.cat(collected_latents_1[:num_needed_1], dim=0) if collected_latents_1 else torch.empty(0, latent_dim, device=device)
    final_labels_1 = torch.cat(collected_labels_1[:num_needed_1], dim=0) if collected_labels_1 else torch.empty(0, dtype=torch.long, device=device)
    if final_latents_0.numel() == 0 and final_latents_1.numel() == 0:
        print("Error: No target latents collected."); return torch.empty(0, latent_dim, device=device), torch.empty(0, dtype=torch.long, device=device)
    target_latents_combined = torch.cat((final_latents_0, final_latents_1), dim=0)
    target_labels_combined = torch.cat((final_labels_0, final_labels_1), dim=0)
    perm = torch.randperm(target_latents_combined.size(0))
    target_latents_shuffled = target_latents_combined[perm]; target_labels_shuffled = target_labels_combined[perm]
    print(f"Constructed target latent dataset with {target_latents_shuffled.size(0)} samples.")
    print(f"  Digit 0s: {final_latents_0.size(0)}, Digit 1s: {final_latents_1.size(0)}")
    return target_latents_shuffled.to(device), target_labels_shuffled.to(device)

# --- Visualization Helpers (Unchanged from your provided snippet) ---
def plot_latent_space(latents, labels, title="Latent Space", filename="latent_space.pdf"):
    plt.figure(figsize=(8, 8)); unique_labels = sorted(np.unique(labels))
    num_distinct_labels = len(unique_labels)
    if num_distinct_labels == 2 or all(l in [0, 1] for l in unique_labels):
        color_map = {0: '#0072B2', 1: '#D55E00'}
        colors_list = [color_map[l] for l in unique_labels]
    elif num_distinct_labels == 1 and unique_labels[0] == -1:
        colors_list = ['blue']
    else:
        colors_list = plt.cm.viridis(np.linspace(0, 1, max(1,num_distinct_labels)))
    ax = plt.gca()
    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        current_color = colors_list[i]
        if label_val == -1 and num_distinct_labels == 1 :
            ax.scatter(latents[idx, 0], latents[idx, 1], color=current_color, alpha=0.6, s=10)
        else:
            ax.scatter(latents[idx, 0], latents[idx, 1], color=current_color, alpha=0.6, s=10)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    plt.grid(True); 
    save_kwargs = dict(bbox_inches="tight", pad_inches=0)
    
    ext = filename.lower().split(".")[-1]
    if ext == "png":
        save_kwargs["pil_kwargs"] = {"compress_level": 9}
    elif ext in ("jpg", "jpeg"):
        save_kwargs["pil_kwargs"] = {"quality": jpg_quality,
                                 "optimize": True,
                                 "progressive": True}
    elif ext == "pdf":
        mpl.rcParams["pdf.compression"] = 9

    plt.savefig(filename, **save_kwargs)
    plt.close(); print(f"Saved latent space plot to {filename}")

def plot_generated_images(images, labels=None, num_images=25, title="Generated Images", filename="generated_images.png"):
    if images.shape[0] == 0: print(f"No images to plot for {filename}."); return
    if images.shape[0] < num_images: num_images = images.shape[0]
    cols = int(np.sqrt(num_images)); rows = (num_images + cols - 1) // cols if cols > 0 else 1
    if rows == 0 : rows = 1
    fig = plt.figure(figsize=(max(1, cols) * 1.0, max(1, rows) * 1.0));
    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy() if labels is not None else None
    for i in range(num_images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images_np[i].squeeze(), cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0);
    save_kwargs = dict(bbox_inches="tight", pad_inches=0)

    ext = filename.lower().split(".")[-1]
    if ext == "png":
        save_kwargs["pil_kwargs"] = {"compress_level": 9}
    elif ext in ("jpg", "jpeg"):
        save_kwargs["pil_kwargs"] = {"quality": jpg_quality,
                                 "optimize": True,
                                 "progressive": True}
    elif ext == "pdf":
        mpl.rcParams["pdf.compression"] = 9
    
    plt.savefig(filename, **save_kwargs)
    plt.close(fig)
    print(f"Saved generated images plot to {filename}")

def save_standalone_legend(filename="legend_digits.pdf"):
    # Initial figure size can be relatively small, as bbox_inches='tight' will handle final dimensions.
    # It needs to be large enough to properly render the legend before cropping.
    fig_legend = plt.figure(figsize=(2.5, 0.6)) # Adjusted for typical legend size
    ax_legend = fig_legend.add_subplot(111)

    color_map_legend = {0: '#0072B2', 1: '#D55E00'} # Blue and Orange/Brown
    labels_legend = {0: "MNIST digit 0", 1: "MNIST digit 1"}
    from matplotlib.lines import Line2D # Import Line2D for dot markers
    legend_elements = []
    for label_val in sorted(color_map_legend.keys()):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', # Invisible line for the marker
                                      label=labels_legend[label_val],
                                      markerfacecolor=color_map_legend[label_val],
                                      markeredgecolor='white', # Make marker edge white or match dot color
                                      markeredgewidth=1.5,    # Ensure marker edge is visible if needed
                                      markersize=10, linestyle='None'))

    # Create the legend
    legend = fig_legend.legend(handles=legend_elements,
                               loc='center',       # Position in the center of the temporary axes
                               frameon=True,       # Draw the frame/border
                               fancybox=True,      # Use a box with curved corners
                               ncol=len(legend_elements), # Arrange items in a single line
                               edgecolor='darkgray', # Color of the legend border
                               facecolor='white'   # Background color *inside* the legend border
                               )
    # Ensure the legend box's own background (facecolor) is fully opaque.
    legend.get_frame().set_alpha(1.0)

    # Turn off the axes of the subplot, we only want the legend.
    ax_legend.axis('off')

    fig_legend.patch.set_alpha(0.0)
    ax_legend.patch.set_alpha(0.0) # Axes background also transparent

    fig_legend.savefig(filename,
                       bbox_inches='tight',
                       pad_inches=0.01,
                       transparent=True)
    plt.close(fig_legend)
    print(f"Saved standalone legend, tightly cropped, to {filename}")


# --- UOT/GAN Training Loop (MODIFIED) ---
def train_generator_on_latents(args, generator, optimizer_g, target_latents,
                               autoencoder, classifier,
                               device):
    print(f"\n--- 4. Train Generator using {args.loss_type.upper()} loss on Target Latents ---")
    twd_calculator = None
    if args.loss_type == 'twd':
        print("Initializing TWD calculator for Generator training.")
        # Use PartialTSW for linear (supports partial transport) or TSW for other ftypes
        if args.twd_ftype == 'linear':
            print("Using PartialTSW (supports unbalanced masses)")
            twd_calculator = PartialTSW(ntrees=args.twd_ntrees, nlines=args.twd_nlines, p=args.sw_p,
                                        delta=args.twd_delta, mass_division=args.twd_mass_division,
                                        device=device)
        else:
            print(f"Using TSW with ftype={args.twd_ftype} (unbalanced masses may not work)")
            twd_calculator = TSW(ntrees=args.twd_ntrees, nlines=args.twd_nlines, p=args.sw_p,
                                 delta=args.twd_delta, mass_division=args.twd_mass_division,
                                 ftype=args.twd_ftype, d=args.latent_dim, device=device)

    for epoch in range(args.num_epochs):
        generator.train(); epoch_g_loss = 0.0; num_batches_processed = 0
        if target_latents.size(0) == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}: Target latents dataset is empty. Skipping training."); continue
        permuted_indices = torch.randperm(target_latents.size(0))
        target_latents_shuffled = target_latents[permuted_indices]
        for batch_idx, i in enumerate(range(0, target_latents_shuffled.size(0), args.batch_size)):
            optimizer_g.zero_grad()
            real_batch_latents = target_latents_shuffled[i : i + args.batch_size].to(device) # Ensure on device
            current_batch_size = real_batch_latents.size(0)
            if current_batch_size == 0: continue
            z_noise = torch.randn(current_batch_size, args.noise_dim, device=device)
            fake_latents = generator(z_noise)

            if args.loss_type == 'twd' and twd_calculator:
                # ... (your TWD code unchanged)
                theta, intercept = generate_trees_frames(ntrees=args.twd_ntrees, nlines=args.twd_nlines, d=args.latent_dim, device=device)
                total_mass_X_tensor = torch.tensor(1.0, device=device); total_mass_Y_tensor = torch.tensor(1.0, device=device)
                if args.twd_unbalanced:
                    data_loader_len = (target_latents_shuffled.size(0) + args.batch_size -1) // args.batch_size
                    current_step_float = float(epoch*data_loader_len + batch_idx); total_steps_float = float(args.num_epochs*data_loader_len) if args.num_epochs*data_loader_len > 0 else 1.0
                    progress = max(0.0, min(1.0, current_step_float / total_steps_float))
                    min_m, max_m = args.min_mass_generated, args.max_mass_generated; scheduled_mass_y = max_m
                    if args.twd_unbalanced_scheduler == 'constant': scheduled_mass_y = max_m
                    elif args.twd_unbalanced_scheduler == 'linear_increasing': scheduled_mass_y = min_m + progress * (max_m - min_m)
                    elif args.twd_unbalanced_scheduler == 'linear_decreasing': scheduled_mass_y = max_m - progress * (max_m - min_m)
                    total_mass_Y_tensor = torch.tensor(scheduled_mass_y, device=device)
                g_loss = twd_calculator(real_batch_latents, fake_latents, theta, intercept, total_mass_X=total_mass_X_tensor, total_mass_Y=total_mass_Y_tensor)
            elif args.loss_type == 'sw':
                g_loss = sliced_wasserstein(real_batch_latents, fake_latents, num_projections=args.sw_projections, p=args.sw_p, device=device)
            elif args.loss_type == 'usot':
                mass_X = torch.ones(real_batch_latents.size(0), device=device) / real_batch_latents.size(0)
                mass_Y = torch.ones(fake_latents.size(0), device=device) / fake_latents.size(0)
                g_loss, _, _, _, _, _ = unbalanced_sliced_ot(mass_X, mass_Y, real_batch_latents, fake_latents, num_projections=args.sw_projections, p=args.sw_p,
                                                                rho1=args.rho1, rho2=args.rho2, niter=10, mode='icdf')
            elif args.loss_type == 'suot':
                mass_X = torch.ones(real_batch_latents.size(0), device=device)
                mass_Y = torch.ones(fake_latents.size(0), device=device)
                g_loss, _, _, _, _, _ = sliced_unbalanced_ot(mass_X, mass_Y, real_batch_latents, fake_latents, num_projections=args.sw_projections, p=args.sw_p,
                                                                rho1=args.rho1, rho2=args.rho2, niter=10, mode='icdf')
            elif args.loss_type == 'sopt':
                g_loss = sopt(real_batch_latents.cpu(), fake_latents.cpu(), n_proj=args.sw_projections, reg=args.sopt_reg) # SOPT/SPOT/PAWL might need CPU
            elif args.loss_type == 'spot':
                g_loss = spot(real_batch_latents.cpu(), fake_latents[:args.spot_k].cpu(), n_proj=args.sw_projections)
            elif args.loss_type == 'pawl':
                g_loss = pawl(real_batch_latents.cpu(), fake_latents.cpu(), n_proj=args.sw_projections, k=args.pawl_k)
            # --- ADD POT UNBALANCED SINKHORN LOSS ---
            elif args.loss_type == 'pot':
                # Ensure tensors are on the correct device and dtype for POT
                # a and b are marginals (weights for each sample). Uniform for now.
                a = torch.ones(real_batch_latents.size(0), device=device, dtype=real_batch_latents.dtype) / real_batch_latents.size(0)
                b = torch.ones(fake_latents.size(0), device=device, dtype=fake_latents.dtype) / fake_latents.size(0)

                # M is the cost matrix. Using squared Euclidean distance (L2^2).
                # ot.dist will use the PyTorch backend.
                # The power p for the cost metric. If args.sw_p = 2, this is squared Euclidean.
                # If args.sw_p = 1, this is L1 distance.
                # M = ot.dist(real_batch_latents, fake_latents, metric='euclidean').pow(args.sw_p) # M_ij = ||x_i - y_j||_2^p
                # Common choice: Squared Euclidean distance, equivalent to metric='sqeuclidean' if p=2
                if args.pot_cost_metric_p == 2:
                    M = ot.dist(real_batch_latents, fake_latents, metric='sqeuclidean')
                elif args.pot_cost_metric_p == 1:
                     M = ot.dist(real_batch_latents, fake_latents, metric='euclidean') # L1
                else: # Generic p-norm to the power p
                    M = torch.cdist(real_batch_latents, fake_latents, p=args.pot_cost_metric_p).pow(args.pot_cost_metric_p)


                # Call the POT function
                # entropic_kl_uot_ti is the transport cost, which will be our loss
                _, log_uot_ti = ot.unbalanced.sinkhorn_unbalanced2(
                    a, b, M,
                    reg=args.pot_reg,
                    reg_m=args.pot_reg_m_kl, # reg_m_kl in your snippet refers to reg_m here
                    method="sinkhorn_translation_invariant",
                    numItermax=args.pot_num_iter_max,
                    stopThr=args.pot_stop_thr,
                    log=True, # To get the log dictionary
                    # reg_type="kl" # This seems to be default for sinkhorn_unbalanced if reg_m is for KL
                )
                g_loss = log_uot_ti['cost']
                # Check if g_loss contains NaN or Inf
                if torch.isnan(g_loss) or torch.isinf(g_loss):
                    print(f"Warning: NaN or Inf detected in POT loss at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                    print(f"a shape: {a.shape}, b shape: {b.shape}, M shape: {M.shape}")
                    print(f"M min: {M.min().item()}, M max: {M.max().item()}, M mean: {M.mean().item()}")
                    print(f"reg: {args.pot_reg}, reg_m: {args.pot_reg_m_kl}")
                    # Potentially skip this batch or handle error appropriately
                    continue # Skip backward pass and optimizer step for this batch

            # --- END POT UNBALANCED SINKHORN LOSS ---
            else:
                raise ValueError(f"Unsupported loss type: {args.loss_type}")

            g_loss.backward()
            optimizer_g.step()
            epoch_g_loss += g_loss.item()
            num_batches_processed += 1

        avg_g_loss = epoch_g_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Generator Loss ({args.loss_type.upper()}): {avg_g_loss:.6f}")

        if (epoch + 1) % args.vis_every == 0 and args.latent_dim >=2:
             generator.eval(); autoencoder.eval(); classifier.eval()
             with torch.no_grad():
                 num_points_for_epoch_viz = min(5000, target_latents.size(0) if target_latents.numel() > 0 else 5000, args.num_final_samples_eval)
                 if num_points_for_epoch_viz == 0: print(f"Skipping epoch {epoch+1} visualization as num_points_for_epoch_viz is 0."); generator.train(); continue
                 z_vis_noise = torch.randn(num_points_for_epoch_viz, args.noise_dim, device=device)
                 fake_latents_tensor_vis = generator(z_vis_noise)
                 decoded_images_epoch_vis = autoencoder.decode(fake_latents_tensor_vis)
                 predictions_epoch_vis = classifier(decoded_images_epoch_vis).argmax(dim=1)
                 plot_latent_space(fake_latents_tensor_vis.cpu().numpy(), predictions_epoch_vis.cpu().numpy(),
                                   title=f"G's Latent Output (Epoch {epoch+1}, Colored by Predicted Class)",
                                   filename=os.path.join(args.output_dir, f"g_generated_latent_epoch_{epoch+1}_classified.pdf"))
             generator.train()

# --- Main Function (Unchanged parts omitted for brevity, show only around arg parsing) ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Set POT backend at the start if it's the chosen loss type ---
    # This is another place you could set it, or do it within the training loop as shown above.
    # if args.loss_type == 'pot_unbalanced':
    #     ot.backend.set_backend("pytorch")
    #     print("POT backend set to PyTorch globally.")


    print("\n--- 1. Loading Pre-trained AE and Classifier ---")
    autoencoder = load_autoencoder(args.ae_ckp_path, args.latent_dim, device)
    classifier = load_classifier(args.classifier_ckp_path, 2, device)

    transform_mnist = transforms.Compose([transforms.ToTensor()])
    mnist_full_train = MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    if not (0 <= args.percent_0 <= 100): raise ValueError("percent_0 must be between 0 and 100.")

    print("\n--- 2. Preparing Filtered MNIST for Initial AE Latent Space Visualization ---")
    filtered_mnist_for_viz = FilteredMNIST(mnist_full_train, [0, 1], [args.percent_0, 100.0 - args.percent_0], seed=args.seed)
    if len(filtered_mnist_for_viz) > 0:
        viz_dataloader = DataLoader(filtered_mnist_for_viz, batch_size=args.batch_size, shuffle=False)
        print("\n--- 3. Visualizing Initial Latent Space of REAL MNIST Digits (Encoded by Pre-trained AE) ---")
        all_latents_list_real, all_labels_list_real = [], []
        with torch.no_grad():
            max_viz_points_initial = 5000
            for i, (data, labels_batch) in enumerate(viz_dataloader):
                if i * args.batch_size >= max_viz_points_initial: break
                data = data.to(device); latents = autoencoder.encode(data)
                all_latents_list_real.append(latents.cpu()); all_labels_list_real.append(labels_batch.cpu())
        if all_latents_list_real:
            all_latents_np_real = torch.cat(all_latents_list_real, dim=0).numpy()
            all_labels_np_real = torch.cat(all_labels_list_real, dim=0).numpy()
            plot_latent_space(all_latents_np_real, all_labels_np_real,
                              title=f"Pre-trained AE Latent Space of Real Digits ({args.percent_0}% 0)",
                              filename=os.path.join(args.output_dir, "initial_real_latent_space.pdf"))
            save_standalone_legend(os.path.join(args.output_dir, "legend_digits.pdf")) # Save legend
        else: print("No real latents for initial AE viz.")
    else: print("Skipping initial AE latent space viz (FilteredMNIST for viz empty).")


    print("\n--- 3.5 Constructing Synthetic Target Latent Dataset for Generator Training ---")
    target_latents, target_labels = construct_target_latent_dataset(
        autoencoder, classifier, num_target_samples=args.target_latent_dataset_size,
        percent_0_target=args.percent_0, latent_dim=args.latent_dim, device=device,
        batch_size_sampling=args.batch_size, max_sample_multiplier=args.max_sampling_multiplier
    )
    if target_latents.size(0) < args.target_latent_dataset_size * 0.01 or target_latents.size(0) == 0:
        print(f"Error: Insufficient target latents ({target_latents.size(0)}). Exiting."); return

    if target_latents.size(0) > 0 and args.latent_dim >= 2:
        print("\n--- Visualizing Synthetically Constructed Target Latent Space ---")
        num_plot_points_synthetic = min(5000, target_latents.size(0))
        if target_labels.size(0) == target_latents.size(0) and target_labels.size(0) > 0:
            plot_indices_synthetic = torch.randperm(target_latents.size(0))[:num_plot_points_synthetic]
            plot_latent_space(
                target_latents[plot_indices_synthetic].cpu().numpy(),
                target_labels[plot_indices_synthetic].cpu().numpy(),
                title=f"Synthetic Target Latent Space ({args.percent_0}% Target 0)",
                filename=os.path.join(args.output_dir, "synthetic_target_latent_space.pdf")
            )
        else: print("Warning: Not enough labels or latents to plot synthetic target space, or mismatch.")

        if target_latents.size(0) > 0:
            print("\n--- Visualizing Images corresponding to Synthetic Target Latent Space ---")
            num_synthetic_images_to_plot = min(getattr(args, 'num_vis_final', 25), target_latents.size(0))
            if num_synthetic_images_to_plot > 0:
                latents_to_decode_synthetic = target_latents[:num_synthetic_images_to_plot].to(device)
                labels_for_synthetic_images = target_labels[:num_synthetic_images_to_plot].cpu()
                decoded_synthetic_images = None
                with torch.no_grad():
                    autoencoder.eval()
                    decoded_synthetic_images_batch = autoencoder.decode(latents_to_decode_synthetic)
                    decoded_synthetic_images = decoded_synthetic_images_batch.cpu()
                if decoded_synthetic_images is not None and decoded_synthetic_images.size(0) > 0:
                    plot_generated_images(
                        decoded_synthetic_images,
                        labels_for_synthetic_images,
                        num_images=decoded_synthetic_images.shape[0],
                        filename=os.path.join(args.output_dir, "synthetic_target_images.pdf")
                    )
                else: print("No synthetic images were decoded for visualization.")
            else: print("No synthetic images selected for plotting based on num_synthetic_images_to_plot.")

    generator = nn.Sequential(
        nn.Linear(args.noise_dim, args.noise_dim * 2), nn.BatchNorm1d(args.noise_dim * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(args.noise_dim * 2, args.noise_dim * 4), nn.BatchNorm1d(args.noise_dim * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(args.noise_dim * 4, args.latent_dim), nn.Tanh()
    ).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g)

    training_start_time = time.time() # Record start time
    train_generator_on_latents(args, generator, optimizer_g, target_latents, autoencoder, classifier, device)
    training_end_time = time.time() # Record end time
    training_duration_seconds = training_end_time - training_start_time

    print("Generator training finished.")
    generator_save_path = os.path.join(args.output_dir, "generator_final.pth")
    torch.save(generator.state_dict(), generator_save_path)
    print(f"Saved final generator model to {generator_save_path}")

    print(f"\n--- 5. Generating {args.num_final_samples_eval} samples from TRAINED generator ---")
    generator.eval()
    final_generated_latents_list = []
    num_generated = 0
    with torch.no_grad():
        while num_generated < args.num_final_samples_eval:
            num_to_gen_this_batch = min(args.batch_size, args.num_final_samples_eval - num_generated)
            if num_to_gen_this_batch <=0: break
            z_final_noise = torch.randn(num_to_gen_this_batch, args.noise_dim, device=device)
            final_latents_batch = generator(z_final_noise)
            final_generated_latents_list.append(final_latents_batch)
            num_generated += num_to_gen_this_batch
    if not final_generated_latents_list: print("No final latents generated by TRAINED generator."); return
    final_generated_latents = torch.cat(final_generated_latents_list, dim=0)

    print("Decoding generated latents...")
    decoded_images_list = []
    with torch.no_grad():
        for i in range(0, final_generated_latents.size(0), args.batch_size):
             latents_batch_decode = final_generated_latents[i:i+args.batch_size]
             decoded_batch = autoencoder.decode(latents_batch_decode)
             decoded_images_list.append(decoded_batch.cpu())
    if not decoded_images_list: print("No images decoded."); return
    decoded_images = torch.cat(decoded_images_list, dim=0)

    print("Classifying decoded images...")
    predictions_list = []
    with torch.no_grad():
        for i in range(0, decoded_images.size(0), args.batch_size):
            images_batch = decoded_images[i:i+args.batch_size].to(device)
            outputs = classifier(images_batch)
            _, predicted = torch.max(outputs.data, 1)
            predictions_list.append(predicted.cpu())
    if not predictions_list: print("No predictions made."); return
    predictions = torch.cat(predictions_list, dim=0)

    count_0 = (predictions == 0).sum().item(); count_1 = (predictions == 1).sum().item()
    count_other = len(predictions) - count_0 - count_1; total_predictions = len(predictions)
    if total_predictions > 0:
        print(f"Classification results for {total_predictions} generated samples:")
        print(f"- Pred '0': {count_0} ({count_0/total_predictions*100:.2f}%)")
        print(f"- Pred '1': {count_1} ({count_1/total_predictions*100:.2f}%)")
        
        if count_other > 0: print(f"- Pred other: {count_other} ({count_other/total_predictions*100:.2f}%)")
    else:
        print("No predictions to report.")

    # Save prediction counts to a CSV file
    summary_file_path = os.path.join("prediction_summary.csv")
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    percent_0_val = (count_0 / total_predictions * 100) if total_predictions > 0 else 0.0
    percent_1_val = (count_1 / total_predictions * 100) if total_predictions > 0 else 0.0
    
    summary_line = f"{args.exp_name},{timestamp_str},{training_duration_seconds:.2f},{total_predictions},{count_0},{count_1},{percent_0_val:.2f},{percent_1_val:.2f}\n"
    
    try:
        with open(summary_file_path, 'a') as f:
            f.write(summary_line)
        print(f"Saved prediction summary to {summary_file_path}")
    except Exception as e:
        print(f"Error saving prediction summary to {summary_file_path}: {e}")

    plot_generated_images(decoded_images, predictions, num_images=min(args.num_vis_final, decoded_images.shape[0]),
                           title="Final Generated & Classified Samples (from Trained G)",
                           filename=os.path.join(args.figures_dir, f"images/{args.exp_name}.png"))
    if args.latent_dim >=2:
        plot_latent_space(final_generated_latents.cpu().numpy(), predictions.numpy(),
                        title="Final G's Latent Output (Colored by Predicted Class)",
                        filename=os.path.join(args.figures_dir, f"latent/{args.exp_name}.pdf"))
    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train G on synthetic AE Latent Space")
    parser.add_argument('--ae_ckp_path', type=str, default='runs_wae/wae_mmd_prior_uniform_square_l500.0_z2_balanced_train.pth', help='Path to PRE-TRAINED Autoencoder model (.pth)')
    parser.add_argument('--classifier_ckp_path', type=str, default='runs_wae/classifier_0_vs_1_balanced_train.pth', help='Path to PRE-TRAINED Classifier model (.pth)')
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--noise_dim', type=int, default=2) # Should match generator input
    parser.add_argument('--percent_0', type=float, default=90.0, help='Target % of digit 0 in synthetic latent dataset')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=30, help="Epochs for Generator training")
    parser.add_argument('--lr_g', type=float, default=2e-4)
    # --- MODIFIED choices for loss_type ---
    parser.add_argument('--loss_type', type=str, default='sw',
                        choices=['sw', 'twd', 'usot', 'suot', 'sopt', 'spot', 'pawl', 'pot'])
    parser.add_argument('--sw_projections', type=int, default=20)
    parser.add_argument('--sw_p', type=int, default=2, help="Power for SW and also for p-norm in POT cost if not sqeuclidean")
    # TWD args
    parser.add_argument('--twd_ntrees', type=int, default=5)
    parser.add_argument('--twd_nlines', type=int, default=4)
    parser.add_argument('--twd_ftype', type=str, default='linear', choices=['linear', 'poly', 'circular', 'pow', 'circular_concentric'])
    parser.add_argument('--twd_mass_division', type=str, default='distance_based', choices=['uniform', 'distance_based'])
    parser.add_argument('--twd_delta', type=float, default=2.0)
    parser.add_argument('--twd_unbalanced', action='store_true', default=False, help='Use unbalanced TWD')
    parser.add_argument('--min_mass_generated', type=float, default=1.0)
    parser.add_argument('--max_mass_generated', type=float, default=1.0)
    parser.add_argument('--twd_unbalanced_scheduler', type=str, default='constant', choices=['constant', 'linear_increasing', 'linear_decreasing', 'cosine_increasing', 'cosine_decreasing', 'cyclic_cosine', 'reversed_cyclic_cosine'])
    parser.add_argument('--twd_num_cycles', type=int, default=0)
    # USOT/SUOT args
    parser.add_argument('--rho1', type=float, default=0.01)
    parser.add_argument('--rho2', type=float, default=1)
    # SOPT args
    parser.add_argument('--sopt_reg', type=float, default=1)
    # SPOT args
    parser.add_argument('--spot_k', type=int, default=10)
    # --- POT ARGS ---
    parser.add_argument('--pot_reg', type=float, default=0.1, help="Entropic regularization for POT Sinkhorn")
    parser.add_argument('--pot_reg_m_kl', type=float, default=1.0, help="Marginal KL regularization for POT Unbalanced Sinkhorn")
    parser.add_argument('--pot_num_iter_max', type=int, default=100, help="Max iterations for POT Sinkhorn") 
    parser.add_argument('--pot_stop_thr', type=float, default=1e-9, help="Stop threshold for POT Sinkhorn")
    parser.add_argument('--pot_cost_metric_p', type=int, default=2, help="Power p for the cost metric ||x-y||^p. p=2 is sq Euclidean if metric='sqeuclidean', p=1 is L1 if metric='euclidean'.")
    # PAWL args
    parser.add_argument('--pawl_k', type=int, default=10, help="Number of points for PAWL")

    parser.add_argument('--target_latent_dataset_size', type=int, default=50000)
    parser.add_argument('--max_sampling_multiplier', type=int, default=30)
    parser.add_argument('--num_final_samples_eval', type=int, default=5000)
    parser.add_argument('--num_vis_final', type=int, default=100)
    parser.add_argument('--vis_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./output_uot')
    parser.add_argument('--figures_dir', type=str, default='./figures')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic = True # Can slow down, but good for reproducibility
        # torch.backends.cudnn.benchmark = False   # Ensure reproducibility
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)