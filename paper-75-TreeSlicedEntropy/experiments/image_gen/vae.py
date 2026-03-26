import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset, Subset # ConcatDataset no longer needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import math
import argparse
import random # For sampling in augmentation

from model import Autoencoder, Classifier # Import models
# Remove TSW imports if not used for WAE-MMD, or keep if you plan to switch loss
# from TW_concurrent_lines import TWConcurrentLines, generate_trees_frames

# --- Model Loading Functions ---
def load_autoencoder(path, latent_dim, device):
    """Loads the Autoencoder model from a checkpoint (typically a globally pre-trained AE)."""
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
    """Loads the Classifier model from a checkpoint (typically a globally pre-trained Classifier)."""
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

# --- NEWLY ADDED: Generic Model State Loader ---
def load_model_state(model, path, device):
    """Generic function to load model state_dict into an existing model instance and return status."""
    if path and os.path.exists(path):
        print(f"Attempting to load model state from {path}...")
        try:
            model.to(device) # Ensure model is on the correct device before loading
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval() # Set to evaluation mode after loading
            print("Model state loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model state from {path}: {e}")
            try: # Fallback
                print("Trying fallback: loading state_dict to CPU first...")
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print("Fallback loading of model state successful.")
                return True
            except Exception as e2:
                 print(f"Fallback loading of model state failed: {e2}")
                 return False
    else:
        print(f"Info: Model state file not found at '{path}' (or path not provided).")
        return False

# --- Dataset Preparation ---
class FilteredMNIST(Dataset):
    def __init__(self, mnist_dataset, target_digits):
        self.mnist_dataset = mnist_dataset
        self.target_digits = target_digits
        
        indices_by_digit = {digit: [] for digit in range(10)}
        try:
            dataset_targets = mnist_dataset.targets
            if isinstance(dataset_targets, torch.Tensor): dataset_targets = dataset_targets.tolist()
        except AttributeError:
            dataset_targets = [label for _, label in mnist_dataset]

        for i in range(len(mnist_dataset)):
            label = dataset_targets[i]
            if isinstance(label, torch.Tensor): label = label.item()
            if label in target_digits:
                 indices_by_digit[label].append(i)

        final_indices = []
        for digit in target_digits:
            final_indices.extend(indices_by_digit[digit])
        
        random.Random(42).shuffle(final_indices)
        self.subset = Subset(mnist_dataset, final_indices)

    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        return self.subset[idx]

class BalancedAugmentedMNIST(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def create_balanced_augmented_train_dataset(root_dir='./data', target_digits=[0, 1], seed=42):
    print(f"\n--- Creating Balanced and Augmented Training Dataset for digits {target_digits} ---")
    mnist_pil_train = datasets.MNIST(root=root_dir, train=True, download=True, transform=None)
    basic_to_tensor = transforms.ToTensor()
    augmentation_pipeline = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor()
    ])
    indices_class_0 = [i for i, target_label in enumerate(mnist_pil_train.targets) if target_label == target_digits[0]] # Renamed target
    indices_class_1 = [i for i, target_label in enumerate(mnist_pil_train.targets) if target_label == target_digits[1]] # Renamed target
    count_0 = len(indices_class_0); count_1 = len(indices_class_1)
    print(f"Original counts - Digit {target_digits[0]}: {count_0}, Digit {target_digits[1]}: {count_1}")
    all_samples_list = []
    for idx in indices_class_0:
        img_pil, label = mnist_pil_train[idx]
        all_samples_list.append((basic_to_tensor(img_pil), label))
    for idx in indices_class_1:
        img_pil, label = mnist_pil_train[idx]
        all_samples_list.append((basic_to_tensor(img_pil), label))

    num_to_augment = 0
    minority_indices = []
    minority_label = -1

    if count_0 > count_1:
        minority_indices = indices_class_1; minority_label = target_digits[1]
        num_to_augment = count_0 - count_1
        print(f"Digit {target_digits[1]} is minority. Augmenting {num_to_augment} samples.")
    elif count_1 > count_0:
        minority_indices = indices_class_0; minority_label = target_digits[0]
        num_to_augment = count_1 - count_0
        print(f"Digit {target_digits[0]} is minority. Augmenting {num_to_augment} samples.")
    else:
        print("Classes are already balanced. No augmentation for balancing needed.")

    if num_to_augment > 0 and minority_indices: # Ensure minority_indices is not empty
        rng = random.Random(seed)
        for _ in range(num_to_augment):
            original_pil_idx = rng.choice(minority_indices)
            img_pil, _ = mnist_pil_train[original_pil_idx]
            augmented_img_tensor = augmentation_pipeline(img_pil)
            all_samples_list.append((augmented_img_tensor, minority_label))

    rng_shuffle = random.Random(seed); rng_shuffle.shuffle(all_samples_list)
    final_count_0 = sum(1 for _, label in all_samples_list if label == target_digits[0])
    final_count_1 = sum(1 for _, label in all_samples_list if label == target_digits[1])
    print(f"Balanced dataset created. Total samples: {len(all_samples_list)}")
    print(f"  Final counts - Digit {target_digits[0]}: {final_count_0}, Digit {target_digits[1]}: {final_count_1}")
    return BalancedAugmentedMNIST(all_samples_list)

# --- MMD Math Helpers (Unchanged) ---

def compute_gaussian_kernel(x, y, sigma=1.0): # For MMD
    beta = 1. / (2. * sigma**2); dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-beta * dist_sq)
def compute_mmd(z_encoded, z_prior, kernel_sigma=None): # For MMD
    if kernel_sigma is None:
        all_samples = torch.cat((z_encoded, z_prior), dim=0)
        dists_sq = torch.cdist(all_samples, all_samples, p=2)**2
        if dists_sq[dists_sq > 1e-9].numel() > 0:
             median_dist_sq = torch.median(dists_sq[dists_sq > 1e-9])
             kernel_sigma = torch.sqrt(median_dist_sq / 2.0)
        else: kernel_sigma = torch.tensor(1.0, device=z_encoded.device) 
        kernel_sigma = torch.clamp(kernel_sigma, min=1e-3)
    K_zz = compute_gaussian_kernel(z_encoded, z_encoded, kernel_sigma)
    K_pp = compute_gaussian_kernel(z_prior, z_prior, kernel_sigma)
    K_zp = compute_gaussian_kernel(z_encoded, z_prior, kernel_sigma)
    batch_size = z_encoded.size(0)
    if batch_size <= 1: return torch.tensor(0.0, device=z_encoded.device)
    mmd_val = ( K_zz.sum() - K_zz.diag().sum() ) / (batch_size * (batch_size - 1)) \
            + ( K_pp.sum() - K_pp.diag().sum() ) / (batch_size * (batch_size - 1)) \
            - 2 * K_zp.mean()
    mmd_val = F.relu(mmd_val); return mmd_val

# --- Visualization Helpers (Unchanged) ---
def plot_latent_space(latents, labels, title="Latent Space", filename="latent_space.png"):
    plt.figure(figsize=(8, 8))
    unique_labels = sorted(np.unique(labels))
    colors = plt.cm.viridis(np.linspace(0, 1, max(1,len(unique_labels))))
    for i, label_val in enumerate(unique_labels):
        idx = labels == label_val
        plt.scatter(latents[idx, 0], latents[idx, 1], color=colors[i], label=f"Digit {label_val}", alpha=0.6, s=10)
    plt.title(title); plt.xlabel("Latent Dim 1"); plt.ylabel("Latent Dim 2")
    if len(unique_labels) > 0 and not (len(unique_labels)==1 and unique_labels[0] == -1) : plt.legend() # Avoid legend for all -1
    plt.grid(True); plt.savefig(filename); plt.close(); print(f"Saved latent space plot to {filename}")

def plot_generated_images(images, labels=None, num_images=25, title="Generated Images", filename="generated_images.png"):
    if images.shape[0] == 0: print(f"No images to plot for {filename}."); return
    if images.shape[0] < num_images: num_images = images.shape[0]
    cols = int(np.sqrt(num_images)); rows = (num_images + cols - 1) // cols if cols > 0 else 1
    if rows == 0 : rows = 1
    plt.figure(figsize=(cols * 1.5, rows * 1.5)); images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy() if labels is not None else None
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1); plt.imshow(images_np[i].squeeze(), cmap='gray')
        plt.xticks([]); plt.yticks([])
        if labels_np is not None: plt.title(f"Pred: {labels_np[i]}", fontsize=8)
    plt.suptitle(title); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close()
    print(f"Saved generated images plot to {filename}")

# --- Classifier Training (Unchanged) ---
def train_classifier(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train(); total_loss = 0; processed_samples = 0; num_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device); optimizer.zero_grad(); output = model(data)
        loss = criterion(output, target); loss.backward(); optimizer.step()
        total_loss += loss.item() * data.size(0); processed_samples += data.size(0)
        if batch_idx > 0 and batch_idx % log_interval == 0: print(f'Train Classifier Epoch: {epoch} [{processed_samples}/{len(train_loader.dataset)} ({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}')
    avg_loss = total_loss / len(train_loader.dataset); print(f'====> Classifier Epoch {epoch} Average loss: {avg_loss:.4f}')

# --- Classifier Testing (Unchanged) ---
def test_classifier(model, device, test_loader, criterion):
    model.eval(); test_loss = 0; correct = 0; total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device); output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True); correct += pred.eq(target.view_as(pred)).sum().item(); total_samples += data.size(0)
    avg_test_loss = test_loss / total_samples if total_samples > 0 else 0
    accuracy = 100. * correct / total_samples if total_samples > 0 else 0
    print(f'\nTest Classifier set: Average loss: {avg_test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n'); return accuracy


# --- WAE Loss and Training (Unchanged from previous WAE script) ---
def loss_function_wae_mmd(recon_x, x, z_encoded, wae_lambda=10.0, prior_type='gaussian', latent_dim=2):
    batch_size = x.size(0)
    recon_loss = F.binary_cross_entropy(recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction='sum') / batch_size
    if prior_type == 'gaussian': z_prior = torch.randn_like(z_encoded)
    elif prior_type == 'uniform_square': z_prior = 2 * torch.rand_like(z_encoded) - 1
    else: raise ValueError(f"Unsupported prior_type: {prior_type}")
    mmd_loss = compute_mmd(z_encoded, z_prior)
    total_loss = recon_loss + wae_lambda * mmd_loss
    return total_loss, recon_loss, mmd_loss
def train_wae_mmd(epoch, wae_model, wae_optimizer, train_loader, wae_lambda, device, log_interval, prior_type, latent_dim):
    wae_model.train(); train_loss = 0; recon_loss_acc = 0; mmd_loss_acc = 0
    processed_samples = 0; num_batches = len(train_loader)
    if num_batches == 0: print(f"Warning: Train loader is empty for WAE training epoch {epoch}. Skipping."); return 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device); wae_optimizer.zero_grad()
        recon_batch, z_encoded, mu, logvar = wae_model.forward_train(data) # Assumes forward_train in model.Autoencoder
        loss, recon, mmd = loss_function_wae_mmd(recon_batch, data, z_encoded, wae_lambda, prior_type, latent_dim)
        loss.backward(); wae_optimizer.step()
        train_loss += loss.item() * data.size(0); recon_loss_acc += recon.item() * data.size(0)
        mmd_loss_acc += mmd.item() * data.size(0); processed_samples += data.size(0)
        if batch_idx > 0 and batch_idx % log_interval == 0:
             print(f'Train WAE Epoch: {epoch} [{processed_samples}/{len(train_loader.dataset)} ({100. * batch_idx / num_batches:.0f}%)]\t'
                   f'Loss: {loss.item():.4f} (Recon: {recon.item():.4f}, MMD: {mmd.item():.6f})')
    avg_loss = train_loss / processed_samples if processed_samples > 0 else 0
    avg_recon = recon_loss_acc / processed_samples if processed_samples > 0 else 0
    avg_mmd = mmd_loss_acc / processed_samples if processed_samples > 0 else 0
    print(f'====> WAE Epoch: {epoch} Avg loss: {avg_loss:.4f} | Recon: {avg_recon:.4f}, MMD: {avg_mmd:.6f} (lambda={wae_lambda}, prior={prior_type})')
    return avg_loss

# --- Visualization Function for WAE (Unchanged) ---
def visualize_latent_sampling_classification_wae(
        wae_model, classifier_model, device, num_samples, radius,
        latent_dim, batch_size, save_path, prior_type_viz='uniform_square'):
    if latent_dim < 2: print(f"Error: Latent dimension is {latent_dim}. Visualization requires dim >= 2."); return
    if not isinstance(classifier_model, nn.Module): print(f"Error: Invalid classifier model passed."); return
    print(f"Generating WAE visualization by sampling {num_samples} latent points (shape: square, half-side={radius}, dim={latent_dim}), decoding, and classifying...")
    wae_model.eval(); classifier_model.eval(); all_z_samples = []; all_pred_labels = []
    z1_cpu = (2 * torch.rand(num_samples) - 1) * radius; z2_cpu = (2 * torch.rand(num_samples) - 1) * radius
    if latent_dim == 2: z_samples_cpu = torch.stack((z1_cpu, z2_cpu), dim=1)
    else:
        zeros_cpu = torch.zeros(num_samples, latent_dim - 2)
        z_samples_cpu = torch.cat((z1_cpu.unsqueeze(1), z2_cpu.unsqueeze(1), zeros_cpu), dim=1)
    num_batches_vis = math.ceil(num_samples / batch_size)
    with torch.no_grad():
        for i in range(num_batches_vis):
            start_idx = i * batch_size; end_idx = min((i + 1) * batch_size, num_samples)
            z_batch = z_samples_cpu[start_idx:end_idx].to(device); all_z_samples.append(z_batch.cpu())
            generated_images_batch = wae_model.decode(z_batch)
            classifier_output_batch = classifier_model(generated_images_batch)
            pred_labels_batch = classifier_output_batch.argmax(dim=1); all_pred_labels.append(pred_labels_batch.cpu())
    all_z_samples_np = torch.cat(all_z_samples, dim=0).numpy()
    all_pred_labels_np = torch.cat(all_pred_labels, dim=0).numpy()
    plt.figure(figsize=(10, 8)); plot_dim_0 = 0; plot_dim_1 = 1
    z_plot = all_z_samples_np[:, [plot_dim_0, plot_dim_1]]
    colors = ['#1f77b4' if label == 0 else '#ff7f0e' for label in all_pred_labels_np]
    plt.scatter(z_plot[:, 0], z_plot[:, 1], c=colors, s=5, alpha=0.6)
    class_labels = ['Predicted 0', 'Predicted 1']
    scatter1 = plt.Line2D([0], [0], marker='o', color='w', label=class_labels[0], ms=8, mfc='#1f77b4', alpha=0.8)
    scatter2 = plt.Line2D([0], [0], marker='o', color='w', label=class_labels[1], ms=8, mfc='#ff7f0e', alpha=0.8)
    title = f'Classifier Preds on Decoded Samples (WAE Trained with {args.prior_type.replace("_", " ").title()} Prior)\nLatent Visualization Samples from Uniform Square (half-side={radius})'
    xlabel = f'Latent Dimension {plot_dim_0}'; ylabel = f'Latent Dimension {plot_dim_1}'
    if latent_dim > 2: title += f'\n(Showing Dims {plot_dim_0} & {plot_dim_1} of {latent_dim}; others set to 0)'
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    square_outline = patches.Rectangle((-radius,-radius),2*radius,2*radius,lw=1.5,ls='--',ec='red',fc='none',label=f'Viz Area: Sq (half-side={radius})')
    plt.gca().add_patch(square_outline)
    handles = [scatter1, scatter2, square_outline]; labels_legend = class_labels + [square_outline.get_label()]
    plt.legend(handles=handles, labels=labels_legend, title="Legend")
    plt.grid(True, linestyle='--', alpha=0.6); plt.axhline(0, color='grey', lw=0.5); plt.axvline(0, color='grey', lw=0.5)
    plt.axis('equal'); plt.savefig(save_path, bbox_inches='tight'); print(f"Saved WAE latent sampling visualization to {save_path}"); plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parser ---
    # (Your existing parser setup from the script)
    parser = argparse.ArgumentParser(description='WAE-MMD with (optionally balanced) MNIST 0 vs 1 Classification')
    parser.add_argument('--wae-lambda', type=float, default=1000.0, help='Weight for WAE MMD term')
    parser.add_argument('--prior-type', type=str, default='uniform_square', choices=['gaussian', 'uniform_square'], help='WAE prior')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs-classifier', type=int, default=20)
    parser.add_argument('--epochs-wae', type=int, default=50)
    parser.add_argument('--lr-classifier', type=float, default=1e-3)
    parser.add_argument('--lr-wae', type=float, default=3e-5)
    parser.add_argument('--latent-dim', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-dir', type=str, default='./runs_wae')
    parser.add_argument('--force-train-classifier', action='store_true', default=False)
    parser.add_argument('--force-train-wae', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--vis-samples', type=int, default=5000)
    parser.add_argument('--vis-radius', type=float, default=1.0, help='Half-side for square viz sampling area')
    parser.add_argument('--vis-batch-size', type=int, default=512)
    # --percent_0 is removed as training data is now 50/50 balanced for the two target digits.
    # If you need to control the proportion in the balanced set, that logic needs to be added to create_balanced_augmented_train_dataset
    
    # Adding argument for pre-trained AE and Classifier paths (if used by create_balanced_augmented_train_dataset indirectly)
    # These are distinct from the paths where this script saves its trained models.
    parser.add_argument('--ae_ckp_path', type=str, default=None, help='Path to pre-trained Autoencoder model (.pth) for dataset creation/viz')
    parser.add_argument('--classifier_ckp_path', type=str, default=None, help='Path to pre-trained Classifier model (.pth) for dataset creation/viz')


    args = parser.parse_args() # Parse arguments

    # Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if use_cuda: torch.cuda.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"WAE prior: {args.prior_type}, MMD Lambda: {args.wae_lambda}")

    # Model paths for models trained BY THIS SCRIPT
    classifier_model_name_script = f'classifier_0_vs_1_balanced_train.pth' # Renamed
    wae_model_name_script = f'wae_mmd_prior_{args.prior_type}_l{args.wae_lambda}_z{args.latent_dim}_balanced_train.pth' # Renamed
    classifier_save_path = os.path.join(args.save_dir, classifier_model_name_script)
    wae_save_path = os.path.join(args.save_dir, wae_model_name_script)
    visualization_save_path = os.path.join(args.save_dir, f'viz_wae_prior_{args.prior_type}_r{args.vis_radius}_n{args.vis_samples}_l{args.wae_lambda}_z{args.latent_dim}_balanced.png') # Renamed


    # --- Create Balanced Augmented Training Dataset (for WAE and Classifier) ---
    balanced_train_dataset = create_balanced_augmented_train_dataset(
        root_dir='./data', target_digits=[0, 1], seed=args.seed
    )
    if len(balanced_train_dataset) == 0:
        print("Failed to create a balanced training dataset. Exiting.")
        exit()
    # Train loader for WAE and Classifier (uses balanced data)
    train_loader = DataLoader(balanced_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_cuda)
    print(f"Using BALANCED and AUGMENTED training loader with {len(balanced_train_dataset)} samples.")

    # --- Prepare Test Dataset (original imbalance of 0s and 1s) ---
    basic_transform_test = transforms.Compose([transforms.ToTensor()]) # Renamed
    full_test_dataset_mnist = datasets.MNIST('./data', train=False, download=True, transform=basic_transform_test)
    # FilteredMNIST for test set will contain only 0s and 1s, reflecting their natural proportion in the test set
    test_dataset_filtered_for_0_1 = FilteredMNIST(full_test_dataset_mnist, target_digits=[0, 1])
    test_loader = DataLoader(test_dataset_filtered_for_0_1, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_cuda)
    print(f"Using original filtered TEST loader with {len(test_dataset_filtered_for_0_1)} samples for digits 0 & 1.")


    # --- Train or Load Classifier (trained/loaded by this script) ---
    # This classifier is specific to distinguishing 0 and 1, trained on balanced data.
    classifier_script = Classifier(num_classes=2).to(device) # Renamed
    classifier_optimizer = optim.Adam(classifier_script.parameters(), lr=args.lr_classifier)
    classifier_criterion = nn.CrossEntropyLoss()
    classifier_loaded = load_model_state(classifier_script, classifier_save_path, device) # Uses generic loader
    classifier_ready = False

    if not classifier_loaded or args.force_train_classifier:
        print("\n--- Training Classifier (Digits 0 vs 1, on BALANCED data) ---")
        best_acc = 0
        for epoch in range(1, args.epochs_classifier + 1):
            train_classifier(classifier_script, device, train_loader, classifier_optimizer, classifier_criterion, epoch, args.log_interval)
            accuracy = test_classifier(classifier_script, device, test_loader, classifier_criterion)
            if accuracy > best_acc:
                best_acc = accuracy
                print(f"Saving best classifier model (Acc on test: {best_acc:.2f}%) to {classifier_save_path}")
                torch.save(classifier_script.state_dict(), classifier_save_path)
        print("--- Classifier Training Finished ---\n")
        # Ensure the best or last trained model is loaded for subsequent use
        if os.path.exists(classifier_save_path):
             load_model_state(classifier_script, classifier_save_path, device)
        else: # Fallback if somehow not saved
            print("Warning: Classifier was trained but save file not found. Using current model state.")
    
    if load_model_state(classifier_script, classifier_save_path, device): # Double check load or use current state
        classifier_ready = True
        print("--- Classifier (trained by this script) Ready ---")
        print("Final test of this script's classifier on original imbalanced test set:")
        test_classifier(classifier_script, device, test_loader, classifier_criterion)
    elif args.force_train_classifier: # If it was just trained
        classifier_ready = True
        print("--- Classifier (just trained by this script) Ready ---")
        print("Final test of this script's classifier on original imbalanced test set:")
        test_classifier(classifier_script, device, test_loader, classifier_criterion)
    else:
        print("Classifier model (for this script) not available. Cannot proceed with visualization that requires it.")
        classifier_ready = False


    # --- Train or Load WAE (trained/loaded by this script) ---
    # This WAE model is specific to this script's training run.
    wae_model_script = Autoencoder(latent_dim=args.latent_dim).to(device) # Renamed
    wae_optimizer = optim.Adam(wae_model_script.parameters(), lr=args.lr_wae)
    wae_loaded = load_model_state(wae_model_script, wae_save_path, device) # Uses generic loader
    wae_ready = False

    if not wae_loaded or args.force_train_wae:
        print(f"\n--- Training WAE-MMD (Prior: {args.prior_type}, Lambda={args.wae_lambda}, Z_dim={args.latent_dim}, on BALANCED data) ---")
        for epoch in range(1, args.epochs_wae + 1):
             train_wae_mmd(epoch, wae_model_script, wae_optimizer, train_loader,
                               args.wae_lambda, device, args.log_interval, args.prior_type, args.latent_dim)
        print("--- WAE Training Finished ---")
        print(f"Saving final WAE model to {wae_save_path}")
        torch.save(wae_model_script.state_dict(), wae_save_path)
        wae_ready = True # It was just trained
    elif wae_loaded: # Successfully loaded from file
         print("--- WAE (trained by this script) Loaded from File ---")
         wae_ready = True
    else: # Not loaded and not force_trained
        print("WAE model (for this script) file not found and training was not forced.")
        wae_ready = False


    # --- Generate Visualization (using the WAE and Classifier from this script's run) ---
    if wae_ready and classifier_ready:
        print("\n--- Generating Latent Space Visualization (Sampled Points -> WAE Decoded Images -> Classifier Predictions) ---")
        visualize_latent_sampling_classification_wae(
            wae_model=wae_model_script,       # Use WAE model from this script
            classifier_model=classifier_script, # Use Classifier from this script
            device=device,
            num_samples=args.vis_samples,
            radius=args.vis_radius,
            latent_dim=args.latent_dim,
            batch_size=args.vis_batch_size,
            save_path=visualization_save_path,
            prior_type_viz=args.prior_type
        )
    elif not wae_ready: print("Skipping visualization as WAE (from this script's run) is not available.")
    elif not classifier_ready: print("Skipping visualization as Classifier (from this script's run) is not available.")

    print("\nScript finished.")