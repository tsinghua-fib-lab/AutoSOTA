import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.fft import dstn, idstn, dctn, idctn


name = 'ns-nonbounded'

output_base_path = "data/"+name+"-merged-spectral/merge_new_{}.npy"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

Nx = 128
Ny =  128
cx, cy = Nx//2, Ny//2
keepx = 32
keepy = 32
pad_a=0
pad_u=0
seq = 10
transform = 'fft' 



slice_x = slice(cx +pad_u - keepx//2,cx+pad_u + keepx//2) 
slice_y = slice(cy +pad_u - keepx//2,cy+pad_u + keepx//2) 
mask_u = np.zeros((2*(cx+pad_u), max(1,2*(cy+pad_u))), dtype=bool)
mask_u[slice_x, slice_y] = True



if transform == 'fft':
    sum_vals = np.zeros((2*seq,keepx,keepy))
    sum_sqs = np.zeros((2*seq,keepx,keepy))
    global_min = np.full((2*seq,keepx,keepy), np.inf)
    global_max = np.full((2*seq,keepx,keepy), -np.inf)
count = 0

filelen = len(os.listdir(f"data/training/{name}/"))

if name ==  'ns-nonbounded':
    data_key = "u"


print("First pass: computing global mean and std")
# === First pass to compute global mean and std ===
for j in range(1,filelen+1):
    print(f"\nProcessing file {j} (first pass)...")
    training_raw_data = sio.loadmat(f"data/training/{name}/{name}_{j}.mat")
    u = training_raw_data[data_key]
    
    
    if transform == 'fft':
        min_vals = np.full((2*seq,keepx,keepy), np.inf)
        max_vals = np.full((2*seq,keepx,keepy), -np.inf)
        sum_vals_j = np.zeros((2*seq,keepx,keepy))
        sum_sqs_j = np.zeros((2*seq,keepx,keepy))     
    for i in range(u.shape[0]):
            if name == "ns-nonbounded":
                # slice along last axis (time)
                f = u[i, :, :, :]
            else:
                raise ValueError(f"Unknown dataset name: {name}")
            
            if transform == 'fft':
                
                f_fft = np.fft.fft2(f, axes=(0, 1), norm='ortho')
                f_fft_shifted = np.fft.fftshift(f_fft, axes=(0, 1))               
                components = np.concatenate([np.stack([np.real(f_fft_shifted[slice_x, slice_y, ii]), np.imag(f_fft_shifted[slice_x, slice_y, ii])], axis=0)
                                                for ii in range(f_fft_shifted.shape[-1])], axis=0) # shape (seq*2, H, W)
    
                sum_vals += components
                sum_sqs += (components**2)
                count += 1
                
                sum_vals_j += components
                sum_sqs_j += (components**2)
                min_vals = np.minimum(min_vals, components)
                max_vals = np.maximum(max_vals, components)

    # Update global min/max
    global_min = np.minimum(global_min, min_vals)
    global_max = np.maximum(global_max, max_vals)
    
    mean_j = sum_vals_j / (u.shape[0])
    std_j = np.sqrt(sum_sqs_j / (u.shape[0]) - mean_j**2)

    print(f"File {j} stats:")
    print(f"  Mean: {mean_j}")
    print(f"  Std : {std_j}")
    print(f"  Min : {min_vals}")
    print(f"  Max : {max_vals}")

# === Compute global stats ===
mean = sum_vals / count
std = np.sqrt(sum_sqs / count - mean ** 2)

# IN ORDER TO REDUCE MAX AND MIN
std_mult = 2
std = std*std_mult

print("\n=== Global Stats ===")
print("Global mean:", mean)
print("Global std :", std)
print("Global min :", global_min)
print("Global max :", global_max)


# Save stats
os.makedirs("processed", exist_ok=True)
np.save(f"processed/{name}_mean.npy", mean)
np.save(f"processed/{name}_std.npy", std)
np.save(f"processed/{name}_min.npy", global_min)
np.save(f"processed/{name}_max.npy", global_max)

##############################################################################

mean = np.load(f"processed/{name}_mean.npy")
std = np.load(f"processed/{name}_std.npy")
global_min = np.load(f"processed/{name}_min.npy")
global_max = np.load(f"processed/{name}_max.npy")


# Load raw training data from .mat files
for j in range(1,filelen+1):
    print(f"Processing file {j}...")
    training_raw_data = sio.loadmat(f"data/training/{name}/{name}_{j}.mat")
   
    u = training_raw_data[data_key]

    if transform == 'fft':
        min_vals = np.full(2*seq, np.inf)
        max_vals = np.full(2*seq, -np.inf)
        mean_acc = np.zeros(2*seq)
        std_acc = np.zeros(2*seq)
        all_hist_data = [[] for _ in range(2*seq)]  # collect hist data per channel

    for i in range(u.shape[0]):
            if name == "ns-nonbounded":
                # slice along last axis (time)
                f = u[i, :, :, :]
            
            if transform == 'fft':
                f_fft = np.fft.fft2(f, axes=(0, 1), norm='ortho')
                f_fft_shifted = np.fft.fftshift(f_fft, axes=(0, 1))
                components = np.concatenate([np.stack([np.real(f_fft_shifted[slice_x, slice_y, ii]), np.imag(f_fft_shifted[slice_x, slice_y, ii])], axis=0)
                                                for ii in range(f_fft_shifted.shape[-1])], axis=0) # shape (seq*2, H, W)
                
                normalized = (components - mean) / (std + 1e-8)
                # Final shape: (keepx, keepy, 2*seq) or (keepx, 2*seq) 
                combined = np.moveaxis(normalized.squeeze(), 0, -1)
               
                # Save the combined array to a new .npy file
                output_file_path = output_base_path.format(i+(j-1)*u.shape[0])
                np.save(output_file_path, combined)
                
                reshaped = combined.reshape(-1, 2*seq)
                mean_acc += reshaped.mean(axis=0)
                std_acc += reshaped.std(axis=0)
                min_vals = np.minimum(min_vals, reshaped.min(axis=0))
                max_vals = np.maximum(max_vals, reshaped.max(axis=0))
   
            
            if combined.min()<-20:
                print(f"Saved combined array for index {i} to {output_file_path}")
                plt.hist(normalized[:,:,-1].flatten(), bins=300, density=True, color='steelblue', alpha=0.85, label='Histogram')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
            
    # Print stats per file
    print(f"Normalized file {j} stats:")
    print(f"  Mean: {mean_acc / u.shape[0]}")
    print(f"  Std : {std_acc / u.shape[0]}")
    print(f"  Min : {min_vals}")
    print(f"  Max : {max_vals}")
    
 

print("Finished processing all files.")
