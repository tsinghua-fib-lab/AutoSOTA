"""Quick pre-processing for evaluation data with unique task IDs."""
import h5py, numpy as np, os, glob

def build_coordinate_train(L_PE, L_RO):
    x = np.linspace(0, 1, L_PE)
    y = np.linspace(0, 1, L_RO)
    x, y = np.meshgrid(x, y, indexing="ij")
    return np.stack([x, y], -1)

def crop_center(img, h, w):
    sh, sw = img.shape[-2], img.shape[-1]
    return img[..., (sh-h)//2:(sh+h)//2, (sw-w)//2:(sw+w)//2]

input_dir = "/datasets/fastmri_dl"
output_dir = "/datasets/fastmri_eval_v2"
target_res = 256
ACS_lines = 24
R = 10

os.makedirs(output_dir, exist_ok=True)

h5_files = sorted(glob.glob(os.path.join(input_dir, "*test*.h5")))
task_counter = 0

for fi, h5_path in enumerate(h5_files):
    try:
        with h5py.File(h5_path, "r") as f:
            kspace = f["kspace"][:]
            gt_img = f.get("reconstruction_rss", None)
            if gt_img is None:
                gt_img = f.get("reconstruction_esc", None)
            if gt_img is None:
                continue
            gt_img = gt_img[:]
    except Exception as e:
        print("Skip %s: %s" % (os.path.basename(h5_path), e))
        continue

    nslices = min(kspace.shape[0], 5)
    
    for si in range(nslices):
        slice_ksp = kspace[si]
        ksp_cropped = crop_center(slice_ksp, target_res, target_res)
        
        # IFFT for normalization
        zf = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ksp_cropped)))
        norm = np.max(np.abs(zf))
        if norm == 0:
            continue
        
        # Create Cartesian 1D mask
        mask = np.zeros((target_res, target_res), dtype=np.float32)
        step = int(R)
        samples = list(range(0, target_res, step))
        center = target_res // 2 - ACS_lines // 2
        center_idx = list(range(center, center + ACS_lines))
        all_idx = sorted(set(samples + center_idx))
        all_idx = [i for i in all_idx if 0 <= i < target_res]
        mask[:, all_idx] = 1.0
        
        forward_fft = ksp_cropped / norm
        forward_fft_und = forward_fft * mask
        
        # Ground truth
        if gt_img.shape[-1] == target_res:
            gt_slice = gt_img[si]
        else:
            gt_slice = crop_center(gt_img[si], target_res, target_res)
        gt_complex = gt_slice.astype(np.complex128) / norm
        
        csmp = np.ones((1, target_res, target_res), dtype=np.complex128)
        coords = build_coordinate_train(target_res, target_res)
        
        task_dir = os.path.join(output_dir, "task_eval_%04d" % task_counter)
        os.makedirs(task_dir, exist_ok=True)
        
        sample_path = os.path.join(task_dir, "sample_0000.h5")
        with h5py.File(sample_path, "w") as hf_out:
            hf_out.create_dataset("forward_fft", data=forward_fft.astype(np.complex64))
            hf_out.create_dataset("forward_fft_und", data=forward_fft_und.astype(np.complex64))
            hf_out.create_dataset("mask", data=mask.astype(np.float32))
            hf_out.create_dataset("csmp", data=csmp.astype(np.complex64))
            hf_out.create_dataset("img_full", data=gt_complex.astype(np.complex64))
            hf_out.create_dataset("slice_idx", data=si)
            hf_out.create_dataset("coordinates", data=coords.astype(np.float32))
            hf_out.attrs["R"] = R
            hf_out.attrs["ACS"] = ACS_lines
            hf_out.attrs["Type"] = "Cartesian_1D"
            hf_out.attrs["num_captions"] = 0
        
        task_counter += 1

print("Created %d eval tasks in %s" % (task_counter, output_dir))
