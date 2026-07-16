import numpy as np
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
import sys
sys.path.append("../..")
from utils import calibrate
from tueplots import bundles
bundles.icml2024()
    
if __name__ == "__main__":
    # Auto-detect device (use cuda:0 if available, otherwise cpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    N_MODELS_FOR_TEST = 4

    # FILE_NAME = "irsl_testtime_resmat1"
    FILE_NAME = "irsl_testtime_resmat2"
    cache_dir = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")

    # Note: weights_only=False is required for PyTorch 2.6+
    testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
    data_tensor = testtime_resmat["data_tensor"].numpy() 
    model_names = testtime_resmat["models"]
    questions   = testtime_resmat["questions"]
    if FILE_NAME == "irsl_testtime_resmat1":
        helm_zs   = np.array(testtime_resmat["zs"])
        datasets    = testtime_resmat["scenarios"]
    elif FILE_NAME == "irsl_testtime_resmat2":
        datasets    = testtime_resmat["datasets"]
    print(data_tensor.shape)
    n_models_for_train = data_tensor.shape[0] - N_MODELS_FOR_TEST
    n_samples_for_train = data_tensor.shape[-1]

    probmat = torch.nanmean(
        torch.tensor(data_tensor[:n_models_for_train, :, :n_samples_for_train], dtype=torch.float, device=device),
        dim = -1,
    )
    n_test_takers, n_items = probmat.shape
    assert not torch.isnan(probmat).any()
    
    z_optimized = calibrate(probmat, device)

    rho2, _ = spearmanr(z_optimized, np.nanmean(data_tensor[:n_models_for_train, :, :], axis=(0, 2)))
    rho3, _ = spearmanr(z_optimized, np.nanmean(data_tensor[n_models_for_train:, :, :], axis=(0, 2)))
    print(f"\
        \ncorr with torch.nanmean(data_tensor[:n_models_for_train, :, :], dim=(0, 2)) = {rho2:.6f} \
        \ncorr with torch.nanmean(data_tensor[n_models_for_train:, :, :], dim=(0, 2)) = {rho3:.6f} \
    ")
    
    test_model_names = model_names[n_models_for_train:]
    assert len(questions) == len(datasets) == z_optimized.shape[0] == data_tensor.shape[1]
    out_path = f"{FILE_NAME}_withz.pt"
    payload = {
        "data_tensor": data_tensor,
        "models": model_names,
        "test_models": test_model_names,
        "questions": questions,
        "datasets": datasets,
        "zs": z_optimized,
    }
    if FILE_NAME == "irsl_testtime_resmat1":
        payload["helm_zs"] = helm_zs
        
    torch.save(payload, out_path)
    