import argparse
import torch
from loguru import logger
from alignment_utils import load_features
from gpa import alignment
from save_features import  get_config
from utils import seed_everything, MAX_ITER_MAP


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().item()
    acc = 100 * acc / target.shape[0]
    return acc

  
def process_feature(args, cfg):


    logger.info("Loading VFM features...")
    VFM_train_arrays, VFM_test_arrays = load_features(cfg, args)
    args.model_type = "clip"
    logger.info("Loading VLM features...")
    VLM_train_arrays, VLM_test_arrays = load_features(cfg, args)


    VFM_train_features = VFM_train_arrays['visual_features']
    VFM_test_features = VFM_test_arrays[0]['visual_features']


    VLM_train_features = VLM_train_arrays['visual_features']
    VLM_test_features = VLM_test_arrays[0]['visual_features']


    train_label = VLM_train_arrays['labels']
    test_label = VLM_test_arrays[0]['labels']
    

    clip_prototypes = VLM_test_arrays[0]['text_features']
    train_clip_prototypes = VLM_train_arrays['text_features']


    gpu_id = args.gpu_id 
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


    VLM_train_features = VLM_train_features.to(device)
    VLM_test_features = VLM_test_features.to(device)
    VFM_train_features = VFM_train_features.to(device)
    VFM_test_features = VFM_test_features.to(device)
    train_label = train_label.to(device)
    test_label = test_label.to(device)
    train_clip_prototypes = train_clip_prototypes.to(device)
    clip_prototypes = clip_prototypes.to(device)

    query_train_features = torch.cat((VFM_train_features, VLM_train_features), dim=1)
    query_test_features = torch.cat((VFM_test_features, VLM_test_features), dim=1)

    return VLM_train_features, VLM_test_features, query_train_features, query_test_features, train_label, test_label, clip_prototypes, clip_prototypes, VLM_train_arrays, VLM_test_arrays


def init_mu(K, d, z, query_features):

    mu = torch.zeros(K, d,
                        device=query_features.device)
    n_most_confident = 8
    topk_values, topk_indices = torch.topk(z, k=n_most_confident, dim=0)  # 8 pseudo-labels per class

    mask = torch.zeros_like(z).scatter_(0, topk_indices, 1)
    filtered_z = z * mask
    for c in range(K):
        class_indices = mask[:, c].nonzero().squeeze(1)
        class_features = query_features[class_indices]
        class_z = filtered_z[
            class_indices, c].unsqueeze(
            1)

        combined = class_features * class_z
        component_mean = combined[:n_most_confident].mean(dim=0)
        mu[c, :] = component_mean
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def update_mu(query_features, z, prev_C=None, momentum=0.9, topk=None):
    """
    Update class centers (prototypes) based on soft assignments.

    Args:
        query_features: [N, d] Tensor of sample features
        z            : [N, K] Soft assignments / confidence scores
        prev_C       : [K, d] Previous class centers. If None, use current computation
        momentum     : EMA smoothing factor (higher -> slower update)
        topk         : int or None. If specified, use only the top-k most confident samples per class

    Returns:
        new_C: [K, d] Updated class centers
    """

    N, K = z.shape
    d = query_features.shape[1]

    # Select top-k confident samples per class if requested
    if topk is not None:
        # Create a mask to keep only top-k samples
        mask = torch.zeros_like(z)
        topk_vals, topk_idx = torch.topk(z, k=min(topk, N), dim=0)
        mask.scatter_(0, topk_idx, 1.0)
        z_filtered = z * mask
    else:
        z_filtered = z

    # Compute total weight per class
    class_mass = z_filtered.sum(dim=0) + 1e-6  # [K]

    # Compute weighted average of features per class
    C_new = torch.einsum('nk,nd->kd', z_filtered, query_features)
    C_new = C_new / class_mass.unsqueeze(1)

    # Normalize the class centers
    C_new = C_new / (C_new.norm(dim=1, keepdim=True) + 1e-6)

    # If no previous class centers, return the new ones
    if prev_C is None:
        return C_new

    # Otherwise, apply EMA smoothing
    C_smooth = momentum * prev_C + (1 - momentum) * C_new
    C_smooth = C_smooth / (C_smooth.norm(dim=1, keepdim=True) + 1e-6)

    return C_smooth


def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=10, tol=1e-5):
    out = out - out.max(dim=1, keepdim=True)[0]
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]
    Q /= torch.sum(Q)

    for _ in range(sinkhorn_iterations):
        Q_prev = Q.clone()
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
        err = torch.max(torch.abs(Q - Q_prev))
        if err < tol:
            break

    Q *= B
    return Q.t()


def main(args, cfg):

    VLM_train_features, VLM_test_features, VFM_train_features, VFM_test_features, train_label , test_label, train_clip_prototypes, clip_prototypes, train_a, test_a = process_feature(args, cfg)

    d = VFM_train_features.size(1)
    
    query_features = VFM_train_features
    support_features = VLM_train_features
    query_label = train_label

    y_hat = 100 * support_features @ clip_prototypes.T
    K =  y_hat[0].shape[0]

    z = torch.softmax(y_hat, dim=1)   
    C1 = init_mu(K, d, z, query_features)

    dataset_name = cfg.DATA.DATASET_NAME
    max_iter = MAX_ITER_MAP.get(dataset_name, 3)
 
    alpha = 0.1
    beta = 1 - alpha

    for it in range(max_iter):

        print(f"\n========== Iter {it+1} ==========\n")
        dist_C1 = torch.cdist(query_features, C1, p=2)        # [N, K]
        dist_Y = torch.cdist(support_features, clip_prototypes, p=2)               
        tau = 0.1
        logits_C1 = -dist_C1 / tau
        logits_Y = -dist_Y / tau
        P_C1 = logits_C1
        P_Y = logits_Y

        fusion = alpha * P_C1 + beta * P_Y  # [N, K]
        print("Updated z (after Sinkhorn)")
        z = sinkhorn(fusion, epsilon=0.05, sinkhorn_iterations=50)
        acc = cls_acc(z, query_label)
        print("\n**** P accuracy: {:.2f}. ****\n".format(acc))
        C1 = update_mu(query_features, z)
        P = z
        
    acc = alignment(args, VFM_train_features, VFM_test_features, clip_prototypes,  test_label, P)  
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--device", type=str, default="cuda:4")

    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--fewshot-path", type=str, default=None)
    parser.add_argument("--model-chekpoint", type=str, default=None)

    parser.add_argument(
        "--model-type",
        type=str,
        choices=[
            "clip", "dinov2", "dinov3"
        ],
        required=True,
    )

    parser.add_argument(
        "--refinement-loss",
        type=str,
        choices=["csls", "adaptive", "contrastive", "triplet", "alignment"],
        default="alignment"
    )

    parser.add_argument("--unsupervised", action="store_true")
    parser.add_argument("--n-iters", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--cosine-end-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--n-unsup-iters", type=int, default=5)

    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--arerank-scale", type=float, default=4.0)
    parser.add_argument("--spectral-proj", action="store_true")
    parser.add_argument("--orthogonalize", action="store_true")
    parser.add_argument("--orth-beta", type=float, default=0.01)
    parser.add_argument("--pseudo-align", action="store_true")
    parser.add_argument("--beta-procrustes", type=float, default=None)

    parser.add_argument("--gaussian-noise", type=float, default=0.035)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--interpolate-features", action="store_true")
    parser.add_argument("--five-crop", action="store_true")

    parser.add_argument("--multi-dataset", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=1, help="GPU device id (default: 1)")

    arguments = parser.parse_args()

    # ========= Multi-dataset Configuration =========
    MULTI_DATASETS = {
        "ImageUCF101": 200,
        "FGVCAircraft": 1000,
        "Food101": 200,
        "OxfordFlowers": 100,
        "OxfordPets": 100,
        "Caltech101": 1000,
        "StanfordCars": 1000,
        "EuroSAT": 100,
        "DescribableTextures": 1000,
        "SUN397": 100,
        "ImageNet": 200,
    }




    DEFAULT_DATA_PATH = "/data/"
    IMAGENET_DATA_PATH = "/data/"

    def update_opts(opts, key, value):
        new_opts, skip = [], False
        for opt in opts:
            if skip:
                skip = False
                continue
            if opt == key:
                new_opts.extend([key, value])
                skip = True
            else:
                new_opts.append(opt)
        if key not in opts:
            new_opts.extend([key, value])
        return new_opts

    # ================== Result Cache ==================
    all_results = {}

    # ================== Main Execution Logic ==================
    if arguments.multi_dataset:
        for dataset_name, n_iters in MULTI_DATASETS.items():
            print("\n" + "=" * 60)
            print(f"🚀 Running dataset: {dataset_name}")
            print("=" * 60)

            # ---------- Dataset Path Handling ----------
            if dataset_name == "ImageNet":
                data_path = IMAGENET_DATA_PATH
            else:
                data_path = DEFAULT_DATA_PATH

            # ---------- Update Configuration Options ----------
            opts = arguments.opts or []
            opts = update_opts(opts, "DATA.DATASET_NAME", dataset_name)
            opts = update_opts(opts, "DATA.DATA_PATH", data_path)
            opts = update_opts(opts, "DATA.N_SHOT", "16")
            arguments.opts = opts

            arguments.n_iters = n_iters

            # ---------- Run Evaluation ----------
            cfg = get_config(arguments)
            seed_everything(cfg.RNG_SEED)

            acc = main(arguments, cfg)
            all_results[dataset_name] = acc
    else:
        cfg = get_config(arguments)
        seed_everything(cfg.RNG_SEED)
        acc = main(arguments, cfg)
        dataset_name = cfg.DATA.DATASET_NAME
        all_results[dataset_name] = acc

    # ================== Final Results Table ==================
    print("\n================== FINAL RESULTS ==================")
    print(f"{'Dataset':25s} | {'Accuracy (%)':>12s}")
    print("-" * 42)

    for name, acc in all_results.items():
        print(f"{name:25s} | {acc * 100:12.2f}")
    print("-" * 42)
