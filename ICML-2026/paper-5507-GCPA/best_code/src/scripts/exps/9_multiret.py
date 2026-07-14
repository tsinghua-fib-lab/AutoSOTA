import itertools
import json
from pathlib import Path

import hydra
import omegaconf
import torch
from latentis.data import PROJECT_ROOT
from latentis.space._base import Space

from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate.aligner import MatrixAligner, Translator
from latentis.transform.translate.functional import svd_align_state

from cycloreps.translator.gpa import GeneralizedProcrustesTranslator
from cycloreps.translator.gcca import GeneralizedCCATranslator
from scripts.exp_utils import pca_match, z, ensure_normalised
from cycloreps.utils.utils import seed_everything
import logging 

logger = logging.getLogger(__name__)


def normalize_view_name(name, full_to_short_names):
    return full_to_short_names.get(name, name)

def permute_view_in_place(space_tensors, split, view, fraction, seed=None):
    if fraction <= 0:
        return
    if not (0 < fraction <= 1):
        raise ValueError(f"permute_fraction must be in (0, 1], got {fraction}")

    x = space_tensors[split][view]
    n = x.shape[0]
    num = int(n * fraction)
    if num < 2:
        logger.info("Permutation skipped: too few samples (%d).", num)
        return

    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(int(seed))

    idx = torch.randperm(n, generator=gen)[:num]
    perm = idx[torch.randperm(num, generator=gen)]

    x_perm = x.clone()
    x_perm[idx] = x[perm]
    space_tensors[split][view] = x_perm
    logger.info(
        "Permuted %d/%d samples (%.2f) in %s for view %s",
        num,
        n,
        fraction,
        split,
        view,
    )

# sim: [N, N] similarity (higher is better)
# K can be an int or a list of ints
def hits_at_k(sim: torch.Tensor, K=(1,5,10)):
    if isinstance(K, int):
        K = [K]
    # topk over columns per row
    vals, idx = sim.topk(k=max(K), dim=1)              # idx: [N, maxK]
    # ground-truth indices (diagonal)
    gt = torch.arange(sim.size(0), device=sim.device)  # [N]
    # expand gt to compare against top-k indices
    gt = gt.unsqueeze(1).expand(-1, idx.size(1))       # [N, maxK]
    # hits matrix: True if gt in top-k list
    hits = (idx == gt)
    # cumulative: if it appears in top-k, it also counts for larger k
    cum_hits = hits.cumsum(dim=1).clamp(max=1)         # [N, maxK]
    out = {k: cum_hits[:, k-1].float().mean().item() for k in K}
    return out


# Mean Reciprocal Rank (MRR)
def mrr(sim: torch.Tensor):
    # rank of true index per row
    # argsort descending to get ranks; more efficient with topk+argwhere if needed
    ranks = torch.argsort(torch.argsort(-sim, dim=1), dim=1)  # [N,N], rank 0 is best
    gt = torch.arange(sim.size(0), device=sim.device)
    r = ranks[torch.arange(sim.size(0), device=sim.device), gt].float() + 1.0
    return (1.0 / r).mean().item()


def load_space(data_dir, dataset_name, encoder_name, split, full_to_short_names, lang=None, tensor=False):

    print('Loading space for dataset:', dataset_name, 'encoder:', encoder_name, 'split:', split)

    if lang is not None:
        path = Path(data_dir) / dataset_name / "encodings" / encoder_name.replace('/', '-') / lang / split
    else:
        path = Path(data_dir) / dataset_name / "encodings" / encoder_name.replace('/', '-') / split 
    space = Space.load_from_disk(path)

    if lang is not None:
        space.encoder_name = "LABSE"
    else:
        space.encoder_name = full_to_short_names[encoder_name]
    space.dataset_name = dataset_name

    if tensor:
        return space.as_tensor()
    else:
        return space


def compute_and_save(
    space_tensors,
    spaces,
    short_names,
    splits,
    align_cfg,
    run_name,
):
    for encoder_name, embeddings in spaces['train'].items():
        print(f'{encoder_name} has {embeddings.shape[0]} {embeddings.shape[1]}-dimensional samples')

    retrieval_metrics = {
        name: 
            {
                other_name: {
                    'initial': 0.0,
                    'pairwise': 0.0,
                    'cycle-cons-gp': 0.0,
                    'cycle-cons-gp++': 0.0,
                    'gcca': 0.0,
                }
            
            for other_name in short_names}
        for name in short_names}

    dtype = torch.double
    # Pairwise: Orthogonal
    for encoder, other_encoder in itertools.combinations(short_names, 2):
        print(f"Testing {encoder} vs {other_encoder}")

        assert spaces["test"][encoder].keys == spaces["test"][other_encoder].keys
        keys = spaces["test"][encoder].keys 

        X_A = space_tensors["train"][encoder].to(dtype)
        X_B = space_tensors["train"][other_encoder].to(dtype)

        x_A_mu = X_A.mean(0, keepdim=True)
        x_A_sig = X_A.std(0, unbiased=False, keepdim=True) + 1e-8
        x_B_mu = X_B.mean(0, keepdim=True)
        x_B_sig = X_B.std(0, unbiased=False, keepdim=True) + 1e-8

        translator_ortho_ab = Translator(
            aligner=MatrixAligner(name="ortho", align_fn_state=svd_align_state),
            x_transform=StandardScaling(),
            y_transform=StandardScaling(),
            dim_matcher=ZeroPadding(),
        )
        translator_ortho_ab.fit(x=X_A, y=X_B)

        translator_ortho_ba = Translator(
            aligner=MatrixAligner(name="ortho", align_fn_state=svd_align_state),
            x_transform=StandardScaling(),
            y_transform=StandardScaling(),
            dim_matcher=ZeroPadding(),
        )
        translator_ortho_ba.fit(x=X_B, y=X_A)

        x_A_test = space_tensors["test"][encoder].to(dtype)
        x_B_test = space_tensors["test"][other_encoder].to(dtype)

        x_A_transformed = z(
            X=translator_ortho_ab.transform(x_A_test)["x"],
            mu=x_B_mu,
            sig=x_B_sig,
        )
        x_B_transformed = z(
            X=translator_ortho_ba.transform(x_B_test)["x"],
            mu=x_A_mu,
            sig=x_A_sig,
        )

        sim_ab = ensure_normalised(x_A_transformed) @ ensure_normalised(z(x_B_test, mu=x_B_mu, sig=x_B_sig)).T
        sim_ba = ensure_normalised(x_B_transformed) @ ensure_normalised(z(x_A_test, mu=x_A_mu, sig=x_A_sig)).T
        print(sim_ab.shape)

        scores = hits_at_k(sim_ab, K=(1,5,10))
        reverse_scores = hits_at_k(sim_ba, K=(1,5,10))
        print("Hits@1/5/10:", scores)
        print("MRR:", mrr(sim_ab))
        print("Reverse Hits@1/5/10:", reverse_scores)
        print("Reverse MRR:", mrr(sim_ba))

        retrieval_metrics[encoder][other_encoder]['pairwise'] = scores[1]
        retrieval_metrics[other_encoder][encoder]['pairwise'] = reverse_scores[1]

    # Generalized Procrustes: universe
    gc_kwargs = {}
    for key in ("gc_epochs", "gc_steps_per_epoch", "gc_batch_size", "gc_lr",
                "gc_weight_decay", "gc_hidden_mult", "gc_dropout",
                "gc_val_fraction", "gc_tau", "gc_lam", "gc_val_batches",
                "gc_num_restarts"):
        if key in align_cfg.procrustes:
            gc_kwargs[key] = align_cfg.procrustes[key]
    translator_gp = GeneralizedProcrustesTranslator(
        max_iter=align_cfg.procrustes.max_iter,
        tol=align_cfg.procrustes.tol,
        device="cpu",
        gc_enabled=False,
        **gc_kwargs,
    )
    translator_gp_corrected = GeneralizedProcrustesTranslator(
        max_iter=align_cfg.procrustes.max_iter,
        tol=align_cfg.procrustes.tol,
        device="cpu",
        gc_enabled=True,
        **gc_kwargs,
    )
    gp_tensor_spaces = {split: space_tensors[split] for split in splits}
    translator_gp.fit(gp_tensor_spaces['train'])
    translator_gp_corrected.fit(gp_tensor_spaces['train'])

    for encoder, other_encoder in itertools.combinations(short_names, 2):

        print(f'Comparing {encoder} to {other_encoder} using GP in-universe')
        assert spaces["test"][encoder].keys == spaces["test"][other_encoder].keys
        keys = spaces["test"][encoder].keys 

        X_univ = translator_gp.to_universe(
        x=space_tensors['test'][encoder],
        src=encoder,
        )

        Y_univ = translator_gp.to_universe(
            x=space_tensors['test'][other_encoder],
            src=other_encoder,
        )

        sim = ensure_normalised(X_univ) @ ensure_normalised(Y_univ).T
        print(sim.shape)

        scores = hits_at_k(sim, K=(1,5,10))
        reverse_scores = hits_at_k(sim.T, K=(1,5,10))
        print("Hits@1/5/10:", scores)
        print("MRR:", mrr(sim))

        retrieval_metrics[encoder][other_encoder]['cycle-cons-gp'] = scores[1]
        retrieval_metrics[other_encoder][encoder]['cycle-cons-gp'] = reverse_scores[1]

        print(f'Comparing {encoder} to {other_encoder} using GP++ in-universe')

        X_univ = translator_gp_corrected.to_universe(
        x=space_tensors['test'][encoder],
        src=encoder,
        )

        Y_univ = translator_gp_corrected.to_universe(
            x=space_tensors['test'][other_encoder],
            src=other_encoder,
        )

        sim = ensure_normalised(X_univ) @ ensure_normalised(Y_univ).T
        scores = hits_at_k(sim, K=(1,5,10))
        reverse_scores = hits_at_k(sim.T, K=(1,5,10))
        print("Hits@1/5/10:", scores)
        print("MRR:", mrr(sim))
        retrieval_metrics[encoder][other_encoder]['cycle-cons-gp++'] = scores[1]
        retrieval_metrics[other_encoder][encoder]['cycle-cons-gp++'] = reverse_scores[1]

    # Generalized Procrustes: universe
    translator_gcca = GeneralizedCCATranslator(device="cpu")
    gcca_tensor_spaces = {split: space_tensors[split] for split in splits}
    translator_gcca.fit(gcca_tensor_spaces['train'])

    for encoder, other_encoder in itertools.combinations(short_names, 2):

        print(f'Comparing {encoder} to {other_encoder} using GCCA in-universe')
        assert spaces["test"][encoder].keys == spaces["test"][other_encoder].keys
        keys = spaces["test"][encoder].keys 

        X_univ = translator_gcca.to_universe(
        x=space_tensors['test'][encoder],
        src=encoder,
        )

        Y_univ = translator_gcca.to_universe(
            x=space_tensors['test'][other_encoder],
            src=other_encoder,
        )

        sim = ensure_normalised(X_univ) @ ensure_normalised(Y_univ).T
        print(sim.shape)

        scores = hits_at_k(sim, K=(1,5,10))
        reverse_scores = hits_at_k(sim.T, K=(1,5,10))
        print("Hits@1/5/10:", scores)
        print("MRR:", mrr(sim))

        retrieval_metrics[encoder][other_encoder]['gcca'] = scores[1]
        retrieval_metrics[other_encoder][encoder]['gcca'] = reverse_scores[1]

    out_path = PROJECT_ROOT / "results" / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(retrieval_metrics, f, indent=4)

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="9_multiret.yaml", version_base=None)
def main(cfg: omegaconf.DictConfig):
    seed_everything(int(cfg.seed) if "seed" in cfg else 42)
    dataset_name = cfg.dataset_name.split("/")[-1]
    encoder_to_lang = cfg.encoder_to_lang
    encoder_names = list(encoder_to_lang.keys())
    full_to_short_names = cfg.full_to_short_names
    full_to_short_names = {k: v for k, v in full_to_short_names.items() if k in encoder_names}
    short_names = list(full_to_short_names.values())
    splits = cfg.splits
    run_name = cfg.run_name if "run_name" in cfg else "6_mlt_results"
    align_cfg = omegaconf.OmegaConf.load(Path(PROJECT_ROOT / "config" / "alignment.yaml"))

    spaces = {split: {full_to_short_names[name]: load_space(cfg.data_dir, dataset_name, name, split, full_to_short_names) for name in encoder_names} for split in splits}
    space_tensors = {split: {full_to_short_names[name]: load_space(cfg.data_dir, dataset_name, name, split, full_to_short_names, tensor=True) for name in encoder_names} for split in splits}

    dims = set(
                [
                    space.shape[1]
                    for split_dict in spaces.values()
                    for space in split_dict.values()
                ]
            )

    if cfg.pca_enabled:
        space_tensors = pca_match(spaces=space_tensors, min_dim=min(dims))

    permute_seed = cfg.permute_seed if "permute_seed" in cfg else None
    permute_split = "train"

    if "permute_views_sets" not in cfg:
        compute_and_save(
            space_tensors=space_tensors,
            spaces=spaces,
            short_names=short_names,
            splits=splits,
            align_cfg=align_cfg,
            run_name=run_name,
        )
        return

    permute_fraction = float(cfg.permute_fraction)
    for view_group in cfg.permute_views_sets:
        if len(view_group) != 3:
            raise ValueError("Each permute_views_sets entry must have exactly three view names")
        views = [normalize_view_name(v, full_to_short_names) for v in view_group]
        for v_raw, v_norm in zip(view_group, views):
            if v_norm not in space_tensors[permute_split]:
                raise ValueError(f"Unknown permute view: {v_raw}")
        space_tensors_run = {
            split: {k: v for k, v in encs.items()} for split, encs in space_tensors.items()
        }
        for i, view in enumerate(views):
            permute_view_in_place(
                space_tensors=space_tensors_run,
                split=permute_split,
                view=view,
                fraction=permute_fraction,
                seed=permute_seed + i if permute_seed is not None else None,
            )
        out_name = f"{run_name}_{'_'.join(views)}"
        compute_and_save(
            space_tensors=space_tensors_run,
            spaces=spaces,
            short_names=short_names,
            splits=splits,
            align_cfg=align_cfg,
            run_name=out_name,
        )

if __name__ == "__main__":
    main()
