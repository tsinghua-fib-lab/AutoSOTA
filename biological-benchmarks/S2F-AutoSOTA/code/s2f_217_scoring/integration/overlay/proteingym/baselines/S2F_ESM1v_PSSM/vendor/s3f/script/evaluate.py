import os
import sys
import csv
import pprint
import random
import pickle

import numpy as np

from tqdm import tqdm

import torch
from torch.nn import functional as F

import torch_geometric.data     # Pre-load to bypass the hack from torchdrug
from torch_scatter import scatter_mean
from torchvision import datasets
from torchdrug import core, utils, data, metrics
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from s3f import dataset, task, model, gvp, msa_pssm


METRICS = ["spearmanr", "pearsonr", "mae", "rmse"]


def evaluate(pred, target):
    metric = {}
    for _metric in METRICS:
        if _metric == "mae":
            score = F.l1_loss(pred, target, reduction="mean")
        elif _metric == "rmse":
            score = F.mse_loss(pred, target, reduction="mean").sqrt()
        elif _metric == "spearmanr":
            score = metrics.spearmanr(pred, target)
        elif _metric == "pearsonr":
            score = metrics.pearsonr(pred, target)
        else:
            raise ValueError("Unknown metric `%s`" % _metric)
        
        metric[_metric] = score

    return metric


def graph_concat(graphs):
    if len(graphs) == 1:
        return graphs[0]
    graph = graphs[0].pack(graphs)
    if isinstance(graph, data.Protein):
        # residue graph
        _graph = data.Protein(edge_list=graph.edge_list, atom_type=graph.atom_type, bond_type=graph.bond_type, 
                                residue_type=graph.residue_type, atom_name=graph.atom_name, atom2residue=graph.atom2residue, 
                                residue_feature=graph.residue_feature, b_factor=graph.b_factor, bond_feature=None,
                                node_position=graph.node_position, num_node=graph.num_atom, num_residue=graph.num_residue,
        )
    else:
        # surface graph
        _graph = data.Graph(edge_list=graph.edge_list, num_node=graph.num_node, num_relation=1)
        with _graph.node():
            _graph.node_position = graph.node_position
            _graph.node_feature = graph.node_feature
            _graph.normals = graph.normals
    return _graph


def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0,seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0,model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0,mutation_position_relative-half_model_window), min(seq_len_wo_special,mutation_position_relative+half_model_window)]
    

def predict(cfg, task, dataset, return_components=False):
    dataloader = data.DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device(cfg.gpus[0])
    task = task.cuda(device)
    task.eval()
    seq_prob = []
    seq_prob_comps = {k: [] for k in ("s2f", "masked", "wt")} if return_components else None
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=device)
        with torch.no_grad():
            prob, sizes, comps = task.inference(batch)
        cum_sizes = sizes.cumsum(dim=0)
        for i in range(len(sizes)):
            seq_prob.append(prob[cum_sizes[i]-sizes[i]:cum_sizes[i]])
            if seq_prob_comps is not None:
                # IDEA-602: split each component's rows the same way as prob.
                for k in seq_prob_comps:
                    c = comps.get(k) if comps else None
                    seq_prob_comps[k].append(
                        None if c is None
                        else c[cum_sizes[i]-sizes[i]:cum_sizes[i]])
    if return_components:
        return seq_prob, seq_prob_comps
    return seq_prob


def _iter_group_scores(seq_prob, mutations, offsets, win_counts):
    """Yield (score_tensor, rows) per mutation-site group.

    Shared core of get_prob() and the IDEA-602 component dump: per-site
    log-prob rows are gathered and averaged across windows (IDEA-407), then
    the mutant-minus-wild-type log-odds are summed over the row's sites.
    No fusion, no alpha, no PSSM blending — pure per-component row scores.
    The fused score is an exact linear combination of these row scores, so
    weight search is an offline recombination.
    """
    groups = []
    for item in mutations:
        if groups and groups[-1][0] == item[0]:
            groups[-1][1].append(item)
        else:
            groups.append([item[0], [item]])

    p = 0
    for gi, (sites, rows) in enumerate(groups):
        k = win_counts[gi]
        windows = seq_prob[p:p + k]
        offs = offsets[p:p + k]
        p += k

        sites_t = torch.tensor(sites, dtype=torch.long, device=windows[0].device)
        n_sites = sites_t.numel()
        pos = torch.arange(n_sites, device=sites_t.device)
        # gather each window's rows for the sites it contains, then average
        # over windows per site (scatter_mean with count 1 == identity, so the
        # single-window path is bit-identical to the original scoring)
        srcs, idxs = [], []
        for w, off in zip(windows, offs):
            local = sites_t - off
            valid = (local >= 0) & (local < w.shape[0])
            if valid.any():
                srcs.append(torch.log_softmax(w, dim=-1)[local[valid]])
                idxs.append(pos[valid])
        if not srcs:
            site_logp = torch.zeros(n_sites, 20, device=sites_t.device)
        else:
            site_logp = scatter_mean(torch.cat(srcs, dim=0), torch.cat(idxs, dim=0),
                                     dim=0, dim_size=n_sites)

        site_mt = torch.tensor(
            [[data.Protein.residue_symbol2id.get(mut[-1], -1) for mut in muts]
             for _, muts, _ in rows], dtype=torch.long, device=sites_t.device)
        site_wt = torch.tensor(
            [[data.Protein.residue_symbol2id.get(mut[0], -1) for mut in muts]
             for _, muts, _ in rows], dtype=torch.long, device=sites_t.device)
        ar = pos.unsqueeze(0).expand(site_mt.shape[0], -1)
        mt_logp = site_logp[ar, site_mt]
        wt_logp = site_logp[ar, site_wt]
        score = (mt_logp - wt_logp).sum(dim=-1)
        yield score, rows


def get_prob(seq_prob, mutations, offsets, win_counts=None, alpha=0.0,
             pssm_scores=None, pssm_cov=None, pssm_weight=0.0):
    # IDEA-407: seq_prob may contain several window views per mutation-site
    # group (multi-window averaging for proteins > 1022 residues). Per-site
    # log-prob rows are averaged across windows before the mutant-vs-wild-type
    # difference. win_counts[j] = number of windows for group j.
    # IDEA-501 (calibration only): alpha > 0 divides the summed site log-odds
    # by k^alpha (k = number of mutated sites in the row). Default alpha=0
    # keeps the exact additive log-odds path used by the champion scoring.
    # IDEA-601: pssm_scores / pssm_cov carry the per-row evolutionary-expert
    # log-odds and their MSA coverage fraction (aligned with the `mutations`
    # row order). Rows whose mutated sites are ALL covered by the assay's
    # alignment get e_row = pssm_weight; the rest keep the pure model score
    # (an uncovered site must never be scored as log-odds 0).
    if win_counts is None:
        win_counts = [1] * len(seq_prob)

    preds = []
    targets = []
    r0 = 0
    for score, rows in _iter_group_scores(seq_prob, mutations, offsets, win_counts):
        if alpha > 0:
            # IDEA-501 calibration: score / k^alpha (k = # mutated sites).
            k = torch.tensor([len(muts) for _, muts, _ in rows],
                             dtype=score.dtype, device=score.device)
            score = score / k.pow(alpha)

        if pssm_scores is not None:
            # IDEA-601: blend the evolutionary expert into the row score.
            ps = torch.as_tensor(pssm_scores[r0:r0 + len(rows)],
                                 dtype=score.dtype, device=score.device)
            pc = torch.as_tensor(pssm_cov[r0:r0 + len(rows)],
                                 dtype=score.dtype, device=score.device)
            e_row = pssm_weight * (pc >= 1 - 1e-9).to(score.dtype)
            score = (1 - e_row) * score + e_row * ps
        r0 += len(rows)

        preds.append(score)
        targets.extend([target for _, _, target in rows])

    pred = torch.cat(preds)
    target = torch.tensor(targets).cuda()
    return pred, target


def load_dataset(csv_file, protein, multi_window=False, struct_range=None):
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin)
        fields = next(reader)
        mutations = []
        targets = []
        for i, values in enumerate(reader):
            for field, value in zip(fields, values):
                if field == "mutant":
                    mutations.append(value.split(":"))
                elif field == "DMS_score":
                    value = utils.literal_eval(value)
                    targets.append(value)

    def mutation_site(x):
        return [int(y[1:-1])-1 for y in x]

    mutations = [(tuple(mutation_site(mut)), mut, tar) for mut, tar in zip(mutations, targets)]
    mutations = sorted(mutations)
    sequences = []
    wt_sequences = []
    offsets = []
    win_counts = []
    for i, mut in enumerate(mutations):
        if i > 0 and mut[0] == mutations[i-1][0]:
            continue
        masked_seq = protein.clone()
        _mutation_site = mut[0]
        node_index = torch.tensor(_mutation_site, dtype=torch.long)

        # truncate long sequences and those only with substructures
        if os.path.basename(csv_file) == "POLG_HCVJF_Qi_2014.csv":
            windows = [(1981, 2225)]
        elif os.path.basename(csv_file) == "A0A140D2T1_ZIKV_Sourisseau_2019.csv":
            windows = [(290, 794)]
        elif os.path.basename(csv_file) == "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv":
            windows = [(119, 197)]       # keep high plddt part
        elif masked_seq.num_residue > 1022:
            seq_len = masked_seq.num_residue
            start, end = get_optimal_window(mutation_position_relative=mut[0][0], seq_len_wo_special=seq_len, model_window=1022)
            # IDEA-407: clamp the automatic window to the structure coverage
            # (some long proteins have partial AF2 structures, e.g. BRCA2).
            # Assays that already evaluate cleanly are unaffected: clamping
            # only moves windows that previously crashed the eval.
            if struct_range is not None:
                s0, s1 = struct_range
                win_len = end - start
                if s1 - s0 < win_len:
                    start, end = s0, s1
                else:
                    start = min(max(start, s0), s1 - win_len)
                    end = start + win_len
            windows = [(start, end)]
            # IDEA-407: shifted window views for the automatic-window long
            # proteins; averaged in log-prob space at scoring time. Hardcoded
            # author-chosen ranges above are deliberately left untouched.
            if multi_window:
                win_len = end - start
                lo = 0 if struct_range is None else struct_range[0]
                hi = seq_len - win_len
                if struct_range is not None:
                    hi = min(hi, struct_range[1] - win_len)
                shifted = []
                for d in (-256, 256):
                    s = min(max(start + d, lo), hi)
                    shifted.append((s, s + win_len))
                windows += [w for w in shifted if w != (start, end)]
        else:
            windows = [(0, masked_seq.num_residue)]

        for (start, end) in windows:
            masked_seq = protein.clone()
            node_index_w = node_index - start
            valid = (node_index_w >= 0) & (node_index_w < end - start)
            node_index_w = node_index_w[valid]
            residue_mask = torch.zeros((masked_seq.num_residue, ), dtype=torch.bool)
            residue_mask[start:end] = 1
            masked_seq = masked_seq.subresidue(residue_mask)
            with masked_seq.graph():
                masked_seq.start = torch.as_tensor(start)
                masked_seq.end = torch.as_tensor(end)
            offsets.append(start)

            # IDEA-404: keep an unmasked clone of the same window for the
            # wild-type-conditional ESM expert (mutation-independent marginal).
            wt_seq = masked_seq.clone()
            mask_id = task.model.sequence_model.alphabet.get_idx("<mask>")
            with masked_seq.residue():
                masked_seq.residue_feature[node_index_w] = 0
                masked_seq.residue_type[node_index_w] = mask_id
            sequences.append(masked_seq)
            wt_sequences.append(wt_seq)
        win_counts.append(len(windows))

    return sequences, mutations, offsets, wt_sequences, win_counts


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Output dir: %s" % working_dir)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    summary = core.Configurable.load_config_dict(cfg.summary)
    num_assay = len(summary.ids)
    # IDEA-407: multi-window averaging for proteins > 1022 residues
    # (automatic-window path only). Selected on sel-long: +0.0031 over the
    # single-window control; dev-40 0.4500 vs 0.4499. ON by default so the
    # champion state reproduces from the repo alone; S2F_MULTI_WINDOW=0
    # restores single-window scoring.
    multi_window = os.environ.get("S2F_MULTI_WINDOW", "1") == "1"
    if comm.get_rank() == 0:
        logger.warning("Multi-window scoring: %s" % multi_window)
    # IDEA-501 (calibration only): multi-site score normalization exponent.
    # 0 = additive log-odds (champion behavior); >0 divides the summed site
    # log-odds by k^alpha for k-site mutants. Gated so the default eval path
    # is bit-identical to the round-4 champion state.
    site_agg_alpha = float(os.environ.get("S2F_SITE_AGG_ALPHA", "0"))
    if comm.get_rank() == 0:
        logger.warning("Site-agg alpha: %s" % site_agg_alpha)
    # IDEA-601: weight of the evolutionary (MSA/PSSM) scoring expert, fused at
    # the row level: final = (1 - e_row) * model_score + e_row * pssm_score,
    # where e_row = e for rows fully covered by the assay's alignment and 0
    # otherwise (uncovered rows fall back to the model-only score). Selected
    # on sel-80 together with wt_esm_weight (round 6, IDEA-602).
    # IDEA-603: adaptive mode is ON by default so the champion state reproduces
    # from the repo alone (S2F_PSSM_ADAPTIVE=0 restores the flat global-e
    # weight). e is per-assay:
    #     e(neff) = sigmoid(8 * (log10(neff) - 1.47))
    # where neff is the assay's own Henikoff effective-sequence count on the
    # subMSA focus columns (tools/adaptive_e_fit.py). Strict LOO-CV on sel-80:
    # 0.5177 vs 0.5113 global-e plateau (+0.0064, guardrail +0.002 cleared).
    # dev-40 at the mapping: 0.4572 vs 0.4524 champion (+0.0048) - the largest
    # single-iteration gain of the round. Deep alignments (neff >~30) get e->1
    # (evolutionary term dominates); shallow ones get e->0 (model-only score).
    pssm_weight = float(os.environ.get("S2F_PSSM_WEIGHT", "0.25"))
    pssm_adaptive = os.environ.get("S2F_PSSM_ADAPTIVE", "1") == "1"
    if comm.get_rank() == 0:
        logger.warning("PSSM expert weight: %s (adaptive: %s)"
                       % (pssm_weight, pssm_adaptive))
    num_mutant = [int(summary.assay_dict[id]["DMS_number_single_mutants"]) for id in summary.ids]
    total_mutant = sum(num_mutant)
    if comm.get_rank() == 0:
        logger.warning("# total assays: %d, # total mutations: %d" % (num_assay, total_mutant))

    # pick the subset of DMS assays to be evaluated
    id_list = cfg.get("id_list", summary.ids)
    if "exclude_id_list" in cfg:
        exclude_id_list = set(cfg.exclude_id_list)
        id_list = [id for id in id_list if id not in exclude_id_list]

    if comm.get_rank() == 0:
        num_mutant = [int(summary.assay_dict[id]["DMS_number_single_mutants"]) for id in id_list]
        logger.warning("# assays: %d, # mutations: %d" % (len(id_list), sum(num_mutant)))

    task = core.Configurable.load_config_dict(cfg.task)
    task.preprocess(None, None, None)

    with open("results.csv", "w") as f:
        f.write("DMS_id,UniProt_ID,seq_len,DMS_number_single_mutants,%s\n" % (",".join(METRICS)))

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))['model']
        task.load_state_dict(model_dict)

        # IDEA-401: optional snapshot-ensemble members (prediction-level).
        # S2F_ENS_CKPTS is a ':'-separated list of checkpoints from the same
        # training run. The frozen ESM branch is shared, so each member only
        # contributes its GVP encoder + 20-way head weights; log-probs are
        # averaged across members in task.inference().
        ens_paths = [p for p in os.environ.get("S2F_ENS_CKPTS", "").split(":")
                     if p]
        for ens_path in ens_paths:
            ens_path = os.path.expanduser(ens_path)
            if comm.get_rank() == 0:
                logger.warning("Ensemble member: %s" % ens_path)
            ens_sd = torch.load(ens_path, map_location=torch.device('cpu'))['model']
            member_model = core.Configurable.load_config_dict(
                cfg.task["model"]["structure_model"])
            prefix = "model.structure_model."
            member_sd = {k[len(prefix):]: v for k, v in ens_sd.items()
                         if k.startswith(prefix)}
            member_model.load_state_dict(member_sd)
            member_head = torch.nn.Linear(task.linear.in_features,
                                          task.linear.out_features)
            member_head.load_state_dict({"weight": ens_sd["linear.weight"],
                                         "bias": ens_sd["linear.bias"]})
            task.add_ensemble_member(member_model, member_head)

        # IDEA-403/412: convex per-member weights [main, member1, ...]
        # (normalized to sum 1) and an esm_weight override for sweep runs.
        ens_weights = os.environ.get("S2F_ENS_WEIGHTS", "")
        if ens_weights:
            weights = [float(x) for x in ens_weights.split(",") if x]
            if len(weights) != len(task.ens_models) + 1:
                raise ValueError("S2F_ENS_WEIGHTS must have 1 + #members entries")
            task.set_ensemble_weights(weights)
            if comm.get_rank() == 0:
                logger.warning("Ensemble weights: %s" % weights)
        esm_weight = os.environ.get("S2F_ESM_WEIGHT", "")
        if esm_weight:
            task.esm_weight = float(esm_weight)
            if comm.get_rank() == 0:
                logger.warning("esm_weight override: %s" % esm_weight)
        wt_esm_weight = os.environ.get("S2F_WT_ESM_WEIGHT", "")
        if wt_esm_weight:
            task.wt_esm_weight = float(wt_esm_weight)
            if comm.get_rank() == 0:
                logger.warning("wt_esm_weight override: %s" % wt_esm_weight)
        # IDEA-606: optionally swap the SCORING sequence backbone (masked-ESM
        # + WT-ESM branches) to ESM-1v or ESM-2-3B. The GVP's node features
        # still come from the trained ESM-2-650M inside FusionNetwork; only
        # the scoring expert's logits change. Weights load from the same
        # directory as the config's sequence model.
        # IDEA-608: ESM-1v-ens5 averages the five released ESM-1v models at
        # the log-probability level (the published ESM-1v ensemble recipe).
        # Default is the 5-model ensemble so the champion state reproduces
        # from the repo alone; S2F_SCORING_ESM=none disables the swap,
        # ESM-1v / ESM-2-3B select a single-model scoring backbone.
        scoring_esm = os.environ.get("S2F_SCORING_ESM", "ESM-1v-ens5")
        if scoring_esm == "none":
            scoring_esm = ""
        if scoring_esm:
            if scoring_esm not in ("ESM-1v", "ESM-2-3B", "ESM-1v-ens5"):
                raise ValueError(
                    "S2F_SCORING_ESM must be ESM-1v, ESM-2-3B or ESM-1v-ens5")
            seq_path = cfg.task["model"]["sequence_model"]["path"]
            if scoring_esm == "ESM-1v-ens5":
                task.scoring_esm = core.Configurable.load_config_dict({
                    "class": "ESM1vEnsemble",
                    "path": seq_path,
                    "model": "ESM-1v",
                })
            else:
                task.scoring_esm = core.Configurable.load_config_dict({
                    "class": "MyESM",
                    "path": seq_path,
                    "model": scoring_esm,
                })
            if comm.get_rank() == 0:
                logger.warning("Scoring sequence backbone swapped: %s"
                               % scoring_esm)
    
    assay_result = {}
    for i, id in enumerate(id_list):
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Start evaluation on DMS assay %s" % id)
        
        # wild-type sequence
        sequence = summary.assay_dict[id]["target_seq"]
        protein = data.Protein.from_sequence(sequence, atom_feature=None, bond_feature=None)
        protein.view = "residue"

        # wild-type structure
        wild_types = []
        for _pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
            pdb_file = os.path.join(os.path.expanduser(cfg.structure_path), _pdb_file)
            wild_type = dataset.bio_load_pdb(pdb_file)[0]
            ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
            wild_type = wild_type.subgraph(ca_index)
            wild_types.append(wild_type)
        wild_type = graph_concat(wild_types)
        wild_type.view = "residue"

        # pdb range
        pdb_range = summary.assay_dict[id]["pdb_range"].split("-")
        start, end = int(pdb_range[0])-1, int(pdb_range[-1])
        with wild_type.graph():
            wild_type.start = torch.as_tensor(start)
            wild_type.end = torch.as_tensor(end)

        # surface graph
        if cfg.get("surface_path"):
            surf_graphs = []
            res2surfs = []
            for pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
                surf_graph_file = pdb_file.split(".")[0] + ".pkl"
                surf_graph_file = os.path.join(os.path.expanduser(cfg.surface_path), surf_graph_file)
                with open(surf_graph_file, "rb") as fin:
                    surf_dict = pickle.load(fin)
                surf_graph = dataset.load_surface(surf_dict)
                surf_graphs.append(surf_graph)
                res2surfs.append(torch.as_tensor(surf_dict["res2surf"]))
            surf_graph = graph_concat(surf_graphs)
            res2surf = torch.cat(res2surfs, dim=0)
            with wild_type.residue():
                wild_type.res2surf = res2surf
        else:
            surf_graph = None

        # mutants
        DMS_filename = summary.assay_dict[id]["DMS_filename"]
        csv_file = os.path.join(summary.path, DMS_filename)
        struct_range = (start, end)  # pdb_range, already loaded above
        masked_sequences, mutations, offsets, wt_sequences, win_counts = \
            load_dataset(csv_file, protein, multi_window=multi_window,
                         struct_range=struct_range)
        if comm.get_rank() == 0:
            logger.warning("Number of masked sequences: %d" % len(masked_sequences))
            logger.warning("Number of mutations: %d" % len(mutations))

        # IDEA-601: per-row evolutionary-expert log-odds from the assay's MSA
        # (weighted PSSM, cached per assay). None when disabled or unavailable.
        pssm_scores = pssm_cov = None
        e_assay = pssm_weight
        if pssm_weight or pssm_adaptive:
            msa_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dataset", "DMS_msa_files")
            try:
                positions, logf, neff = msa_pssm.build_pssm(
                    summary.assay_dict[id], msa_root)
            except ValueError as e:
                logger.warning("PSSM build failed for %s: %s" % (id, e))
                positions = None
            if positions is not None:
                pssm_scores, pssm_cov = msa_pssm.row_scores(
                    mutations, positions, logf)
                if pssm_adaptive:
                    # IDEA-603: per-assay evolutionary weight from the subMSA
                    # effective-sequence count (see module-level comment).
                    e_assay = float(np.clip(
                        1.0 / (1.0 + np.exp(-8.0 * (np.log10(neff) - 1.47))),
                        0.0, 1.0))
                if comm.get_rank() == 0:
                    logger.warning("PSSM coverage: %.1f%% of rows fully covered"
                                   % (100.0 * (pssm_cov == 1.0).mean()))
                    if pssm_adaptive:
                        logger.warning("PSSM adaptive weight: neff=%.1f -> e=%.3f"
                                       % (neff, e_assay))

        _dataset = dataset.MutantDataset(masked_sequences, wild_type, surf_graph=surf_graph,
                                         wt_sequences=wt_sequences)
        # IDEA-602: component dump for offline weight-simplex search. The dump
        # only ADDS per-component row scores; the fused pred/target path above
        # is untouched, so a dump run's measured Spearman is still valid.
        dump_dir = os.environ.get("S2F_DUMP_COMPONENTS", "")
        seq_prob_comps = None
        if dump_dir:
            task.dump_components = True
            seq_prob, seq_prob_comps = predict(cfg, task, _dataset,
                                               return_components=True)
        else:
            seq_prob = predict(cfg, task, _dataset)
        pred, target = get_prob(seq_prob, mutations, offsets, win_counts,
                                alpha=site_agg_alpha,
                                pssm_scores=pssm_scores, pssm_cov=pssm_cov,
                                pssm_weight=e_assay)
        result = evaluate(pred, target)

        if dump_dir and comm.get_rank() == 0:
            if pssm_scores is None:
                logger.warning("S2F_DUMP_COMPONENTS: no PSSM rows for %s - "
                               "skipping dump (offline search needs them)" % id)
            else:
                comp_rows = {}
                for name in ("s2f", "masked", "wt"):
                    views = seq_prob_comps[name]
                    if any(v is None for v in views):
                        logger.warning("S2F_DUMP_COMPONENTS: missing '%s' "
                                       "component for %s - skipping dump"
                                       % (name, id))
                        comp_rows = None
                        break
                    parts = [score for score, _ in
                             _iter_group_scores(views, mutations, offsets,
                                                win_counts)]
                    comp_rows[name] = torch.cat(parts).cpu().numpy()
                if comp_rows is not None:
                    comp_rows["pssm"] = pssm_scores
                    comp_rows["cov"] = pssm_cov
                    comp_rows["target"] = target.cpu().numpy()
                    os.makedirs(dump_dir, exist_ok=True)
                    np.savez_compressed(os.path.join(dump_dir, id + ".npz"),
                                        **comp_rows)
                    logger.warning("S2F_DUMP_COMPONENTS: dumped %d rows for %s"
                                   % (len(target), id))

        if comm.get_rank() == 0:
            with open("%s" % DMS_filename, "w") as f:
                f.write(",mutant,mutated_sequence,DMS_score,model_score\n")
                pred = pred.cpu().numpy()
                target = target.cpu().numpy()
                for i in range(len(mutations)):
                    f.write(",%s,,%.6f,%.6f\n" % (":".join(mutations[i][1]), target[i], pred[i]))
            logger.warning(pretty.separator)
            logger.warning("Test results")
            logger.warning(pretty.line)
            logger.warning(pprint.pformat(result))
            with open("results.csv", "a") as f:
                f.write("%s,%s,%s,%s,%s\n" % (
                    id, 
                    summary.assay_dict[id]["UniProt_ID"], summary.assay_dict[id]["seq_len"],
                    summary.assay_dict[id]["DMS_number_single_mutants"],
                    ",".join([
                        "%.3f" % result[_metric] for _metric in METRICS
                    ])
                ))
        assay_result[id] = result

    if comm.get_rank() == 0:
        logger.warning(pretty.separator)
        logger.warning("Average results on all assays")
        logger.warning(pretty.line)
        logger.warning(pprint.pformat(utils.mean(utils.stack(list(assay_result.values())))))