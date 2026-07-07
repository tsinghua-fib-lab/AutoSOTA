# -----------------------------------------------------------------------------#
# Full protocol for a single seed / Main CLI
# -----------------------------------------------------------------------------#
from util import set_seed
from datasets import load_dataset, make_splits_and_adjacency, audit_splits
# from models import *
from regimes import run_embedding_transfer, compute_default_prompt_consts
from train import *
from misc import *

"""
Run the full endpoint + regime transfers across vanilla and specialized models
"""
def run_protocol_once(family: str, name: str, seed: int, root: str, cfg: ProtocolConfig,
                      device: torch.device,
                      enable_prompt_transfer: bool = False,
                        prompt_consts: Optional[List[float]] = None,
                      enable_joint_baseline: bool = False,
                        joint_mode: str = "lambda_grid",
                        joint_lambdas: Optional[List[float]] = None,
                        joint_alpha: float = 0.5,
                      ) -> Dict[str, float]:
    set_seed(seed)

    data, num_features, num_classes = load_dataset(family, name, root)
    splits = make_splits_and_adjacency(data, seed, cfg)

    train_mask = splits["train_mask"]
    val_mask = splits["val_mask"]
    test_mask = splits["test_mask"]

    edge_index_train = splits["edge_index_train"]

    pos_train = splits["pos_train"]
    pos_val = splits["pos_val"]
    pos_test = splits["pos_test"]

    neg_train = splits["neg_train"]
    neg_val = splits["neg_val"]
    neg_test = splits["neg_test"]

    audit_splits(edge_index_train=edge_index_train, pos_train=pos_train, pos_val=pos_val, pos_test=pos_test,
                 neg_train=neg_train, neg_val=neg_val, neg_test=neg_test)

    # --- Endpoint F (node classifier, base) ---
    F_model, F_val_acc, F_test_acc = train_node_model(
        data=data,
        edge_index_train=edge_index_train,
        num_features=num_features,
        num_classes=num_classes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        device=device,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        epochs=cfg.node_epochs,
        patience=cfg.patience,
    )

    # --- Endpoint H (link predictor, base) ---
    H_model, H_val_auc, H_test_auc = train_link_model(
        data=data,
        edge_index_train=edge_index_train,
        num_features=num_features,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        epochs=cfg.link_epochs,
        patience=cfg.patience,
    )

    # --- Weight transfer F -> H (warm start encoder) ---
    F_encoder_state = F_model.encoder.state_dict()
    H_F2H_model, _, H_F2H_test_auc = train_link_model(
        data=data,
        edge_index_train=edge_index_train,
        num_features=num_features,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        epochs=cfg.link_epochs,
        patience=cfg.patience,
        init_encoder_state=F_encoder_state,
    )

    # --- Weight transfer H -> F (warm start encoder) ---
    H_encoder_state = H_model.encoder.state_dict()
    F_H2F_model, _, F_H2F_test_acc = train_node_model(
        data=data,
        edge_index_train=edge_index_train,
        num_features=num_features,
        num_classes=num_classes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        device=device,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        epochs=cfg.node_epochs,
        patience=cfg.patience,
        init_encoder_state=H_encoder_state,
    )

    results: Dict[str, float] = {
        "NC_base_test_acc": F_test_acc,
        "LP_base_test_auc": H_test_auc,
        "NC_H2F_test_acc": F_H2F_test_acc,
        "LP_F2H_test_auc": H_F2H_test_auc,
        # "NC_base_val_acc": F_val_acc,
        # "LP_base_val_auc": H_val_auc,
    }

    # # --- Direct barrier (endpoint + cross losses, F<->H) ---
    # barrier = compute_barrier(
    #     F_model=F_model,
    #     H_model=H_model,
    #     data=data,
    #     edge_index_train=edge_index_train,
    #     train_mask=train_mask,
    #     val_mask=val_mask,
    #     pos_val=pos_val,
    #     neg_val=neg_val,
    #     device=device,
    #     num_features=num_features,
    #     num_classes=num_classes,
    #     cfg=cfg,
    # )
    # results["Barrier_direct_FH"] = barrier

    # # --- ProbeCompat (warm-start-delta based) ---
    # probe_res = compute_probecompat(
    #     F_model=F_model,
    #     H_model=H_model,
    #     data=data,
    #     edge_index_train=edge_index_train,
    #     train_mask=train_mask,
    #     val_mask=val_mask,
    #     test_mask=test_mask,
    #     pos_train=pos_train,
    #     neg_train=neg_train,
    #     pos_val=pos_val,
    #     neg_val=neg_val,
    #     pos_test=pos_test,
    #     neg_test=neg_test,
    #     device=device,
    #     num_classes=num_classes,
    #     NC_base_test_acc=F_test_acc,
    #     LP_base_test_auc=H_test_auc,
    #     NC_H2F_test_acc=F_H2F_test_acc,
    #     LP_F2H_test_auc=H_F2H_test_auc,
    #     cfg=cfg,
    # )
    # results.update(probe_res)

    # --- Embedding transfer: F -> H (replace & concat) ---
    et_F2H_replace = run_embedding_transfer(
        mode="replace",
        source="F",
        F_model=F_model,
        H_model=H_model,
        data=data,
        edge_index_train=edge_index_train,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        cfg=cfg,
    )
    results.update(et_F2H_replace)

    # --- Embedding transfer: H -> F (replace) ---
    et_H2F_replace = run_embedding_transfer(
        mode="replace",
        source="H",
        F_model=F_model,
        H_model=H_model,
        data=data,
        edge_index_train=edge_index_train,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        cfg=cfg,
    )
    results.update(et_H2F_replace)

    # --- Embedding transfer: F -> H (concat) ---
    et_F2H_concat = run_embedding_transfer(
        mode="concat",
        source="F",
        F_model=F_model,
        H_model=H_model,
        data=data,
        edge_index_train=edge_index_train,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        cfg=cfg,
    )
    results.update(et_F2H_concat)

    # --- Embedding transfer: H -> F (concat) ---
    et_H2F_concat = run_embedding_transfer(
        mode="concat",
        source="H",
        F_model=F_model,
        H_model=H_model,
        data=data,
        edge_index_train=edge_index_train,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        cfg=cfg,
    )
    results.update(et_H2F_concat)

    # --- Prompt-transfer experiments ---
    if enable_prompt_transfer:
        # Determine prompt constants: CLI overrides automatic computation
        if prompt_consts is None:
            prompt_vals = compute_default_prompt_consts(data)
        else:
            prompt_vals = list(prompt_consts)

        x = data.x
        dev = x.device

        # Build prompt tensor directly on the same device and dtype as x
        prompt_tensor = torch.tensor(
            prompt_vals,
            dtype=x.dtype,
            device=dev,
        ).view(1, -1)  # (1, P)

        # Broadcast to all nodes and augment features
        prompt_mat = prompt_tensor.expand(x.size(0), -1)  # (N, P), on dev
        x_prompt = torch.cat([x, prompt_mat], dim=-1)     # (N, d + P), on dev
        in_ch_prompt = x_prompt.size(1)

        # NC prompt-transfer
        F_PT_model, _, F_PT_test_acc = train_node_model(
            data=data,
            edge_index_train=edge_index_train,
            num_features=num_features,
            num_classes=num_classes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            epochs=cfg.node_epochs,
            patience=cfg.patience,
            init_encoder_state=None,
            x_override=x_prompt,
            in_channels_override=in_ch_prompt,
        )

        # LP prompt-transfer
        H_PT_model, _, H_PT_test_auc = train_link_model(
            data=data,
            edge_index_train=edge_index_train,
            num_features=num_features,
            pos_train=pos_train,
            neg_train=neg_train,
            pos_val=pos_val,
            neg_val=neg_val,
            pos_test=pos_test,
            neg_test=neg_test,
            device=device,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            epochs=cfg.link_epochs,
            patience=cfg.patience,
            init_encoder_state=None,
            x_override=x_prompt,
            in_channels_override=in_ch_prompt,
        )

        results["NC_PT_test_acc"] = float(F_PT_test_acc)
        results["LP_PT_test_auc"] = float(H_PT_test_auc)

    # --- Joint training baseline  ---
    if enable_joint_baseline:
        if joint_mode == "lambda_grid":
            lambdas = joint_lambdas if joint_lambdas is not None else [0.0, 0.5, 1.0]

            joint_res = train_joint_lambda_sweep(
                data=data,
                edge_index_train=edge_index_train,
                num_features=num_features,
                num_classes=num_classes,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                pos_train=pos_train,
                neg_train=neg_train,
                pos_val=pos_val,
                neg_val=neg_val,
                pos_test=pos_test,
                neg_test=neg_test,
                device=device,
                hidden=cfg.hidden,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                epochs=2*max(cfg.node_epochs, cfg.link_epochs),
                patience=2*cfg.patience,
                lambdas=[float(x) for x in lambdas],
                alpha_val_select=float(joint_alpha),
            )

            # store with stable keys
            results["NC_joint_test_acc"] = float(joint_res["NC_joint_test_acc"])
            results["LP_joint_test_auc"] = float(joint_res["LP_joint_test_auc"])
            results["Joint_best_lambda"] = float(joint_res["Joint_best_lambda"])
            # results["Joint_val_select_score"] = float(joint_res["Joint_val_select_score"])

        elif joint_mode == "uncertainty":
            joint_res = train_joint_model(
                data=data,
                edge_index_train=edge_index_train,
                num_features=num_features,
                num_classes=num_classes,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                pos_train=pos_train,
                neg_train=neg_train,
                pos_val=pos_val,
                neg_val=neg_val,
                pos_test=pos_test,
                neg_test=neg_test,
                device=device,
                hidden=cfg.hidden,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                epochs=2*max(cfg.node_epochs, cfg.link_epochs),
                patience=2*cfg.patience,
                mode="uncertainty",
                alpha_val_select=float(joint_alpha),
            )

            results["NC_joint_test_acc"] = float(joint_res["NC_joint_test_acc"])
            results["LP_joint_test_auc"] = float(joint_res["LP_joint_test_auc"])
            # results["Joint_val_select_score"] = float(joint_res["Joint_val_select_score"])
            # # optional diagnostics
            # if "log_sigma_nc" in joint_res:
            #     results["Joint_log_sigma_nc"] = float(joint_res["log_sigma_nc"])
            # if "log_sigma_lp" in joint_res:
            #     results["Joint_log_sigma_lp"] = float(joint_res["log_sigma_lp"])

        else:
            raise ValueError(f"Unknown joint_mode: {joint_mode}")

    return results


# -----------------------------------------------------------------------------#
# Main CLI
# -----------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(description="Leakage-free Node<->Link protocol runner (extended).")

    p.add_argument("--root", type=str, required=True,
                   help="Root directory for datasets.")
    p.add_argument("--setting", type=str, choices=["transductive", "inductive"], default="transductive",
                   help="Evaluation setting for nodes.",)
    p.add_argument("--family", type=str, choices=["Planetoid", "WebKB", "Actor", "RomanEmpire", "Airports"],
                   help="Dataset family (ignored if --run_all).",)
    p.add_argument("--name", type=str,
                   help="Dataset name (e.g. Cora, Citeseer, PubMed, Texas, USA, ...).",)
    p.add_argument("--run_all", action="store_true",
                   help="Run all predefined datasets across families.",)
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="List of random seeds.")
    p.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--node_epochs", type=int, default=200)
    p.add_argument("--link_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--barrier_steps", type=int, default=11)
    p.add_argument("--probe_epochs", type=int, default=200)

    # Joint-training baseline options
    p.add_argument("--enable_joint_baseline",action="store_true",
                   help="Run joint-training baseline (shared encoder, NC+LP heads).",)
    p.add_argument("--joint_mode", type=str, choices=["lambda_grid", "uncertainty"], default="lambda_grid",
                   help="Joint baseline weighting strategy.",)
    p.add_argument("--joint_lambdas", type=float, nargs="*",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   help="Lambda grid for joint training when --joint_mode=lambda_grid.",)
    p.add_argument("--joint_alpha",type=float, default=0.5,
                   help="Validation selection score: alpha*val_acc + (1-alpha)*val_auc.",)

    # Prompt-transfer options
    p.add_argument("--enable_prompt_transfer", action="store_true",
                   help="Run prompt-transfer experiments (NC_PT, LP_PT).",)
    p.add_argument("--prompt_consts", type=float, nargs="*", default=None,
                   help=(
                       "Optional list of scalar constants for prompt-transfer "
                       "(e.g., edge/node homophily, global clustering). "
                       "If omitted and --enable_prompt_transfer is set, "
                       "defaults will be computed per dataset."),)

    return p.parse_args()

def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = ProtocolConfig(
        setting=args.setting,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        node_epochs=args.node_epochs,
        link_epochs=args.link_epochs,
        patience=args.patience,
        barrier_steps=args.barrier_steps,
        probe_epochs=args.probe_epochs,
    )

    if args.run_all:
        todo: List[Tuple[str, str]] = []

        # Planetoid
        for name in ["Cora", "Citeseer", "PubMed"]: todo.append(("Planetoid", name))

        # WebKB
        for name in ["Texas", "Cornell", "Wisconsin"]: todo.append(("WebKB", name))

        # Actor
        todo.append(("Actor", "Actor"))

        # RomanEmpire
        todo.append(("RomanEmpire", "roman-empire"))

        # Airports
        for name in ["USA", "Europe", "Brazil"]: todo.append(("Airports", name))

    else:
        if args.family is None or args.name is None:
            raise ValueError("Either use --run_all or specify --family and --name.")
        todo = [(args.family, args.name)]

    print(f"Running setting={cfg.setting} on device={device}")
    print(f"Seeds: {args.seeds}")
    print("Datasets:", todo)

    for family, name in todo:
        metrics_per_seed: Dict[str, List[float]] = {}

        print("\n=== Dataset: family={}, name={} ===".format(family, name))

        for seed in args.seeds:
            res = run_protocol_once(
                family=family,
                name=name,
                seed=seed,
                root=args.root,
                cfg=cfg,
                device=device,
                enable_prompt_transfer=args.enable_prompt_transfer,
                prompt_consts=args.prompt_consts,
                enable_joint_baseline=args.enable_joint_baseline,
                joint_mode=args.joint_mode,
                joint_lambdas=args.joint_lambdas,
                joint_alpha=args.joint_alpha,
            )

            for k, v in res.items():
                metrics_per_seed.setdefault(k, []).append(float(v))

            # # Explicit per-seed print including embedding transfer and barrier/probe/prompt
            # line = (
            #     f" Seed {seed:3d}: "
            #     f"NC_base={res['NC_base_test_acc']:.4f}, "
            #     f"LP_base={res['LP_base_test_auc']:.4f}, "
            #     f"LP_F2H_warm={res['LP_F2H_test_auc']:.4f}, "
            #     f"NC_H2F_warm={res['NC_H2F_test_acc']:.4f}, "
            #     f"LP_ET_F2H_rep={res['LP_ET_F2H_replace_test_auc']:.4f}, "
            #     f"LP_ET_F2H_cat={res['LP_ET_F2H_concat_test_auc']:.4f}, "
            #     f"NC_ET_H2F_rep={res['NC_ET_H2F_replace_test_acc']:.4f}, "
            #     f"NC_ET_H2F_cat={res['NC_ET_H2F_concat_test_acc']:.4f}, "
            #     f"Barrier={res['Barrier_direct_FH']:.4f}, "
            # )
            # if "NC_PT_test_acc" in res and "LP_PT_test_auc" in res:
            #     line += (
            #         f", NC_PT={res['NC_PT_test_acc']:.4f}, "
            #         f"LP_PT={res['LP_PT_test_auc']:.4f}"
            #     )
            # if "NC_joint_test_acc" in res and "LP_joint_test_auc" in res:
            #     if "Joint_best_lambda" in res:
            #         line += (
            #             f", Joint(NC)={res['NC_joint_test_acc']:.4f}, "
            #             f"Joint(LP)={res['LP_joint_test_auc']:.4f}, "
            #             f"lam*={res['Joint_best_lambda']:.2f}"
            #         )
            #     else:
            #         line += (
            #             f", Joint(NC)={res['NC_joint_test_acc']:.4f}, "
            #             f"Joint(LP)={res['LP_joint_test_auc']:.4f}"
            #         )
            # print(line)

        # Aggregate over seeds (this already includes embedding-transfer keys etc.)
        print(" --- Summary over seeds ---")
        for k, vals in metrics_per_seed.items():
            arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(arr)) * 100
            std = float(np.std(arr, ddof=1)) * 100 if arr.size > 1 else 0.0
            print(f" {k:26s}: {mean:.2f} ± {std:.2f}")

if __name__ == "__main__":
    main()