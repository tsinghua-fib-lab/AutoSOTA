import logging
import os
import shutil
import sys
from copy import deepcopy
from typing import Dict, Any, Union

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch import nn
from torch.utils.data import DataLoader

from .data_processing import FEDataset, FEDatasetStructure, extract_features_from_pdb
from .models import Scheduler, SA_DiT, SA_DiT_Original
from .training import DiffusionModelTrainer, BaseTrainer
from .evaluation import *
from .analysis import *

dataset = {'FE': FEDataset, 'FEStructure': FEDatasetStructure}
default_peak_params = {'height': 0, 'distance': 50, 'prominence': 0.02}

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file. Returns a tuple (data_cfg, model_cfg, train_cfg).
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file {config_path}: {e}")
        sys.exit(1)

def creat_dataset(data_cfg: Dict[str, Any], feature_first : bool = True) -> FEDataset:
    # --- Data Loading and Preparation ---
    data_paths = data_cfg.get('data_paths', {})
    preprocessing_params = data_cfg.get('data_preprocessing', {})
    dataset_type = data_cfg.get('dataset_type', 'FE')
    dataset_type = dataset.get(dataset_type)
    lazy = data_cfg.get('lazy', False)
    lazy_res_map = data_cfg.get('lazy_res_map', lazy)
    lazy_esm_seq = data_cfg.get('lazy_esm_seq', False)

    processed_data_dir = data_paths.get('processed_data_dir')
    raw_data_dir = data_paths.get('raw_data_dir')
    fe_curves_file = data_paths.get('processed_fe_curves_file', 'fe_curves.npy')
    sequences_file = data_paths.get('processed_sequences_file', 'sequences.npy')
    conditions_file = data_paths.get('processed_conditions_file', 'conditions.npy')  # Or .csv
    data_table_path = data_paths.get('data_table_file', 'data_table.csv')
    pickle_path = data_paths.get('pickle_path', '')


    fe_curves_path = os.path.join(processed_data_dir, fe_curves_file)
    sequences_path = os.path.join(processed_data_dir, sequences_file) if not lazy else os.path.join(raw_data_dir, sequences_file)
    conditions_path = os.path.join(processed_data_dir, conditions_file)
    data_table_path = os.path.join(processed_data_dir, data_table_path)
    pickle_path = os.path.join(processed_data_dir, pickle_path)


    # Determine condition_input_dim from data config
    condition_columns = preprocessing_params.get('condition_columns')
    if condition_columns is None:
        logging.error("condition_columns not specified in data_config.data_preprocessing.")
        sys.exit(1)

    full_dataset = dataset_type(
        fe_curves_path=fe_curves_path,
        sequences_path=sequences_path,
        conditions_path=conditions_path,
        data_table_path=data_table_path,
        feature_first=feature_first,
        lazy=lazy,
        lazy_res_map=lazy_res_map,
        lazy_esm_seq=lazy_esm_seq,
        pickle_path=pickle_path,
    )

    return full_dataset


def split_dataset(full_dataset: FEDatasetStructure,
                  batch_size: int,
                  train_ratio: list = None,
                  num_workers: int = 0,
                  seed: int = 42) -> tuple:
    # split dataset
    if train_ratio is None:
        train_ratio = [0.8, 0.1, 0.1]
    test_dataloader = None

    #collate_fn = full_dataset.collate_fn_for_lazy_mode if hasattr(full_dataset, 'collate_fn_for_lazy_mode') else None
    collate_fn = None
    if len(train_ratio) <= 2:
        train_dataset, val_dataset = (torch.utils.data.random_split
                                      (full_dataset, train_ratio,generator=torch.Generator().manual_seed(seed)))
    else:
        """
        train_dataset, val_dataset, test_dataset = (
            torch.utils.data.random_split(full_dataset, train_ratio, generator=torch.Generator().manual_seed(seed)))
        """
        train_dataset, val_dataset, test_dataset, _ = (
            full_dataset.split_by_pdb_id(ratios=train_ratio, seed=seed, shuffle=True))

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

    logging.info(f"Dataset split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Create DataLoaders
    persistent_workers = num_workers > 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=persistent_workers,
        num_workers=num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    if test_dataloader is None:
        test_dataloader = val_dataloader

    return train_dataloader, val_dataloader, test_dataloader


def create_model(model_cfg: Dict[str, Any]) -> nn.Module:
    model_type = model_cfg.get('model_type')
    feature_encoder_config = model_cfg.get('feature_encoder_config', {})

    if model_type == 'DiT':
        model = SA_DiT_Original(
            protein_embedding_dim=model_cfg["protein_embed_dim"],
            res_feature_dim=model_cfg["res_feature_dim"],
            contact_channels=model_cfg["contact_channels"],
            seq_len=model_cfg["seq_length"],
            patch_size=model_cfg["patch_size"],
            in_channels=model_cfg["in_channels"],
            hidden_size=model_cfg["hidden_size"],
            num_heads=model_cfg["num_heads"],
            depth=model_cfg["depth"],
            feature_encoder=model_cfg["feature_encoder"],
            learn_sigma=model_cfg["learn_sigma"],
            feature_encoder_config=model_cfg.get("feature_encoder_config", {}),
        )


    elif model_type == 'SA_DiT':
        model = SA_DiT(
            protein_embedding_dim=model_cfg["protein_embed_dim"],
            res_feature_dim=model_cfg["res_feature_dim"],
            contact_channels=model_cfg["contact_channels"],
            seq_len=model_cfg["seq_length"],
            patch_size=model_cfg["patch_size"],
            in_channels=model_cfg["in_channels"],
            hidden_size=model_cfg["hidden_size"],
            num_heads=model_cfg["num_heads"],
            depth=model_cfg["depth"],
            feature_encoder=model_cfg["feature_encoder"],
            learn_sigma=model_cfg["learn_sigma"],
            feature_encoder_config=model_cfg.get("feature_encoder_config", {}),
        )


    else:
        raise ValueError(f"model_type {model_type} not recognized.")

    all_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {all_params}")

    return model


def create_diffusion_trainer(model_cfg: Dict[str, Any],
                             train_cfg: Dict[str, Any],
                             train_dataloader: DataLoader,
                             val_dataloader: DataLoader,
                             test_dataloader: DataLoader,
                             ) -> DiffusionModelTrainer:
    # --- Model Initialization ---
    logging.info("Initializing model...")
    model = create_model(model_cfg)

    pred_sigma = model_cfg.get("pred_sigma", True)
    learn_sigma = model_cfg.get("learn_sigma", True) and pred_sigma

    # --- Diffusion Setup ---
    diffusion = Scheduler(timestep_respacing=model_cfg["timestep_respacing"],
                          noise_schedule=model_cfg["noise_schedule"],
                          diffusion_steps=model_cfg["diffusion_steps"],
                          predict_flow_v=model_cfg["predict_flow_v"],
                          learn_sigma=learn_sigma,
                          pred_sigma=pred_sigma,
                          snr=model_cfg["snr_loss"],
                          flow_shift=model_cfg["flow_shift"],
                          )

    logging.info("Model initialized.")

    # --- Trainer Setup ---
    logging.info("Setting up trainer...")
    device = torch.device(train_cfg.get('device', 'cpu'))

    trainer = DiffusionModelTrainer(
        model=model,
        diffusion=diffusion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer_config=train_cfg['optimizer'],
        scheduler_config=train_cfg['scheduler'],
        loss_config=train_cfg['loss'],
        epochs=train_cfg['epochs'],
        device=device,
        log_dir=train_cfg.get('log_dir', 'runs/'),
        checkpoint_dir=train_cfg.get('checkpoint_dir', 'checkpoints/'),
        save_interval=train_cfg.get('save_interval', 10),
        eval_interval=train_cfg.get('eval_interval', 5),
        use_ema=train_cfg.get('use_ema', True),
        model_keys=model_cfg.get('model_keys', []),
        patience=train_cfg.get('patience', float('inf')),
    )
    logging.info("Trainer setup complete.")

    return trainer



def create_trainer(model_cfg: Dict[str, Any],
                   train_cfg: Dict[str, Any],
                   train_dataloader: DataLoader,
                   val_dataloader: DataLoader,
                   test_dataloader: DataLoader,
                   ):
    trainer_type = train_cfg.get('trainer_type')
    if trainer_type == 'diffusion':
        return create_diffusion_trainer(model_cfg, train_cfg, train_dataloader, val_dataloader, test_dataloader)
    else:
        raise ValueError(f"trainer_type {trainer_type} not recognized.")


def load_solver_from_checkpoint(config_path: str, checkpoint_path: str) -> DiffusionModelTrainer:
    # Load configs
    config = load_config(config_path)
    data_cfg = config['data_params']
    model_cfg = config['model_params']
    train_cfg = config['train_params']

    # Create dataset and dataloaders
    full_dataset = creat_dataset(data_cfg, feature_first=True)

    train_dataloader, val_dataloader, test_dataloader = split_dataset(full_dataset,
                                                                      batch_size=train_cfg['batch_size'],
                                                                      seed=train_cfg['seed'],
                                                                      num_workers=train_cfg['num_workers'],)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint file not found at: {checkpoint_path}")

    logging.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Instantiate the solver using the provided configuration
    try:
        solver = create_diffusion_trainer(model_cfg, train_cfg, train_dataloader, val_dataloader, test_dataloader)
    except KeyError as e:
        logging.error(f"Missing key in model_config: {e}. Cannot instantiate solver.")
        raise
    except Exception as e:
        logging.error(f"Error instantiating model from config: {e}")
        raise

    # Load solver state from checkpoint
    solver.load_checkpoint(checkpoint_path)

    return solver


# -----------------------------------------------------------------------------
# Training based on the given config
# -----------------------------------------------------------------------------

def train_pipeline(config_path: str, checkpoint_path: str = None, evaluate: bool = True, seed: int = 42) -> DiffusionModelTrainer:
    """
    Main function to set up and run the training process.
    """

    # Load configs
    config = load_config(config_path)
    data_cfg = config['data_params']
    model_cfg = config['model_params']
    train_cfg = config['train_params']

    seed_everything(seed)

    # Create dataset and dataloaders
    full_dataset = creat_dataset(data_cfg, feature_first=True)

    train_dataloader, val_dataloader, test_dataloader = split_dataset(full_dataset,
                                                                      batch_size=train_cfg['batch_size'],
                                                                      train_ratio=train_cfg['train_ratio'],
                                                                      seed=train_cfg['seed'],
                                                                      num_workers=train_cfg['num_workers'],)

    # --- Trainer Initialization ---
    trainer = create_trainer(model_cfg, train_cfg, train_dataloader, val_dataloader, test_dataloader)



    # --- Resume Training if Checkpoint Provided ---
    start_epoch = 1
    if checkpoint_path:
        saved_path = os.path.dirname(checkpoint_path)
        logging.info(f"Attempting to load checkpoint from {checkpoint_path}")
        try:
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            logging.info(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            logging.warning(f"Checkpoint file not found at {checkpoint_path}. Starting training from epoch 1.")
            start_epoch = 1
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            sys.exit(1)
    else:
        # Copy config
        saved_path = os.path.join(trainer.checkpoint_dir, trainer.__class__.__name__, trainer.label)
        copied_config_path = os.path.join(saved_path, 'config.yaml')
        os.makedirs(os.path.dirname(copied_config_path), exist_ok=True)
        shutil.copy(config_path, copied_config_path)

    # --- Start Training ---
    logging.info("Starting training process...")
    # Modify the trainer's train method to accept a starting epoch if implementing resume
    # For simplicity in this example, we'll assume train always starts from 1 or loaded epoch
    # trainer.train(start_epoch=start_epoch) # You might modify Trainer.train()
    trainer.train()  # Current Trainer.train() starts from 1, load_checkpoint updates state

    logging.info("Training script finished.")

    # --- Start Evaluating ---
    if evaluate:
        logging.info("Starting evaluation process...")

        # Generate predicted curves
        trainer.test(num_samples_per_input=1, use_ddim=True, eta=0.0) #TODO Using sample parameters from config

        # Evaluate performance
        evaluate_metrics(saved_path)

        # Plotting results
        #peak_params = data_cfg.get('analysis_params', {}).get('find_peaks', {})
        plotting(saved_path)

    return trainer


# -----------------------------------------------------------------------------
# Optuna objective
# -----------------------------------------------------------------------------

def get_default_pruner() -> optuna.pruners.BasePruner:
    # Skip first 2 trials, then start median pruning after 1st completed step
    return MedianPruner(n_startup_trials=2, n_warmup_steps=1)


def objective(trial: optuna.trial.Trial, cfg_path: str) -> float:
    # ------------------------------------------------------------------
    # 1. Load and mutate config
    # ------------------------------------------------------------------
    base_cfg: Dict[str, Any] = load_config(cfg_path)
    data_cfg = deepcopy(base_cfg["data_params"])
    model_cfg = deepcopy(base_cfg["model_params"])
    train_cfg = deepcopy(base_cfg["train_params"])

    # --- hyper‑parameters to tune -------------------------------------
    # Optimizer
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    train_cfg["optimizer"]["lr"] = lr

    # Model size
    hidden_size = trial.suggest_categorical("hidden_size", [256, 384, 512, 768, 1024])
    model_cfg["hidden_size"] = hidden_size
    model_cfg["num_heads"] = hidden_size // 64  # keep heads proportional

    # Depth (layers)
    depth = trial.suggest_int("depth", 2, 8)
    model_cfg["depth"] = depth

    # Batch size – impacts memory, keep multiples of 8
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    train_cfg["batch_size"] = batch_size

    # Epochs per trial kept small for speed
    train_cfg["epochs"] = 50
    train_cfg["eval_interval"] = 5  # evaluate at end only
    train_cfg["checkpoint_dir"] = os.path.join(train_cfg["checkpoint_dir"] + f'trial_{trial.number}')
    train_cfg["log_dir"] = os.path.join(train_cfg["log_dir"] + f'trial_{trial.number}')

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    full_ds = creat_dataset(data_cfg, feature_first=True, lazy=data_cfg.get("lazy", False))
    train_dl, val_dl, test_dl = split_dataset(
        full_ds,
        batch_size=batch_size,
        train_ratio=train_cfg.get("train_ratio", [0.8, 0.1, 0.1]),
        seed=train_cfg.get("seed", 42),
    )

    # ------------------------------------------------------------------
    # 3. Trainer
    # ------------------------------------------------------------------
    trainer = create_diffusion_trainer(model_cfg, train_cfg, train_dl, val_dl, test_dl)

    # Light training
    trainer.train()

    # ------------------------------------------------------------------
    # 4. Validation metric (FID) – lower is better
    # ------------------------------------------------------------------
    #metrics = trainer.test(num_samples_per_input=1)  # assume function exists
    #kid = metrics["kid_mean"]
    val_loss = trainer.best_val_loss

    # Report intermediate score so pruner can act
    trial.report(val_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss


# -----------------------------------------------------------------------------
# Main entry‑point
# -----------------------------------------------------------------------------

def turning(config: str, trials: int = 30, study_name: str = 'gen_smfs_opt', storage=None) -> None:
    """
    :param config: Path to YAML config
    :param trials: Number of Optuna trials
    :param study_name: Optuna study name
    :param storage: Optuna storage URL (e.g. sqlite:///study.db)
    :return:
    """
    seed_everything()

    sampler = TPESampler(seed=42)
    pruner = get_default_pruner()

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # we minimise FID
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(lambda t: objective(t, config), n_trials=trials)

    print("\n==== Optuna summary ====")
    print("Best trial #{} => KID {:.4f}".format(study.best_trial.number, study.best_value))
    print("Params: ")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Optional: save study
    if storage is None:
        study_path = f"{study_name}_result.pkl"
        with open(study_path, "wb") as f:
            import pickle
            pickle.dump(study, f)
        print(f"Study saved to {study_path}")

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_metrics(path: str):
    # load curves
    true_curves, generated_curves = (np.load(os.path.join(path, 'true_curves.npy'), allow_pickle=True),
                                     np.load(os.path.join(path, 'generated_curves.npy'), allow_pickle=True))
    print("The shape of tested curves: ", true_curves.shape)
    generated_curves = np.clip(generated_curves, 0, 1)

    print("\n--- Curve Shape Metrics ---")
    r2_score_curves = calculate_r2(true_curves, generated_curves)
    print(f"Overall R^2 for curve shapes: {r2_score_curves:.4f}")

    rel_l2_error = calculate_relative_l2_error(true_curves, generated_curves)
    print(f"Average Relative L2 Error: {rel_l2_error:.4f}")

    fid = compute_fid(true_curves, generated_curves)
    print(f"fid: {fid:.4f}")

    #score = compute_discriminative_score(true_curves, generated_curves)
    #print(f"discriminative_score: {score:.4f}")

    correlation_score = compute_acf_score(true_curves, generated_curves)
    print(f"acf_score: {correlation_score:.4f}")

    # --- Test Mechanical Property Evaluation ---
    print("\n--- Mechanical Property Evaluation ---")
    # Need parameters for peak finding for 'num_peaks', 'avg_unfolding_force'
    peak_params = {'height': 0, 'distance': 50, 'prominence': 0.02}
    property_evaluation_metrics = evaluate_mechanical_properties(
        true_curves,
        generated_curves,
        property_extraction_params={'find_peaks': peak_params}
    )

    metrics_list = {}
    # Append top-level metrics
    metrics_list['dir'] = path.split(os.sep)[-1]
    metrics_list['r2_score_curves'] = r2_score_curves
    metrics_list['rel_l2_error'] = rel_l2_error
    metrics_list['fid'] = fid

    print("\nMechanical Property Evaluation Metrics:")
    for prop_name, metrics in property_evaluation_metrics.items():
        print(f"Property: {prop_name}")
        for metric_name, value in metrics.items():
            try:
                print(f"  {metric_name}: {value:.4f}")
                full_metric_name = f"{prop_name}_{metric_name}"
                metrics_list[f"{full_metric_name}"] = value
            except:
                pass

    # Create DataFrame and save to CSV
    filename = os.path.join(path, 'metrics_list.csv')
    metrics_df = pd.DataFrame([metrics_list])
    metrics_df.to_csv(filename, index=False)
    print(f"Metrics have been saved to '{filename}'")

    return metrics_df


def plotting(path: str, peak_params=None):
    true_curves, generated_curves = (np.load(os.path.join(path, 'true_curves.npy'), allow_pickle=True),
                                     np.load(os.path.join(path, 'generated_curves.npy'), allow_pickle=True))
    generated_curves = np.clip(generated_curves, 0, 1)
    save_path = os.path.join(path, 'figs')
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    if peak_params is None:
        peak_params = {'height': 0, 'distance': 50, 'prominence': 0.02}

    # --- Test plot_fe_curve_comparison ---
    print("\n--- Testing plot_fe_curve_comparison ---")
    for i in range(5, 10):
        sample_to_plot = i  # Plot the 6th sample
        extension = np.arange(0, len(true_curves[0].reshape(-1)))
        plot_fe_curve_comparison(
            true_curves[sample_to_plot, :, 0],  # Pass 1D arrays
            generated_curves[sample_to_plot, :, 0],
            sample_idx=sample_to_plot,
            extension_axis=extension,  # Plot with physical extension
            show_peaks=True,  # Also try plotting peaks
            peak_params=peak_params,  # Peak finding parameters
            save_path=save_path
        )

    # --- Test plot_multiple_generated_curves ---
    print("\n--- Testing plot_multiple_generated_curves ---")
    # Assume the first 10 generated curves are for the same input
    plot_multiple_generated_curves(
        generated_curves,
        true_curve_avg=np.mean(true_curves[:, :, 0], axis=0),  # Plot average true curve
        extension_axis=extension,  # Plot with physical extension,
        save_path=save_path
    )

    # --- Test plot_property_distributions ---
    print("\n--- Testing plot_property_distributions ---")
    # First, extract properties from the dummy curves
    peak_params_for_plotting = peak_params
    true_properties_dict = {'unfolding_energy': [], 'max_force': [], 'num_peaks': [],
                            'avg_unfolding_force': []}
    gen_properties_dict = {'unfolding_energy': [], 'max_force': [], 'num_peaks': [],
                           'avg_unfolding_force': []}

    for i in range(len(true_curves)):
        true_curve_1d = true_curves[i, :, 0]
        gen_curve_1d = generated_curves[i, :, 0]

        # Assume dummy extension step 1.0 for energy calculation in this test
        true_properties_dict['unfolding_energy'].append(calculate_unfolding_energy(true_curve_1d, extension_step=1.0))
        true_peak_indices, _ = find_force_peaks(true_curve_1d, **peak_params_for_plotting)
        true_properties_dict['max_force'].append(calculate_max_force(true_curve_1d, true_peak_indices))
        true_properties_dict['num_peaks'].append(len(true_peak_indices))
        true_unfolding_forces = _.get('peak_heights', [])
        true_properties_dict['avg_unfolding_force'].append(
            np.mean(true_unfolding_forces) if len(true_unfolding_forces) > 0 else np.nan)

        gen_properties_dict['unfolding_energy'].append(calculate_unfolding_energy(gen_curve_1d, extension_step=1.0))
        # gen_properties_dict['max_force'].append(max(gen_curve_1d))
        gen_peak_indices, _ = find_force_peaks(gen_curve_1d, **peak_params_for_plotting)
        gen_properties_dict['max_force'].append(calculate_max_force(gen_curve_1d, gen_peak_indices))

    # Now plot distributions for specific properties
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'max_force', save_path=save_path)
    plot_property_distributions(true_properties_dict, gen_properties_dict, 'num_peaks', save_path=save_path)



# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def curve_prediction(pretrained_model_path: str,
                     pdb_id_or_path: str,
                     chain: str,
                     res_start_idx: int = None,
                     res_end_idx: int = None,
                     feature_path: str = './',
                     save_path: str = None,
                     device: str = "cpu",
                     num_samples: int = 100,
                     eta: float = 0.0,
                     parameters: dict = None,
                     overwrite: bool = False,):
    if os.path.exists(save_path) and not overwrite:
        gen_curves = np.load(save_path)

    else:
        if parameters is None:
            parameters = {}

        # Extract protein features from .pdb file
        features = extract_features_from_pdb(pdb_id_or_path=pdb_id_or_path,
                                             chain=chain,
                                             feature_path=feature_path,
                                             device=device,
                                             **parameters)

        # Load pretrained model
        config_path = os.path.join(pretrained_model_path, 'config.yaml')
        checkpoint_path = os.path.join(pretrained_model_path, 'best_model.pt')
        solver = load_solver_from_checkpoint(config_path, checkpoint_path)

        # Generate predictions
        gen_curves = solver.predict(features,
                                    chain,
                                    res_start_idx=res_start_idx,
                                    res_end_idx=res_end_idx,
                                    num_samples_per_input=num_samples,
                                    eta=eta,
                                    save_path=save_path)


    # Extract mechanical properties from generated curves
    mechanical_properties = {}
    max_forces = []
    gen_curves = np.clip(gen_curves, 0, 1)
    for curve in gen_curves:
        curve = curve.reshape(-1)
        peak_indices, _ = find_force_peaks(curve, **default_peak_params)
        max_force = calculate_max_force(curve, peak_indices)
        max_forces.append(max_force)

    max_forces = np.array(max_forces)
    mechanical_properties['Unfolding Force (pN)'] = (max_forces * (13.9489 + -0.178778 + 1e-8) - 0.178778) * 55

    plot_property_distributions(generated_properties=mechanical_properties, property_name='Unfolding Force (pN)',)

    return gen_curves, mechanical_properties
