"""
Simplified OOD evaluation script.

Train and evaluate on MMLU-Pro (or any category) with one command.
Compares R2-Router against baselines: MIRT, NIRT, CARROT-Linear, CARROT-KNN.

Usage:
    python ood_evaluation/run_ood.py                    # MMLU-Pro (default)
    python ood_evaluation/run_ood.py --category "lighteval/MATH/all"
    python ood_evaluation/run_ood.py --quick            # Quick demo with 1 model
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ood_dataset_manager import OODDatasetManager
from main.shared.router_dataset import RouterDataset
from main.r2.predictor_sklearn import TokenPerformancePredictor as SklearnPredictor, route_scores
from main.baselines.carrot.baselines_carrot import CarrotKNNBaseline as CarrotBaseline, CarrotLinearBaseline as LinearCarrotBaseline, route_baseline

# IRT baselines are optional (require sentence-transformers)
try:
    from main.baselines.irt.baselines_irt import IRTBaseline, NIRTBaseline
    IRT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  IRT baselines not available: {e}")
    print("   Install sentence-transformers to enable IRT baselines")
    IRT_AVAILABLE = False
    IRTBaseline = None
    NIRTBaseline = None


# ============================================================================
# CONFIGURATION
# ============================================================================

# Token limits
TOKEN_LIMITS_SCORE = [
    '10_score', '20_score', '30_score', '40_score', '50_score',
    '80_score', '100_score', '150_score', '200_score', '300_score',
    '500_score', '800_score', '1200_score', '2000_score', '4000_score',
    'unlimited_score'
]

TOKEN_LIMITS_COUNT = [
    '10_count', '20_count', '30_count', '40_count', '50_count',
    '80_count', '100_count', '150_count', '200_count', '300_count',
    '500_count', '800_count', '1200_count', '2000_count', '4000_count',
    'unlimited_count'
]

# Model pool (full set of 10 LLMs)
MODELS = {
    "GLM_4_5_Air": {"name": "GLM-4.5-Air", "csv": "data/GLM-4.5-Air.csv", "size": 0.85},
    "GLM_4_6": {"name": "GLM-4.6", "csv": "data/GLM-4.6.csv", "size": 1.75},
    "gemma_3_4b_it": {"name": "gemma-3-4b-it", "csv": "data/gemma-3-4b-it.csv", "size": 0.06815},
    "Llama_3_1_70B": {"name": "Llama-3.1-70B-Instruct", "csv": "data/Llama-3.1-70B-Instruct.csv", "size": 0.40},
    "Llama_3_2_3B": {"name": "Llama-3.2-3B-Instruct", "csv": "data/Llama-3.2-3B-Instruct.csv", "size": 0.02},
    "Qwen2_5_Math_1_5B": {"name": "Qwen2.5-Math-1.5B-Instruct", "csv": "data/Qwen2.5-Math-1.5B-Instruct.csv", "size": 0.09},
    "Qwen2_5_Math_7B": {"name": "Qwen2.5-Math-7B-Instruct", "csv": "data/Qwen2.5-Math-7B-Instruct.csv", "size": 0.35},
    "Qwen3_0_6B": {"name": "Qwen3-0.6B", "csv": "data/Qwen3-0.6B.csv", "size": 0.0173},
    "Qwen3_235B": {"name": "Qwen3-235B-A22B-Instruct-2507", "csv": "data/Qwen3-235B-A22B-Instruct-2507.csv", "size": 0.55},
    "Qwen3_Next_80B": {"name": "Qwen3-Next-80B-A3B-Instruct", "csv": "data/Qwen3-Next-80B-A3B-Instruct.csv", "size": 0.6}
}

def parse_lambda_distribution(lambda_dist_str):
    """
    Parse lambda distribution string into numpy array.

    Format: "min,max,num;min,max,num;..."
    Example: "0,0.2,20;0.2,1.0,50" -> concatenate linspace(0,0.2,20) and linspace(0.2,1.0,50)
    """
    segments = []
    for segment in lambda_dist_str.split(';'):
        parts = segment.split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid lambda distribution segment: {segment}. Expected 'min,max,num'")
        min_val, max_val, num_points = float(parts[0]), float(parts[1]), int(parts[2])
        segments.append(np.linspace(min_val, max_val, num_points))
    return np.unique(np.concatenate(segments))


# ============================================================================
# TRAIN MODELS
# ============================================================================

def train_models(embeddings, train_idx, test_idx, models_config, test_category, quick=False):
    """
    Train predictors for all models on OOD training split.

    Args:
        embeddings: Query embeddings
        train_idx: Training indices (excludes test category)
        test_idx: Test indices (test category only)
        models_config: Dictionary of model configurations
        test_category: Name of the held-out test category (for checkpoint naming)
        quick: If True, only train first model

    Returns:
        Dictionary of trained LLM data
    """
    print("\n" + "=" * 100)
    print("TRAINING MODELS (R2-Router METHOD)")
    print("=" * 100)

    llms = {}
    models_to_train = list(models_config.items())[:1] if quick else list(models_config.items())

    for model_key, model_config in models_to_train:
        print(f"\n{'-'*80}")
        print(f"Training {model_key}...")
        print(f"{'-'*80}")

        if not os.path.exists(model_config["csv"]):
            print(f"⚠️  Skipping: CSV not found at {model_config['csv']}")
            continue

        # Create dataset with OOD split
        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=model_config["csv"],
            target_tokens_score=TOKEN_LIMITS_SCORE,
            train_idx=train_idx,
            test_idx=test_idx
        )

        # Extract training data from dataset
        train_data = dataset.get_train_set_score()
        train_X = train_data["X"]
        train_y = train_data["y"]

        # Separate limited and unlimited training data
        quality_train_limited = {k: v for k, v in train_y.items() if k != 'unlimited_score'}
        quality_train_unlimited = train_y['unlimited_score']
        token_count_train = dataset.get_train_token_unlimited_count()

        # Extract test data from dataset
        test_data = dataset.get_test_set_score()
        test_X = test_data["X"]
        test_y = test_data["y"]

        # Separate limited and unlimited test data
        quality_test_limited = {k: v for k, v in test_y.items() if k != 'unlimited_score'}
        quality_test_unlimited = test_y['unlimited_score']
        token_count_test = dataset.get_test_token_unlimited_count()

        # Check if checkpoint exists
        # Include test category in checkpoint path to avoid conflicts between different OOD splits
        category_safe = test_category.replace('/', '_')
        checkpoint_dir = f"./checkpoints/ood_evaluation/{category_safe}/{model_key}_ridge_alpha10.0"
        checkpoint_exists = (
            os.path.exists(checkpoint_dir) and
            os.path.isfile(os.path.join(checkpoint_dir, "limited_score_predictors.joblib")) and
            os.path.isfile(os.path.join(checkpoint_dir, "unlimited_score_predictor.joblib")) and
            os.path.isfile(os.path.join(checkpoint_dir, "unlimited_token_predictor.joblib"))
        )

        if checkpoint_exists:
            # Load existing checkpoint
            print(f"✓ Loading existing checkpoint from {checkpoint_dir}")
            predictor = SklearnPredictor(
                token_limits=TOKEN_LIMITS_SCORE,
                load_dir=checkpoint_dir
            )
        else:
            # Train new predictor
            print(f"Training new predictor...")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Use Ridge regression with alpha=10.0 (same as IID training)
            # This prevents overfitting and improves OOD generalization
            predictor = SklearnPredictor(
                token_limits=TOKEN_LIMITS_SCORE,
                model_type="ridge",
                alpha=10.0
            )
            predictor.fit(
                embedding_train=train_X,
                quality_train_limited=quality_train_limited,
                quality_train_unlimited=quality_train_unlimited,
                token_count_train=token_count_train,
                embedding_test=test_X,
                quality_test_limited=quality_test_limited,
                quality_test_unlimited=quality_test_unlimited,
                token_count_test=token_count_test,
                save_dir=checkpoint_dir,
                plot_dir=None  # Don't save plots to keep it clean
            )
            print(f"✓ Saved checkpoint to {checkpoint_dir}")

        # Get predictions on TEST set (OOD queries)
        # Use predict_combined() which returns (n_queries, 16) for scores
        pred_scores, pred_counts_unlimited = predictor.predict_combined(test_X)

        # Get true scores from the dataset
        true_scores = np.column_stack([
            test_data["y"][col] for col in TOKEN_LIMITS_SCORE
        ])

        # Get true counts from the dataframe directly (for evaluation only)
        test_idx_list = test_idx.tolist() if hasattr(test_idx, 'tolist') else test_idx
        true_counts = np.column_stack([
            dataset.score_df[col].values[test_idx_list] for col in TOKEN_LIMITS_COUNT
        ])

        # Create predicted token counts for routing
        # IMPORTANT: Router cannot know actual token counts before inference!
        # For limited budgets: use the limit directly (not min with predicted count)
        #   Router uses the limit as the cost estimate for budgeted inference
        # For unlimited: use predicted token count
        token_limits_values = [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]
        pred_counts = np.column_stack([
            np.full_like(pred_counts_unlimited, limit) for limit in token_limits_values
        ] + [pred_counts_unlimited])  # Shape: (n_queries, 16)

        # Store LLM data
        llms[model_key] = {
            'name': model_config["name"],
            'size': model_config["size"],
            'pred_test_score': pred_scores,              # Shape: (n_queries, 16)
            'pred_test_token': pred_counts,              # Shape: (n_queries, 16) - use LIMITS + predicted unlimited!
            'pred_test_count': pred_counts,              # Also store with this name
            'true_test_score': true_scores,              # Shape: (n_queries, 16)
            'true_test_count': true_counts               # Shape: (n_queries, 16)
        }

        print(f"✓ {model_key} trained successfully")

    return llms


# ============================================================================
# TRAIN BASELINES
# ============================================================================

def train_baselines(embeddings, train_idx, test_idx, models_config, test_category, quick=False):
    """
    Train baseline methods on OOD training split.

    Args:
        embeddings: Query embeddings
        train_idx: Training indices (excludes test category)
        test_idx: Test indices (test category only)
        models_config: Dictionary of model configurations
        test_category: Name of the held-out test category (for checkpoint naming)
        quick: If True, only use first model

    Returns:
        Tuple of (baselines_dict, test_data_dict)
        - baselines_dict: Dictionary of trained baseline models
        - test_data_dict: Dictionary with 'embedding_test', 'quality_test', 'token_count_test'
    """
    print("\n" + "=" * 100)
    print("TRAINING BASELINES")
    print("=" * 100)

    # Collect data from all models
    models_to_use = list(models_config.items())[:1] if quick else list(models_config.items())

    quality_train_list = []
    token_count_train_list = []
    quality_test_list = []
    token_count_test_list = []

    for model_key, model_config in models_to_use:
        if not os.path.exists(model_config["csv"]):
            continue

        dataset = RouterDataset(
            embeddings=embeddings,
            score_df_path=model_config["csv"],
            target_tokens_score=TOKEN_LIMITS_SCORE,
            train_idx=train_idx,
            test_idx=test_idx
        )

        # Extract unlimited scores and counts for baselines
        train_unlimited_score = dataset.get_train_set_score()["y"]["unlimited_score"]
        test_unlimited_score = dataset.get_test_set_score()["y"]["unlimited_score"]
        train_unlimited_count = dataset.get_train_token_unlimited_count()
        test_unlimited_count = dataset.get_test_token_unlimited_count()

        quality_train_list.append(train_unlimited_score)
        quality_test_list.append(test_unlimited_score)
        token_count_train_list.append(train_unlimited_count)
        token_count_test_list.append(test_unlimited_count)

    if len(quality_train_list) == 0:
        print("⚠️  No models available for baseline training")
        return {}

    # Stack into matrices (n_queries, n_models)
    embedding_train = embeddings[train_idx]
    embedding_test = embeddings[test_idx]
    quality_train = np.column_stack(quality_train_list)
    token_count_train = np.column_stack(token_count_train_list)
    quality_test = np.column_stack(quality_test_list)
    token_count_test = np.column_stack(token_count_test_list)

    baselines = {}

    # Include category in checkpoint paths
    category_safe = test_category.replace('/', '_')

    # Train CARROT-KNN
    try:
        carrot_knn_dir = f"./checkpoints/ood_evaluation/{category_safe}/carrot_knn"
        carrot_knn_config_file = os.path.join(carrot_knn_dir, "config.txt")

        # Check if checkpoint exists and matches current model pool
        carrot_knn_exists = (
            os.path.exists(carrot_knn_dir) and
            os.path.isfile(os.path.join(carrot_knn_dir, "knn_score.joblib")) and
            os.path.isfile(os.path.join(carrot_knn_dir, "knn_count.joblib")) and
            os.path.isfile(carrot_knn_config_file)
        )

        # Check if model pool matches
        models_match = False
        if carrot_knn_exists:
            with open(carrot_knn_config_file, 'r') as f:
                saved_models = f.read().strip()
            current_models = ','.join(sorted([mk for mk, _ in models_to_use]))
            models_match = (saved_models == current_models)
            if not models_match:
                print(f"\n⚠️  CARROT-KNN model pool changed (was {len(saved_models.split(','))} models, now {len(models_to_use)} models)")
                print("   Will retrain...")
                carrot_knn_exists = False

        if carrot_knn_exists and models_match:
            print("\n✓ Loading existing CARROT-KNN from checkpoint...")
            carrot_knn = CarrotBaseline(load_dir=carrot_knn_dir)
        else:
            print("\nTraining CARROT-KNN...")
            carrot_knn = CarrotBaseline(n_neighbors_score=min(256, len(train_idx)//2))
            carrot_knn.fit(
                embedding_train=embedding_train,
                quality_train=quality_train,
                token_count_train=token_count_train,
                save_dir=carrot_knn_dir
            )
            # Save model pool config
            os.makedirs(carrot_knn_dir, exist_ok=True)
            with open(carrot_knn_config_file, 'w') as f:
                f.write(','.join(sorted([mk for mk, _ in models_to_use])))
            print(f"✓ CARROT-KNN trained successfully, saved to {carrot_knn_dir}")

        baselines["CARROT-KNN"] = carrot_knn
    except Exception as e:
        print(f"❌ CARROT-KNN failed: {e}")
        import traceback
        traceback.print_exc()

    # Train CARROT-Linear
    try:
        carrot_linear_dir = f"./checkpoints/ood_evaluation/{category_safe}/carrot_linear"
        carrot_linear_config_file = os.path.join(carrot_linear_dir, "config.txt")

        # Check if checkpoint exists and matches current model pool
        carrot_linear_exists = (
            os.path.exists(carrot_linear_dir) and
            os.path.isfile(os.path.join(carrot_linear_dir, "linear_score.joblib")) and
            os.path.isfile(os.path.join(carrot_linear_dir, "linear_count.joblib")) and
            os.path.isfile(carrot_linear_config_file)
        )

        # Check if model pool matches
        models_match = False
        if carrot_linear_exists:
            with open(carrot_linear_config_file, 'r') as f:
                saved_models = f.read().strip()
            current_models = ','.join(sorted([mk for mk, _ in models_to_use]))
            models_match = (saved_models == current_models)
            if not models_match:
                print(f"\n⚠️  CARROT-Linear model pool changed (was {len(saved_models.split(','))} models, now {len(models_to_use)} models)")
                print("   Will retrain...")
                carrot_linear_exists = False

        if carrot_linear_exists and models_match:
            print("\n✓ Loading existing CARROT-Linear from checkpoint...")
            carrot_linear = LinearCarrotBaseline(load_dir=carrot_linear_dir)
        else:
            print("\nTraining CARROT-Linear...")
            carrot_linear = LinearCarrotBaseline()
            carrot_linear.fit(
                embedding_train=embedding_train,
                quality_train=quality_train,
                token_count_train=token_count_train,
                save_dir=carrot_linear_dir
            )
            # Save model pool config
            os.makedirs(carrot_linear_dir, exist_ok=True)
            with open(carrot_linear_config_file, 'w') as f:
                f.write(','.join(sorted([mk for mk, _ in models_to_use])))
            print(f"✓ CARROT-Linear trained successfully, saved to {carrot_linear_dir}")

        baselines["CARROT-Linear"] = carrot_linear
    except Exception as e:
        print(f"❌ CARROT-Linear failed: {e}")
        import traceback
        traceback.print_exc()

    # Generate LLM embeddings for IRT baselines
    llm_names = [model_key for model_key, _ in models_to_use
                 if os.path.exists(models_config[model_key]["csv"])]

    if IRT_AVAILABLE:
        from sentence_transformers import SentenceTransformer
        print("\n>>> Generating LLM embeddings for IRT baselines...")
        encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # Use model names as descriptions for LLM embeddings
        llm_texts = [models_config[name]["name"] for name in llm_names]
        llm_embeddings = encoder.encode(llm_texts, show_progress_bar=False)
        llm_embeddings = np.array(llm_embeddings)
        print(f"✓ Generated embeddings for {len(llm_names)} models")
    else:
        llm_embeddings = None

    # Train IRT (MIRT) - optional
    if IRT_AVAILABLE:
        try:
            mirt_dir = f"./checkpoints/ood_evaluation/{category_safe}/irt_mirt"
            mirt_config_file = os.path.join(mirt_dir, "config.txt")
            mirt_exists = (
                os.path.exists(mirt_dir) and
                os.path.isfile(os.path.join(mirt_dir, "mirt_model.pt")) and
                os.path.isfile(os.path.join(mirt_dir, "mirt_config.pt")) and
                os.path.isfile(os.path.join(mirt_dir, "mirt_llm_embeddings.pt")) and
                os.path.isfile(mirt_config_file)
            )

            # Check if model pool matches
            models_match = False
            if mirt_exists:
                with open(mirt_config_file, 'r') as f:
                    saved_models = f.read().strip()
                current_models = ','.join(sorted(llm_names))
                models_match = (saved_models == current_models)
                if not models_match:
                    print(f"\n⚠️  MIRT model pool changed (was {len(saved_models.split(','))} models, now {len(llm_names)} models)")
                    print("   Will retrain...")
                    mirt_exists = False

            if mirt_exists and models_match:
                print("\n✓ Loading existing MIRT from checkpoint...")
                irt = IRTBaseline(load_dir=mirt_dir)
            else:
                print("\nTraining MIRT...")
                irt = IRTBaseline(latent_dim=10)
                irt.fit(
                    embedding_train=embedding_train,
                    quality_train=quality_train,
                    llm_embeddings=llm_embeddings,
                    llm_names=llm_names,
                    epochs=50,
                    save_dir=mirt_dir
                )
                # Save model pool config
                os.makedirs(mirt_dir, exist_ok=True)
                with open(mirt_config_file, 'w') as f:
                    f.write(','.join(sorted(llm_names)))
                print(f"✓ MIRT trained successfully, saved to {mirt_dir}")

            baselines["MIRT"] = irt
        except Exception as e:
            print(f"❌ MIRT failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Skipping MIRT (IRT not available)")

    # Train NIRT - optional
    if IRT_AVAILABLE:
        try:
            nirt_dir = f"./checkpoints/ood_evaluation/{category_safe}/irt_nirt"
            nirt_config_file = os.path.join(nirt_dir, "config.txt")
            nirt_exists = (
                os.path.exists(nirt_dir) and
                os.path.isfile(os.path.join(nirt_dir, "nirt_model.pt")) and
                os.path.isfile(os.path.join(nirt_dir, "nirt_config.pt")) and
                os.path.isfile(os.path.join(nirt_dir, "nirt_llm_embeddings.pt")) and
                os.path.isfile(nirt_config_file)
            )

            # Check if model pool matches
            models_match = False
            if nirt_exists:
                with open(nirt_config_file, 'r') as f:
                    saved_models = f.read().strip()
                current_models = ','.join(sorted(llm_names))
                models_match = (saved_models == current_models)
                if not models_match:
                    print(f"\n⚠️  NIRT model pool changed (was {len(saved_models.split(','))} models, now {len(llm_names)} models)")
                    print("   Will retrain...")
                    nirt_exists = False

            if nirt_exists and models_match:
                print("\n✓ Loading existing NIRT from checkpoint...")
                nirt = NIRTBaseline(load_dir=nirt_dir)
            else:
                print("\nTraining NIRT...")
                nirt = NIRTBaseline(latent_dim=10)
                nirt.fit(
                    embedding_train=embedding_train,
                    quality_train=quality_train,
                    llm_embeddings=llm_embeddings,
                    llm_names=llm_names,
                    epochs=50,
                    save_dir=nirt_dir
                )
                # Save model pool config
                os.makedirs(nirt_dir, exist_ok=True)
                with open(nirt_config_file, 'w') as f:
                    f.write(','.join(sorted(llm_names)))
                print(f"✓ NIRT trained successfully, saved to {nirt_dir}")

            baselines["NIRT"] = nirt
        except Exception as e:
            print(f"❌ NIRT failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Skipping NIRT (IRT not available)")

    # Package test data for evaluation
    test_data = {
        'embedding_test': embedding_test,
        'quality_test': quality_test,
        'token_count_test': token_count_test
    }

    return baselines, test_data


# ============================================================================
# EVALUATE ROUTING PERFORMANCE
# ============================================================================

def evaluate_routing(llms, baselines, test_data, lambda_range, models_config):
    """
    Evaluate routing performance for R2-Router and baselines.

    Args:
        llms: Dictionary of LLM predictors
        baselines: Dictionary of baseline models
        test_data: Dictionary with 'embedding_test', 'quality_test', 'token_count_test'
        lambda_range: Array of lambda values
        models_config: Dictionary of model configurations

    Returns:
        DataFrame with columns: method, cost, normalized_cost, accuracy
    """
    embedding_test = test_data['embedding_test']
    quality_test = test_data['quality_test']
    token_count_test = test_data['token_count_test']
    print("\n" + "=" * 100)
    print("EVALUATING ROUTING PERFORMANCE")
    print("=" * 100)

    results = []

    # Evaluate R2-Router
    print("\nEvaluating R2-Router...")
    r2_cost, core_perf = route_scores(llms, lambda_range)

    # Normalize R2-Router cost
    r2_cost_arr = np.array(r2_cost)
    if r2_cost_arr.max() > r2_cost_arr.min():
        core_normalized = (r2_cost_arr - r2_cost_arr.min()) / (r2_cost_arr.max() - r2_cost_arr.min())
    else:
        core_normalized = np.zeros_like(r2_cost_arr)

    for cost, norm_cost, perf in zip(r2_cost, core_normalized, core_perf):
        results.append({'method': 'R2-Router', 'cost': cost, 'normalized_cost': norm_cost, 'accuracy': perf})

    # Evaluate baselines
    for baseline_name, baseline_obj in baselines.items():
        print(f"Evaluating {baseline_name}...")
        try:
            # Check baseline type to handle different predict() return values
            is_irt_baseline = baseline_name in ['MIRT', 'NIRT']

            if is_irt_baseline:
                # IRT baselines only return scores (no token count prediction)
                Y_hat_score_test = baseline_obj.predict(embedding_test)
                # IRT only routes among LLMs (no token budget optimization)
                # Use constant token count for all models so cost ∝ model_size only
                mean_token_count = token_count_test.mean()
                Y_hat_count_test = np.full_like(quality_test, mean_token_count)
            else:
                # CARROT baselines return (scores, counts)
                Y_hat_score_test, Y_hat_count_test = baseline_obj.predict(embedding_test)

            # Get true scores and counts from test_data
            # Baselines only predict unlimited scores/counts for each model
            Y_score_test_true = quality_test
            Y_count_test_true = token_count_test

            # Create size vector (one size per model)
            model_keys = list(models_config.keys())[:len(Y_score_test_true[0])]
            sizes_vec = np.array([models_config[k]["size"] for k in model_keys])

            # Route using baseline predictions
            bl_cost, bl_perf = route_baseline(
                Y_hat_score=Y_hat_score_test,
                Y_hat_count=Y_hat_count_test,
                Y_score_true=Y_score_test_true,
                Y_count_true=Y_count_test_true,
                lamb_range=lambda_range,
                sizes_vec=sizes_vec
            )

            # Normalize baseline cost
            bl_cost_arr = np.array(bl_cost)
            if bl_cost_arr.max() > bl_cost_arr.min():
                bl_normalized = (bl_cost_arr - bl_cost_arr.min()) / (bl_cost_arr.max() - bl_cost_arr.min())
            else:
                bl_normalized = np.zeros_like(bl_cost_arr)

            for cost, norm_cost, perf in zip(bl_cost, bl_normalized, bl_perf):
                results.append({'method': baseline_name, 'cost': cost, 'normalized_cost': norm_cost, 'accuracy': perf})
            print(f"✓ {baseline_name} evaluated successfully")
        except Exception as e:
            print(f"❌ {baseline_name} evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZE RESULTS
# ============================================================================

def plot_results(results_df, output_path):
    """
    Plot cost-performance curves with both actual and normalized costs.

    Generates two plots:
    1. Actual cost plot (saved as {output_path}_actual.png)
    2. Normalized cost plot (saved as {output_path}_normalized.png)
    """
    # Plot 1: Actual Cost
    plt.figure(figsize=(10, 6))

    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        plt.plot(method_data['cost'], method_data['accuracy'], label=method, linewidth=2)

    plt.xlabel('Cost (Token Count × Model Size)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('OOD Routing Performance: Cost vs Accuracy (Actual Cost)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save actual cost plot
    actual_path = output_path.replace('.png', '_actual.png')
    plt.savefig(actual_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved actual cost plot to {actual_path}")
    plt.close()

    # Plot 2: Normalized Cost
    plt.figure(figsize=(10, 6))

    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method].sort_values('cost')

        # Normalize cost to [0,1]
        cost_array = method_data['cost'].values
        if cost_array.max() > cost_array.min():
            normalized_cost = (cost_array - cost_array.min()) / (cost_array.max() - cost_array.min())
        else:
            normalized_cost = np.zeros_like(cost_array)

        plt.plot(normalized_cost, method_data['accuracy'].values, label=method, linewidth=2)

    plt.xlabel('Normalized Cost [0,1]', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('OOD Routing Performance: Cost vs Accuracy (Normalized Cost)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save normalized cost plot
    normalized_path = output_path.replace('.png', '_normalized.png')
    plt.savefig(normalized_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved normalized cost plot to {normalized_path}")
    plt.close()


# ============================================================================
# COMPUTE METRICS
# ============================================================================

def compute_metrics(results_df, best_llm_accuracy=None, target_accuracy_rate=1.0):
    """
    Compute summary metrics for each method.

    QNC (Query-Normalized Cost) is the relative cost to achieve the same performance
    as the most accurate single LLM. If a method cannot reach that target, QNC = 1.0.

    Args:
        results_df: DataFrame with columns ['method', 'cost', 'accuracy']
        best_llm_accuracy: Target accuracy (from best single LLM). If None, use global max.
        target_accuracy_rate: Rate to apply to best_llm_accuracy (default: 1.0 for 100%).
                             E.g., 0.9 means target is 90% of best LLM's performance.

    Returns:
        DataFrame with columns: method, peak_accuracy,
                                AUDC_normalized, QNC, AUDC_actual
    """
    metrics = []

    # If best_llm_accuracy not provided, find the global maximum accuracy across all methods
    if best_llm_accuracy is None:
        best_llm_accuracy = results_df['accuracy'].max()
        print(f"\n⚠️  Warning: best_llm_accuracy not provided. Using global max: {best_llm_accuracy:.4f}")

    # Apply target accuracy rate
    target_accuracy = best_llm_accuracy * target_accuracy_rate

    print(f"\nQNC Target Configuration:")
    print(f"  Best Single LLM Accuracy: {best_llm_accuracy:.4f}")
    print(f"  Target Accuracy Rate: {target_accuracy_rate:.2f} ({target_accuracy_rate*100:.0f}%)")
    print(f"  QNC Target Accuracy: {target_accuracy:.4f}")

    # Find global min and max costs for normalization
    global_min_cost = results_df['cost'].min()
    global_max_cost = results_df['cost'].max()

    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method].sort_values('cost')

        cost_curve = method_data['cost'].values
        perf_curve = method_data['accuracy'].values

        peak_accuracy = perf_curve.max()

        # === Normalized Cost Metrics (cost normalized to [0,1]) ===
        # Normalize cost to [0,1] using GLOBAL min/max (not per-method)
        if global_max_cost > global_min_cost:
            normalized_cost = (cost_curve - global_min_cost) / (global_max_cost - global_min_cost)
        else:
            normalized_cost = np.zeros_like(cost_curve)

        # AUDC with normalized cost
        audc_normalized = np.trapezoid(perf_curve, normalized_cost)

        # QNC: cost to reach target_accuracy (normalized to [0,1])
        # If method cannot reach target, QNC = 1.0 (100%)
        idx = np.where(perf_curve >= target_accuracy)[0]
        if len(idx) > 0:
            qnc = normalized_cost[idx[0]]
        else:
            qnc = 1.0  # Cannot reach target

        # === Actual Cost Metrics (cost in original units) ===
        # AUDC with actual cost
        audc_actual = np.trapezoid(perf_curve, cost_curve)

        metrics.append({
            'method': method,
            'peak_accuracy': peak_accuracy,
            'AUDC_normalized': audc_normalized,
            'QNC': qnc,
            'AUDC_actual': audc_actual
        })

    return pd.DataFrame(metrics)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='OOD Evaluation for LLM Routing')
    parser.add_argument('--category', type=str, default='TIGER-Lab/MMLU-Pro',
                        help='Test category for OOD evaluation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: use only 1 model')
    parser.add_argument('--output', type=str, default='./comparison_results/ood_evaluation',
                        help='Output directory for results')
    parser.add_argument('--lambda-dist', type=str, default='0,0.2,20;0.2,1.0,50',
                        help='Lambda distribution: "min,max,num;min,max,num;..." (default: 0,0.2,20;0.2,1.0,50)')
    parser.add_argument('--model', action='append', nargs=3,
                        metavar=('NAME', 'SIZE', 'CSV'),
                        help='Model configuration: name size csv_path (can be specified multiple times)')
    parser.add_argument('--target-accuracy-rate', type=float, default=1.0,
                        help='Target accuracy rate for QNC (default: 1.0 for 100%%). E.g., 0.9 means 90%% of best LLM')
    args = parser.parse_args()

    # Build MODELS dictionary from command-line args (if provided)
    if args.model:
        models_config = {}
        for name, size, csv in args.model:
            # Convert name to key (replace special chars with underscores)
            model_key = name.replace('.', '_').replace('-', '_')
            models_config[model_key] = {
                "name": name,
                "csv": csv,
                "size": float(size)
            }
        print(f"\n✓ Using {len(models_config)} models from command line")
    else:
        # Fall back to default MODELS if no command-line args
        models_config = MODELS
        print(f"\n✓ Using default model pool ({len(models_config)} models)")

    print("=" * 100)
    print("OOD EVALUATION")
    print("=" * 100)
    print(f"\nTest category (OOD): {args.category}")
    print(f"Mode: {'Quick (1 model)' if args.quick else 'Full (all models)'}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    print("\n" + "=" * 100)
    print("LOADING DATA")
    print("=" * 100)

    # Use OOD split
    ood_manager = OODDatasetManager(
        embeddings_path="data/prompt_embeddings.pkl",
        ood_splits_path="./ood_evaluation/category_splits/ood_splits.pkl",
        test_category=args.category
    )

    embeddings = ood_manager.get_embeddings()
    train_idx, test_idx = ood_manager.get_split_indices()

    print(f"\nTrain set: {len(train_idx)} queries")
    print(f"Test set: {len(test_idx)} queries")

    # Parse lambda distribution
    lambda_range = parse_lambda_distribution(args.lambda_dist)
    print(f"\nLambda distribution: {args.lambda_dist}")
    print(f"Lambda points: {len(lambda_range)} (range: [{lambda_range.min():.4f}, {lambda_range.max():.4f}])")

    # Train models (passing test_category for checkpoint naming)
    llms = train_models(embeddings, train_idx, test_idx, models_config, args.category, quick=args.quick)

    # Train baselines (passing test_category for checkpoint naming)
    baselines, test_data = train_baselines(embeddings, train_idx, test_idx, models_config, args.category, quick=args.quick)

    # Evaluate routing
    results_df = evaluate_routing(llms, baselines, test_data, lambda_range, models_config)

    # Find best single LLM's average performance (for QNC computation)
    print("\n" + "=" * 100)
    print("FINDING BEST SINGLE LLM")
    print("=" * 100)

    best_llm_name = None
    best_llm_accuracy = 0.0

    for model_key, model_data in llms.items():
        # Get unlimited token accuracy (last column in true_test_score)
        unlimited_accuracy = model_data['true_test_score'][:, -1].mean()
        print(f"{model_data['name']}: {unlimited_accuracy:.4f}")

        if unlimited_accuracy > best_llm_accuracy:
            best_llm_accuracy = unlimited_accuracy
            best_llm_name = model_data['name']

    print(f"\n✓ Best Single LLM: {best_llm_name} with accuracy {best_llm_accuracy:.4f}")
    print(f"  This will be used as the QNC target accuracy.")

    # Compute metrics
    metrics_df = compute_metrics(results_df, best_llm_accuracy=best_llm_accuracy,
                                 target_accuracy_rate=args.target_accuracy_rate)

    # Print results
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    # Print metrics
    print("\n📊 EVALUATION METRICS:")
    print("-" * 100)
    print(metrics_df.to_string(index=False))
    print("\nMetric Definitions:")
    print("  - Peak Accuracy: Maximum performance achieved [0,1]")
    print("  - AUDC_normalized: Area under cost-performance curve (normalized cost) [higher is better]")
    print("  - QNC: Query-Normalized Cost to match best single LLM [0,1] (lower is better)")
    print("  - AUDC_actual: Area under cost-performance curve (actual cost units)")

    print("\n" + "=" * 100)

    # Save results
    cat_name_safe = args.category.replace('/', '_')
    results_df.to_csv(f"{args.output}/{cat_name_safe}_curves.csv", index=False)
    metrics_df.to_csv(f"{args.output}/{cat_name_safe}_metrics.csv", index=False)
    print(f"\n✓ Saved results to {args.output}/")

    # Plot
    plot_results(results_df, f"{args.output}/{cat_name_safe}_plot.png")

    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
