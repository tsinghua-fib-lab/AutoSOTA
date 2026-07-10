# Configuration options for watermark experiments

# Model options used by the paper experiments.
MODEL_OPTIONS = {
    'opt-125m': {
        'model_name': 'facebook/opt-125m',
        'vram_gb': 0.5,
        'ram_gb': 1.5,
        'description': 'OPT 125M - Tiny, fast generation'
    },
    'gpt2': {
        'model_name': 'gpt2',
        'vram_gb': 0.5,
        'ram_gb': 1.5,
        'description': 'GPT-2 117M - Small, classic model'
    },
    'pythia-160m': {
        'model_name': 'EleutherAI/pythia-160m',
        'vram_gb': 0.6,
        'ram_gb': 1.6,
        'description': 'Pythia 160M - Small, research model'
    },
    'gemma-2-9b': {
        'model_name': 'google/gemma-2-9b',
        'vram_gb': 19,
        'ram_gb': 22.0,
        'description': 'Gemma 2 9B - large-model validation'
    },
}

# Dataset options used by the paper experiments.
DATASET_OPTIONS = {
    'c4': {
        'dataset_name': 'allenai/c4',
        'dataset_config': "realnewslike",
        'text_field': 'text',
        'ram_gb': 0.6,  # Streaming mode with 250 samples
        'description': 'C4: Colossal Clean Crawled Corpus - web crawl data'
    },
    'wikipedia': {
        'dataset_name': 'wikimedia/wikipedia',
        'dataset_config': '20231101.en',
        'text_field': 'text',
        'ram_gb': 1.1,  # Streaming mode with 250 samples
        'description': 'Wikipedia articles - encyclopedic content'
    },
    'lfqa': {
        'dataset_name': 'vblagoje/lfqa',
        'dataset_config': None,
        'text_field': 'answers',
        'ram_gb': 0.7,  # Streaming mode with 250 samples
        'description': 'LFQA - Long-form question answering text'
    }
}


def create_config(
    model_key='opt-125m',
    dataset_key='wikipedia',
    num_samples=200,
    truncate_at=50,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    gamma=0.15,
    delta=0.75,
    output_dir='results',
    experiment_name=None,
    seed=0,
    min_generated_length=200,
    top_k=0,
    top_p=1.0,
    cache_dir='./hf_cache',
):
    """
    Create a configuration dictionary for watermark experiment.

    Args:
        model_key: Key from MODEL_OPTIONS
        dataset_key: Key from DATASET_OPTIONS
        num_samples: Number of sentences to process
        truncate_at: Truncation position in tokens
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        gamma: Watermark gamma parameter
        delta: Watermark delta parameter
        output_dir: Directory for output files
        experiment_name: Name for this experiment (auto-generated if None)
        seed: Random seed for reproducibility
        min_generated_length: Minimum character length for generated text (both wm and nowm must meet threshold)
        cache_dir: Directory for Hugging Face cache (default: './hf_cache')

    Returns:
        dict: Complete configuration
    """
    if model_key not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_OPTIONS.keys())}")
    if dataset_key not in DATASET_OPTIONS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from {list(DATASET_OPTIONS.keys())}")

    # Auto-generate experiment name if not provided
    if experiment_name is None:
        sample_mode = "sample" if do_sample else "greedy"
        experiment_name = f"{model_key}_{dataset_key}_g{gamma}_d{delta}_t{temperature}_{sample_mode}"

    config = {
        # Experiment metadata
        'experiment_name': experiment_name,
        'output_dir': output_dir,
        'seed': seed,
        'cache_dir': cache_dir,

        # Model settings
        'model_key': model_key,
        'model_name': MODEL_OPTIONS[model_key]['model_name'],
        'model_description': MODEL_OPTIONS[model_key]['description'],
        'model_vram_gb': MODEL_OPTIONS[model_key]['vram_gb'],

        # Dataset settings
        'dataset_key': dataset_key,
        'dataset_name': DATASET_OPTIONS[dataset_key]['dataset_name'],
        'dataset_config': DATASET_OPTIONS[dataset_key]['dataset_config'],
        'text_field': DATASET_OPTIONS[dataset_key]['text_field'],
        'dataset_description': DATASET_OPTIONS[dataset_key]['description'],
        'split': 'train',
        'num_samples': num_samples,
        'streaming': True,

        # Truncation settings
        'truncate_at': truncate_at,

        # Generation settings
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_k':  top_k,
        'top_p':  top_p,
        'do_sample': do_sample,

        # Watermark settings
        'gamma': gamma,
        'delta': delta,
        'seeding_scheme': 'simple_1',
        'z_threshold': 4.0,

        # Filter settings
        'min_generated_length': min_generated_length,
    }

    return config


def validate_config(config):
    """Validate configuration dictionary."""
    required_keys = [
        'experiment_name', 'model_name', 'dataset_name', 'text_field',
        'num_samples', 'truncate_at', 'max_new_tokens', 'gamma', 'delta'
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate ranges
    if not (0 < config['gamma'] < 1):
        raise ValueError(f"gamma must be between 0 and 1, got {config['gamma']}")
    if config['delta'] < 0:
        raise ValueError(f"delta must be non-negative, got {config['delta']}")
    if config['temperature'] <= 0:
        raise ValueError(f"temperature must be positive, got {config['temperature']}")

    return True
