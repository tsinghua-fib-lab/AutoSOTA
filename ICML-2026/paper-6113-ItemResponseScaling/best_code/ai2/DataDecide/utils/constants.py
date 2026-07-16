SCALING_LAW_DISPLAY_NAMES = {
    '3_param': '3-parameter',
    '3_param-helper_points': '3-parameter with helper points',
    '3_param-step2=0.5': '3-parameter step 2 fit with >50 % checkpoints',
    '3_param-helper_points-step2=0.5': '3-parameter with helpers and >50 % checkpoints',
    '5_param-ai2': '5-parameter',
    '5_param-1_step-ai2': '5-parameter, single step',
    '3_param-1_step': '3-parameter, single step',
    '2_param': '2-parameter'
}

BENCHMARK_DISPLAY_NAMES = {
    "arc_challenge": "ARC Challenge",
    "arc_easy": "ARC Easy",
    "boolq": "BoolQ",
    "csqa": "CommonsenseQA",
    "hellaswag": "HellaSwag",
    "openbookqa": "OpenBookQA",
    "piqa": "PIQA",
    "socialiqa": "SocialIQA",
    "winogrande": "WinoGrande",
    "mmlu": "MMLU",
    'minerva': 'Minerva',
    'gsm8k': 'GSM8K',
    'mbpp': 'MBPP',
    'humaneval': 'HumanEval'
}

DATA_NAME_CLEAN = {
    "dolma17": "Dolma1.7",
    "no_code": "Dolma1.7 (no code)",
    "no_math_no_code": "Dolma1.7 (no math, code)",
    "no_reddit": "Dolma1.7 (no Reddit)",
    "no_flan": "Dolma1.7 (no Flan)",
    "dolma-v1-6-and-sources-baseline": "Dolma1.6++",
    "c4": "C4",
    "prox_fineweb_pro": "FineWeb-Pro",
    "fineweb_edu_dedup": "FineWeb-Edu",
    "falcon": "Falcon",
    "falcon_and_cc": "Falcon+CC",
    "falcon_and_cc_eli5_oh_top10p": "Falcon+CC (QC 10%)",
    "falcon_and_cc_eli5_oh_top20p": "Falcon+CC (QC 20%)",
    "falcon_and_cc_og_eli5_oh_top10p": "Falcon+CC (QC Orig 10%)",
    "falcon_and_cc_tulu_qc_top10": "Falcon+CC (QC Tulu 10%)",
    "DCLM-baseline": "DCLM-Baseline",
    "dolma17-75p-DCLM-baseline-25p": "DCLM-Baseline 25% / Dolma 75%",
    "dolma17-50p-DCLM-baseline-50p": "DCLM-Baseline 50% / Dolma 50%",
    "dolma17-25p-DCLM-baseline-75p": "DCLM-Baseline 75% / Dolma 25%",
    "dclm_ft7percentile_fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm_ft7percentile_fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm_fw_top10": "DCLM-Baseline (QC FW 10%)",
    "dclm_fw_top3": "DCLM-Baseline (QC FW 3%)",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "DCLM-Baseline (QC 10%)",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "DCLM-Baseline (QC 20%)",
}

TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
    "mmlu",
]

FULL_SCHEDULE_LAST_STEP_PER_MODEL = {
    "4M": 5725,  # min step value from {5745, 5725, 5735}
    "6M": 9182,
    "8M": 13039,
    "10M": 15117,
    "14M": 21953,
    "16M": 24432,
    "20M": 14584,  # min step value from {14584, 14594}
    "60M": 29042,  # min step value from {29042, 29052, 29062}
    "90M": 29901,
    "150M": 38157,
    "300M": 45787,
    "530M": 57786,
    "750M": 63589,
    "1B": 69369,
}

MODEL_TO_BATCH = {
    "4M": 32,
    "6M": 32,
    "8M": 32,
    "10M": 32,
    "14M": 32,
    "16M": 32,
    "20M": 64,
    "60M": 96,
    "90M": 160,
    "150M": 192,
    "300M": 320,
    "530M": 448,
    "750M": 576,
    "1B": 704,
}

# non-embedding params
MODEL_TO_PARAMS = {
    "4M": 3744832,
    "6M": 6010464,
    "8M": 8538240,
    "10M": 9900432,
    "12M": 12066600,
    "14M": 14380224,
    "16M": 16004560,
    "20M": 19101888,
    "60M": 57078144,
    "90M": 97946640,
    "150M": 151898880,
    "300M": 319980544,
    "530M": 530074944,
    "750M": 681297408,
    "1B": 1176832000,
}


model_data = {
    "4M": {
        "Batch size (sequences)": 32,
        "Dimension": 64,
        "MLP ratio": 8,
        "Model size (non-embedding)": 3744832,
        "Num heads": 8,
        "Num layers": 8,
        "Training steps": 5725,
    },
    "6M": {
        "Batch size (sequences)": 32,
        "Dimension": 96,
        "MLP ratio": 8,
        "Model size (non-embedding)": 6010464,
        "Num heads": 8,
        "Num layers": 8,
        "Training steps": 9182,
    },
    "8M": {
        "Batch size (sequences)": 32,
        "Dimension": 128,
        "MLP ratio": 8,
        "Model size (non-embedding)": 8538240,
        "Num heads": 8,
        "Num layers": 8,
        "Training steps": 13039,
    },
    "10M": {
        "Batch size (sequences)": 32,
        "Dimension": 144,
        "MLP ratio": 8,
        "Model size (non-embedding)": 9900432,
        "Num heads": 8,
        "Num layers": 8,
        "Training steps": 15117,
    },
    "14M": {
        "Batch size (sequences)": 32,
        "Dimension": 192,
        "MLP ratio": 8,
        "Model size (non-embedding)": 14380224,
        "Num heads": 8,
        "Num layers": 8,
        "Training steps": 21953,
    },
    "16M": {
        "Batch size (sequences)": 32,
        "Dimension": 208,
        "MLP ratio": 8,
        "Model size (non-embedding)": 16004560,
        "Num heads": 8,
        "Num layers": 8,
        "Training steps": 24432,
    },
    "20M": {
        "Batch size (sequences)": 64,
        "Dimension": 192,
        "MLP ratio": 8,
        "Model size (non-embedding)": 19101888,
        "Num heads": 8,
        "Num layers": 16,
        "Training steps": 14584,
    },
    "60M": {
        "Batch size (sequences)": 96,
        "Dimension": 384,
        "MLP ratio": 8,
        "Model size (non-embedding)": 57078144,
        "Num heads": 12,
        "Num layers": 16,
        "Training steps": 29042,
    },
    "90M": {
        "Batch size (sequences)": 160,
        "Dimension": 528,
        "MLP ratio": 8,
        "Model size (non-embedding)": 97946640,
        "Num heads": 12,
        "Num layers": 16,
        "Training steps": 29901,
    },
    "150M": {
        "Batch size (sequences)": 192,
        "Dimension": 768,
        "MLP ratio": 8,
        "Model size (non-embedding)": 151898880,
        "Num heads": 12,
        "Num layers": 12,
        "Training steps": 38157,
    },
    "300M": {
        "Batch size (sequences)": 320,
        "Dimension": 1024,
        "MLP ratio": 8,
        "Model size (non-embedding)": 319980544,
        "Num heads": 16,
        "Num layers": 16,
        "Training steps": 45787,
    },
    "530M": {
        "Batch size (sequences)": 448,
        "Dimension": 1344,
        "MLP ratio": 8,
        "Model size (non-embedding)": 530074944,
        "Num heads": 16,
        "Num layers": 16,
        "Training steps": 57786,
    },
    "750M": {
        "Batch size (sequences)": 576,
        "Dimension": 1536,
        "MLP ratio": 8,
        "Model size (non-embedding)": 681297408,
        "Num heads": 16,
        "Num layers": 16,
        "Training steps": 63589,
    },
    "1B": {
        "Batch size (sequences)": 704,
        "Dimension": 2048,
        "MLP ratio": 8,
        "Model size (non-embedding)": 1176832000,
        "Num heads": 16,
        "Num layers": 16,
        "Training steps": 69369,
    },
}
# Assume `model_data` is already defined from the previous response

SEQUENCE_LENGTH = 2048

for model_name, config in model_data.items():
    batch_size = config["Batch size (sequences)"]
    training_steps = config["Training steps"]
    non_embedding_params = config["Model size (non-embedding)"]

    tokens_trained = training_steps * batch_size * SEQUENCE_LENGTH
    token_param_ratio = tokens_trained / non_embedding_params
    learning_rate = 0.0047 * (non_embedding_params / 108000000) ** (-1 / 3)

    config["Sequence Length"] = SEQUENCE_LENGTH
    config["Tokens trained"] = tokens_trained
    config["token_param_ratio_non_embedding"] = token_param_ratio
    config["LR"] = learning_rate

MODEL_CONFIG_DATA = model_data
