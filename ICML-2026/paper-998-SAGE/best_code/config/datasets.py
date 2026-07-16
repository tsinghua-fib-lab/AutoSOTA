"""
Dataset Configuration File
Defines dataset configurations for all tasks
"""

DATASET_CONFIG = {
    "image_classification": {
        "CIFAR-100": {
            "dataset_name": "cifar100",
            "split": "test",
            "num_samples": None,  # None means use all data
        },
        "CIFAR-10": {
            "dataset_name": "cifar10",
            "split": "test",
            "num_samples": None,
        },
        "ImageNet-1k": {
            "dataset_name": "imagenet-1k",
            "split": "validation",
            "num_samples": None,
        },
        # Can add more datasets later
        # "ImageNet": {...},
    },
    
    "text_classification": {
        "AG_News": {
            "dataset_name": "fancyzhx/ag_news",
            "split": "train",
            "num_samples": None,
        },
        "MMLU": {
            "dataset_name": "cais/mmlu",
            "split": "test",
            "num_samples": None,
        },
        # Can add more later
        # "IMDB": {...},
    },
    
    "llm_generation": {
        "TruthfulQA": {
            "dataset_name": "truthfulqa/truthful_qa",
            "split": "validation",
            "num_samples": None,
        },
        "HaluEval": {
            "dataset_name": "pminervini/HaluEval",
            "subset": "dialogue",  # dialogue, qa, summarization
            "split": "data",
            "num_samples": None,
        },
    },
    
    "vlm_tagging": {
        "Flickr30k": {
            "dataset_name": "flickr30k",
            "split": "test",
            "num_samples": None,
        },
        "COCO": {
            "dataset_name": "lmms-lab/COCO-Caption",
            "split": "val",
            "num_samples": None,
        },
        # Can add more later
        # "COCO": {...},
    }
}

# Dataset class information
DATASET_CLASSES = {
    "CIFAR-100": 100,
    "AG_News": 4,
    "Flickr30k": None,  # Captioning task has no fixed classes
}
