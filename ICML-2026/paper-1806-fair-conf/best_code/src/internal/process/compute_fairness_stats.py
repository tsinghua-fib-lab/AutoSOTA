import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import sys

from substantive.faircp.fairness.llm.llm_result_stats import compute_comprehensive_fairness_statistics

LABEL_MAPS = {
    'bios': {
        0: "Professor",
        1: "Physician",
        2: "Photographer",
        3: "Journalist",
        4: "Psychologist",
        5: "Teacher",
        6: "Dentist",
        7: "Surgeon",
        8: "Painter",
        9: "Model",
    },

    'ravdess': {
        1: "Neutral",
        2: "Calm",
        3: "Happy",
        4: "Sad",
        5: "Angry",
        6: "Fearful",
        7: "Disgust",
        8: "Surprised",
    },

    'facet': {
        1: "Backpacker",
        2: "Boatman",
        3: "Computer User",
        4: "Craftsman",
        5: "Farmer",
        6: "Guard",
        7: "Guitarist",
        8: "Gymnast",
        9: "Hairdresser",
        10: "Horse Rider",
        11: "Laborer",
        12: "Officer",
        13: "Motorcyclist",
        14: "Painter",
        15: "Repairman",
        16: "Salesperson",
        17: "Singer",
        18: "Skateboarder",
        19: "Speaker",
        20: "Tennis Player",
    },

    'acs-education': {
            0: "No schooling (+ primar _ high school only)",
            1: "High School - no college",
            2: "GED - no college",
            3: "Started college/associates",
            4: "Bachelor's Degree",
            5: "Grad School/Professional Degree",
    },

    'acs-income': {
            0: "104 - 9000",
            1: "9000 - 20000",
            2: "20000 - 30000",
            3: "30000 - 38800",
            4: "38800 - 48450",
            5: "48450 - 60000",
            6: "60000 - 75000",
            7: "75000 - 96900",
            8: "96900 - 140000",
            9: "140000 - 1672000",
    },

    'credit': {0: "0", 1: "1", 2: "2", 3: "3"}

}

class MockPrediction:
    """Mock prediction object to match the expected interface."""
    def __init__(self, row):
        self.index = row['index']
        self.method = row['method']
        self.group_text = row['group_text']
        self.label_text = row['label_text']
        self.result = row['result']
        self.conformal_set = row['conformal_set']
        self.difficulty = row['difficulty']

def csv_to_predictions(csv_path: str) -> List[MockPrediction]:
    """Convert CSV file to list of prediction objects."""
    df = pd.read_csv(csv_path)
    predictions = []

    for _, row in df.iterrows():
        predictions.append(MockPrediction(row))

    return predictions

def get_label_map_and_name(dataset_name: str) -> tuple[Dict[int, str], str]:
    dataset_name = dataset_name.lower()

    if dataset_name in LABEL_MAPS:
        return LABEL_MAPS[dataset_name], dataset_name
    else:
        #print(f"New dataset need to add: {dataset_name}")
        print(f"Available datasets: {list(LABEL_MAPS.keys())}")
        # Default fallback
        return {i: f"label_{i}" for i in range(10)}, dataset_name

def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def run_compute_statistics(cfg):
    """Main function to compute statistics from existing CSV."""

    # Get paths
    logs_dir = Path("logs")
    source_folder = logs_dir / _cfg_get(cfg, 'statistics_result_dataset')
    csv_file = source_folder / "llm_individual_result.csv"

    # Create output folder
    output_folder = source_folder / "comprehensive_statistics"
    output_folder.mkdir(exist_ok=True)

    print(f"Computing comprehensive statistics...")
    print(f"Source: {csv_file}")
    print(f"Output: {output_folder}")

    # Check if CSV exists
    if not csv_file.exists():
        print(f"!!Error: {csv_file} not found!")
        return

    # Convert CSV to predictions
    predictions = csv_to_predictions(str(csv_file))
    print(f"Loaded {len(predictions)} predictions")

    # Get dataset name from config
    dataset_name = cfg.get('statistics_dataset')
    label_map, clean_dataset_name = get_label_map_and_name(dataset_name)

    print(f"Using '{clean_dataset_name}' dataset with {len(label_map)} labels")
    print(f"Labels: {list(label_map.values())}")

    # Run comprehensive statistics
    results = compute_comprehensive_fairness_statistics(
        predictions=predictions,
        label_map=label_map,
        output_dir=str(output_folder),
        dataset_name=clean_dataset_name
    )

    if results:
        print(f"Statistics computed successfully!")
        print(f"Results saved to: {output_folder}")
    else:
        print(f"!!Error computing statistics")
