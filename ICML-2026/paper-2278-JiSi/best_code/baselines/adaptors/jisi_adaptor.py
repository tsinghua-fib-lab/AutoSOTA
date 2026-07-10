import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from loguru import logger

from baselines.data_loader import BaselineDataLoader
from baselines.schema import BaselineRecord
from .common import get_unique_models, fill_missing_models_scores, log_filled_statistics


class JiSiAdaptor:
    def __init__(
        self,
        config_path: str,
        only_test_datasets: str = 'all',
        random_seed: int = 42,
        train_ratio: float = 0.8,
        ood_datasets: Optional[List[str]] = None
    ):
        self.config_path = config_path
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.ood_datasets = ood_datasets or []
        self.loader = BaselineDataLoader(config_path=config_path)
        self.only_test_datasets = only_test_datasets

        logger.info(f"Initialized JiSi adaptor with seed={random_seed}, ratio={train_ratio}")
        if self.ood_datasets:
            logger.info(f"OOD datasets: {self.ood_datasets}")

    def _convert_records_to_jsonl_format(
        self,
        records: List[BaselineRecord],
        all_models: List[str]
    ) -> List[Dict]:
        """
        Convert benchmark records to JiSi JSONL format.

        Args:
            records: List of baseline records
            all_models: List of all model names

        Returns:
            List of dictionaries in JiSi format
        """
        from collections import Counter

        # Group records by (dataset, prompt)
        # Note: Removed record_index to ensure consistent prompt counting across adaptors
        prompt_groups = defaultdict(list)
        for record in records:
            # Use dataset_id and prompt as key (without record_index)
            key = (record.dataset_id, record.prompt)
            prompt_groups[key].append(record)

        # Convert to JSONL format
        jsonl_data = []
        filled_counter = Counter()

        for (dataset_id, prompt), prompt_records in prompt_groups.items():
            # Build records dict (model_name -> score)
            records_dict = {}
            usages_dict = {}
            raw_dict = {}

            # Get base index from records (use minimum for consistency)
            index = min(r.record_index for r in prompt_records)

            for record in prompt_records:
                records_dict[record.model_name] = record.score
                usages_dict[record.model_name] = {
                    'completion_tokens': record.completion_tokens,
                    'cost': record.cost,
                    'prompt_tokens': record.prompt_tokens
                }
                raw_dict[record.model_name] = record.raw_output

            # Fill missing models with 0.0 score and zero usage
            filled = fill_missing_models_scores(records_dict, all_models, fill_value=0.0)
            for model in filled:
                # Also fill usage dict for filled models
                usages_dict[model] = {
                    'completion_tokens': 0,
                    'cost': 0.0,
                    'prompt_tokens': 0
                }
                raw_dict[model] = ""
                filled_counter[model] += 1

            jsonl_data.append({
                'query': prompt,
                'dataset': dataset_id,
                'index': index,
                'records': records_dict,
                'usages': usages_dict,
                'raw_output': raw_dict,
                "gt": record.ground_truth,
                "split": record.split,
            })

        # Log filling statistics
        log_filled_statistics(filled_counter)

        # Sort by dataset and index for consistency
        jsonl_data.sort(key=lambda x: (x['dataset'], x['index']))

        return jsonl_data

    def _write_jsonl(self, file_path: Path, data: List[Dict]):
        """
        Write data to JSONL file.

        Args:
            file_path: Output file path
            data: List of dictionaries to write
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Wrote {len(data)} lines to {file_path}")

    def convert(self, output_dir: str = "data/jisi") -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all records
        logger.info("Loading baseline records...")
        all_records = self.loader.load_all_records()
        logger.info(f"Loaded {len(all_records)} records")

        # Split by dataset then prompt (to ensure each dataset has train/test representation)
        train_records, test_records = self.loader.split_by_dataset_then_prompt(
            all_records,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            ood_datasets=self.ood_datasets,
            only_test_datasets=self.only_test_datasets
        )

        logger.info(f"Split: {len(train_records)} train records, {len(test_records)} test records")

        # Get all unique models from entire baseline
        all_models = get_unique_models(all_records)
        logger.info(f"Found {len(all_models)} unique models")

        # Create output directory with seed and split ratio
        seed_str = f"seed{self.random_seed}"
        split_str = f"split{self.train_ratio:.1f}"
        output_path = output_path / f"{seed_str}_{split_str}"
        output_path.mkdir(parents=True, exist_ok=True)


        train_file = output_path / "train.jsonl"
        test_file = output_path / "test.jsonl"

        # Convert and write train data
        logger.info(f"Writing train data to {train_file}")
        train_data = self._convert_records_to_jsonl_format(train_records, all_models)
        self._write_jsonl(train_file, train_data)

        # Convert and write test data
        logger.info(f"Writing test data to {test_file}")
        test_data = self._convert_records_to_jsonl_format(test_records, all_models)
        self._write_jsonl(test_file, test_data)

        # Generate baseline scores file (based on test set only)
        baseline_file = output_path / "baseline_scores.json"
        logger.info(f"Generating baseline scores to {baseline_file}")
        baseline_scores = self._generate_baseline_scores(test_records)
        self._write_baseline_scores(baseline_file, baseline_scores)

        logger.info("JiSi conversion complete!")

        return {
            'train': str(train_file),
            'test': str(test_file),
            'baseline_scores': str(baseline_file)
        }

    def _generate_baseline_scores(self, records: List[BaselineRecord]) -> Dict[str, Dict[str, float]]:
        """
        Generate baseline scores: model performance on each dataset.

        Args:
            records: List of baseline records (should be test records only for fair comparison)

        Returns:
            Dictionary mapping {model_name: {dataset_id: avg_score}}
        """
        # Aggregate scores by model and dataset
        model_dataset_scores = defaultdict(lambda: defaultdict(list))

        for record in records:
            model_dataset_scores[record.model_name][record.dataset_id].append(record.score)

        # Calculate average scores
        baseline_scores = {}
        for model_name, datasets in model_dataset_scores.items():
            baseline_scores[model_name] = {}
            for dataset_id, scores in datasets.items():
                # Calculate average and convert to percentage (0-100 scale)
                avg_score = sum(scores) / len(scores) if scores else 0.0
                baseline_scores[model_name][dataset_id] = round(avg_score * 100, 2)

        logger.info(f"Generated baseline scores for {len(baseline_scores)} models across datasets")
        return baseline_scores

    def _write_baseline_scores(self, file_path: Path, baseline_scores: Dict[str, Dict[str, float]]):
        """
        Write baseline scores to JSON file.

        Args:
            file_path: Output file path
            baseline_scores: Baseline scores dictionary
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_scores, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote baseline scores to {file_path}")

def main():
    """Main entry point for JiSi adaptor."""
    parser = argparse.ArgumentParser(
        description='Convert benchmark result files to JiSi train/test JSONL files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/baseline_config.yaml',
        help='Path to baseline configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/test splitting'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (0.0-1.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/jisi',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--ood-datasets',
        type=str,
        default='',
        help='Comma-separated list of dataset IDs to treat as OOD (all go to test). Example: hle'
    )
    parser.add_argument(
        '--only-test-datasets',
        type=str,
        default='all',
        help='Comma-separated list of dataset IDs to point out the only test datasets (all go to test). If all, '
             'test all datasets Default: all'
    )

    args = parser.parse_args()

    ood_datasets = []
    if args.ood_datasets:
        ood_datasets = [d.strip() for d in args.ood_datasets.split(',') if d.strip()]

    # Create adaptor and convert
    adaptor = JiSiAdaptor(
        only_test_datasets=args.only_test_datasets,
        config_path=args.config,
        random_seed=args.seed,
        train_ratio=args.split_ratio,
        ood_datasets=ood_datasets
    )

    output_files = adaptor.convert(output_dir=args.output_dir)

    logger.info("Generated files:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")

    logger.info("\nTo use these files, update your JiSi config:")
    logger.info(f'  "train_data_path": "{output_files["train"]}"')
    logger.info(f'  "test_data_path": "{output_files["test"]}"')
    logger.info(f'  "baseline_scores_path": "{output_files["baseline_scores"]}"')


if __name__ == '__main__':
    main()
