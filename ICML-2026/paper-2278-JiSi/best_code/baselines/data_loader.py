"""
Benchmark-result data loader for JiSi.

This module provides efficient loading and transformation of collected benchmark
results into the unified records consumed by the JiSi adaptor.
"""

import json
import hashlib
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
from loguru import logger
import re

from .schema import BaselineRecord, BaselineMetadata, AggregatedStats


class BaselineDataLoader:
    """
    Efficient data loader for baseline benchmark results.

    Supports:
    - Iterator-based loading for memory efficiency
    - Batch processing for large datasets
    - Multiple output formats (dict, pandas, parquet)
    - Configurable filtering and caching
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        include_reference_models: bool = False
    ):
        """
        Initialize the baseline data loader.

        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (alternative to config_path)
            include_reference_models: Whether to load reference models alongside main models
        """
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.config = loaded_config.get('baseline', {})
        elif config:
            self.config = config
        else:
            # Use default configuration
            self.config = self._default_config()

        # Extract configuration
        self.results_dir = Path(self.config.get('results_dir', 'results/bench'))
        self.filters = self.config.get('filters', {})
        self.output_config = self.config.get('output', {})
        self.columns_config = self.config.get('columns', {})
        self.include_reference_models = include_reference_models

        # Extract reference models (for aggregation)
        self.reference_models = self.filters.get('reference_models', None) or []

        # Validate no overlap between models and reference_models
        models = self.filters.get('models', None)
        if models and self.reference_models:
            overlapping = set(self.reference_models) & set(models)
            if overlapping:
                raise ValueError(
                    f"reference_models and models cannot overlap. "
                    f"Found duplicates: {sorted(overlapping)}. "
                    f"Please move reference models from 'models' to 'reference_models' only."
                )

        # Cache setup
        self.cache_dir = Path(self.output_config.get('cache_dir', 'baselines/.cache'))
        self.use_cache = self.output_config.get('use_cache', True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized BaselineDataLoader with results_dir={self.results_dir}")

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'results_dir': 'results/bench',
            'filters': {
                'skip_demo': True,
                'datasets': None,
                'models': None,
                'splits': None,
                'exclude_datasets': None,
                'exclude_models': None,
                'exclude_splits': None
            },
            'output': {
                'format': 'parquet',
                'cache_dir': 'baselines/.cache',
                'use_cache': True,
                'batch_size': 10000
            },
            'columns': {
                # Explicit list of columns to include in output
                'include': [
                    'dataset_id', 'split', 'model_name', 'record_index',
                    'prompt', 'score', 'cost'
                ]
            }
        }

    def _calculate_config_hash(self) -> str:
        """Calculate hash of configuration for cache invalidation."""
        config_for_hash = {
            'config': self.config,
            'include_reference_models': self.include_reference_models
        }
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a result file should be skipped based on filters."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Skip demo results if configured
            if self.filters.get('skip_demo', True) and data.get('demo', False):
                return True

            # Apply dataset filter (whitelist has priority)
            if self.filters.get('datasets') is not None:
                # Whitelist: only keep if in the list
                if data.get('dataset_name') not in self.filters['datasets']:
                    return True
            elif self.filters.get('exclude_datasets') is not None:
                # Blacklist: skip if in the exclude list
                if data.get('dataset_name') in self.filters['exclude_datasets']:
                    return True

            # Apply model filter (whitelist has priority)
            # Combine models and reference_models for loading
            models_list = self.filters.get('models', None)
            reference_models_list = self.filters.get('reference_models', None) if self.include_reference_models else None

            if models_list is not None or reference_models_list is not None:
                # Build combined list of all models to load
                all_models_to_load = set()
                if models_list:
                    all_models_to_load.update(models_list)
                if reference_models_list:
                    all_models_to_load.update(reference_models_list)

                # Only load if in the combined list
                if data.get('model_name') not in all_models_to_load:
                    return True
            elif self.filters.get('exclude_models') is not None:
                # Blacklist: skip if in the exclude list
                if data.get('model_name') in self.filters['exclude_models']:
                    return True

            # Apply split filter (whitelist has priority)
            if self.filters.get('splits') is not None:
                # Whitelist: only keep if in the list
                if data.get('split') not in self.filters['splits']:
                    return True
            elif self.filters.get('exclude_splits') is not None:
                # Blacklist: skip if in the exclude list
                if data.get('split') in self.filters['exclude_splits']:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking file {file_path}: {e}")
            return True  # Skip problematic files

    def _get_latest_file_by_timestamp(self, files: List[Path]) -> Path:
        """
        Get the latest file from a list based on filename timestamp.
        Uses YYYYMMDD_HHMMSS pattern in filename, falls back to file modification time.

        Args:
            files: List of file paths

        Returns:
            Path to the latest file
        """
        def extract_timestamp(file_path: Path) -> int:
            match = re.search(r'(\d{8})_(\d{6})\.json$', file_path.name)
            if match:
                # Convert to integer: YYYYMMDDHHMMSS
                return int(match.group(1) + match.group(2))
            # Fallback to file modification time if no timestamp in filename
            return int(file_path.stat().st_mtime)

        return max(files, key=extract_timestamp)

    def find_result_files(self) -> List[Path]:
        """
        Find all result JSON files matching the filters.
        For each dataset/split/model combination, only returns the latest file.
        """
        all_files = list(self.results_dir.rglob('*.json'))
        logger.info(f"Found {len(all_files)} total result files")

        # Apply filters
        filtered_files = [f for f in all_files if not self._should_skip_file(f)]
        skipped_count = len(all_files) - len(filtered_files)
        logger.info(f"After config filtering (skip_demo, datasets, models, splits): {len(filtered_files)} files")
        if skipped_count > 0:
            logger.info(f"  ├─ Skipped {skipped_count} files (demo results or excluded by config)")

        # Group by dataset/split/model, keep only the latest file per group
        grouped = defaultdict(list)
        for f in filtered_files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                key = (data['dataset_name'], data['split'], data['model_name'])
                grouped[key].append(f)
            except Exception as e:
                logger.warning(f"Error grouping file {f}: {e}")
                continue

        # Keep only the latest file per group (by filename timestamp)
        deduplicated = []
        for key, files in grouped.items():
            latest = self._get_latest_file_by_timestamp(files)
            deduplicated.append(latest)

        duplicate_count = len(filtered_files) - len(deduplicated)
        logger.info(f"After deduplication (keep latest per dataset/split/model): {len(deduplicated)} unique combinations")
        if duplicate_count > 0:
            logger.info(f"  ├─ Removed {duplicate_count} duplicate/old version(s)")
        logger.info(f"  └─ Final: {len(deduplicated)} unique dataset/split/model combinations to load")

        return sorted(deduplicated)


    def load_records_iter(self) -> Iterator[BaselineRecord]:
        """
        Iterate over all baseline records.

        Yields:
            BaselineRecord objects one at a time for memory efficiency

        Example:
            >>> loader = BaselineDataLoader()
            >>> for record in loader.load_records_iter():
            ...     print(record.dataset_id, record.model_name, record.score)
        """
        result_files = self.find_result_files()
        logger.info(f"Starting iteration over {len(result_files)} result files")

        total_records = 0

        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                dataset_id = data['dataset_name']
                split = data['split']
                model_name = data['model_name']

                for record_data in data.get('records', []):
                    # Normalize fields to ensure string types for parquet compatibility
                    def normalize_to_string(value):
                        """Convert value to string, handling lists and None."""
                        if value is None:
                            return ""
                        if isinstance(value, (list, dict)):
                            return json.dumps(value)
                        return str(value)

                    # Create BaselineRecord
                    record = BaselineRecord(
                        dataset_id=dataset_id,
                        split=split,
                        model_name=model_name,
                        record_index=record_data['index'],
                        origin_query=normalize_to_string(record_data.get('origin_query', '')),
                        prompt=normalize_to_string(record_data.get('prompt', '')),
                        prediction=normalize_to_string(record_data.get('prediction', '')),
                        raw_output=record_data.get('raw_output'),
                        ground_truth=normalize_to_string(record_data.get('ground_truth', '')),
                        score=record_data['score'] if record_data.get('score') is not None else 0.0,
                        prompt_tokens=record_data.get('prompt_tokens', 0),
                        completion_tokens=record_data.get('completion_tokens', 0),
                        cost=record_data.get('cost', 0.0)
                    )

                    yield record
                    total_records += 1

            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue

        logger.info(f"Finished iteration: {total_records} total records")

    def load_all_records(self) -> List[BaselineRecord]:
        """
        Load all records into memory as a list.

        Warning: May consume significant memory for large datasets.
        Consider using load_records_iter() for large datasets.

        Returns:
            List of BaselineRecord objects
        """
        return list(self.load_records_iter())

    def to_dict_list(self, compact: bool = False) -> List[Dict[str, Any]]:
        """
        Convert all records to a list of dictionaries.

        Args:
            compact: If True, use column selection based on config

        Returns:
            List of dictionaries
        """
        # Get column selection from config
        included_columns = self.columns_config.get('include', None)

        if compact:
            return [record.to_dict_compact(
                included_columns=included_columns
            ) for record in self.load_records_iter()]
        else:
            return [record.to_dict() for record in self.load_records_iter()]

    def to_dataframe(self, use_cache: bool = True):
        """
        Convert all records to a pandas DataFrame.

        Args:
            use_cache: Whether to use cached parquet file if available

        Returns:
            pandas.DataFrame with all baseline records

        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        cache_file = self.cache_dir / "baseline_dataframe.parquet"
        config_hash = self._calculate_config_hash()
        metadata_file = self.cache_dir / "metadata.json"

        # Check cache
        if use_cache and self.use_cache and cache_file.exists():
            try:
                # Verify cache is valid
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('config_hash') == config_hash:
                        logger.info(f"Loading DataFrame from cache: {cache_file}")
                        return pd.read_parquet(cache_file)
                    else:
                        logger.info("Cache invalidated due to config change")
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")

        # Load from scratch
        logger.info("Loading all records into DataFrame...")
        records_dict = self.to_dict_list(compact=True)
        df = pd.DataFrame(records_dict)

        # Cache the DataFrame
        if self.use_cache:
            logger.info(f"Caching DataFrame to {cache_file}")
            df.to_parquet(cache_file, index=False)

            # Save metadata
            metadata = {
                'config_hash': config_hash,
                'total_records': len(df),
                'created_at': pd.Timestamp.now().isoformat()
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        return df

    def to_parquet(self, output_path: Union[str, Path]) -> None:
        """
        Export all records to a Parquet file.

        Args:
            output_path: Path to output parquet file
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_parquet(). "
                "Install with: pip install pandas pyarrow"
            )

        df = self.to_dataframe(use_cache=False)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(df)} records to {output_path}")
        df.to_parquet(output_path, index=False)
        logger.info(f"Successfully wrote parquet file: {output_path}")

    def iter_batches(self, batch_size: Optional[int] = None):
        """
        Iterate over records in batches (as pandas DataFrames).

        Args:
            batch_size: Number of records per batch (default from config)

        Yields:
            pandas.DataFrame for each batch

        Example:
            >>> loader = BaselineDataLoader()
            >>> for batch_df in loader.iter_batches(batch_size=1000):
            ...     process(batch_df)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for iter_batches(). "
                "Install with: pip install pandas"
            )

        if batch_size is None:
            batch_size = self.output_config.get('batch_size', 10000)

        # Get column selection from config
        included_columns = self.columns_config.get('include', None)

        batch = []
        for record in self.load_records_iter():
            batch.append(record.to_dict_compact(
                included_columns=included_columns
            ))

            if len(batch) >= batch_size:
                yield pd.DataFrame(batch)
                batch = []

        # Yield remaining records
        if batch:
            yield pd.DataFrame(batch)

    def get_aggregated_stats(self) -> Dict[str, Dict[str, AggregatedStats]]:
        """
        Compute aggregated statistics grouped by dataset/split and model.

        Returns:
            Nested dict: {dataset/split: {model_name: AggregatedStats}}

        Example:
            >>> stats = loader.get_aggregated_stats()
            >>> print(stats['aime/hybrid']['gpt-4'].avg_score)
            0.85
        """
        logger.info("Computing aggregated statistics...")

        # Group records by dataset/split and model
        grouped = defaultdict(lambda: defaultdict(list))

        for record in self.load_records_iter():
            key = f"{record.dataset_id}/{record.split}"
            grouped[key][record.model_name].append(record)

        # Compute statistics for each group
        results = {}
        for dataset_key, models in grouped.items():
            results[dataset_key] = {}
            for model_name, records in models.items():
                total_records = len(records)
                correct_records = sum(1 for r in records if r.score > 0)
                avg_score = sum(r.score for r in records) / total_records if total_records > 0 else 0.0

                total_cost = sum(r.cost for r in records)
                total_prompt_tokens = sum(r.prompt_tokens for r in records)
                total_completion_tokens = sum(r.completion_tokens for r in records)

                dataset_id, split = dataset_key.split('/')

                stats = AggregatedStats(
                    dataset_id=dataset_id,
                    split=split,
                    model_name=model_name,
                    avg_score=avg_score,
                    total_records=total_records,
                    correct_records=correct_records,
                    total_cost=total_cost,
                    avg_cost_per_record=total_cost / total_records if total_records > 0 else 0.0,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    avg_prompt_tokens=total_prompt_tokens / total_records if total_records > 0 else 0.0,
                    avg_completion_tokens=total_completion_tokens / total_records if total_records > 0 else 0.0
                )

                results[dataset_key][model_name] = stats

        logger.info(f"Computed statistics for {len(results)} dataset/split combinations")
        return results

    def split_by_dataset_then_prompt(
        self,
        records: List[BaselineRecord],
        train_ratio: float = 0.8,
        random_seed: int = 42,
        ood_datasets: Optional[List[str]] = None,
        only_test_datasets: str = 'all',
    ) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
        """
        Split records into train and test sets by dataset, then by prompts within each dataset.

        This method ensures:
        1. First groups all records by dataset_id
        2. Within each dataset, splits prompts (not records) into train/test
        3. Each prompt appears in EITHER train OR test, not both
        4. All model evaluations for the same prompt stay together
        5. OOD datasets (if specified) are all placed in test set

        This prevents data leakage and ensures each dataset has representation in both
        train and test sets (unless it's an OOD dataset).
        
        Args:
            records: List of baseline records to split
            train_ratio: Proportion of prompts for training (0.0-1.0)
            random_seed: Random seed for reproducibility
            ood_datasets: Optional list of dataset IDs to treat as OOD (all go to test)
        
        Returns:
            Tuple of (train_records, test_records)
            
        Example:
            >>> loader = BaselineDataLoader(config_path="config.yaml")
            >>> all_records = loader.load_all_records()
            >>> train, test = loader.split_by_prompt(
            ...     all_records, 
            ...     train_ratio=0.8, 
            ...     random_seed=42,
            ...     ood_datasets=["hle"]
            ... )
        """
        import random
        from .adaptors.common import validate_train_ratio
        if only_test_datasets == 'all':
            has_test_datasets = None
        else:
            has_test_datasets =  [d.strip() for d in only_test_datasets.split(',') if d.strip()]


        validate_train_ratio(train_ratio)

        # Set random seed
        random.seed(random_seed)
        
        # Default OOD datasets to empty list
        ood_datasets = ood_datasets or []
        ood_datasets_set = set(ood_datasets)
        
        # Group records by dataset
        dataset_groups = defaultdict(list)
        for record in records:
            dataset_groups[record.dataset_id].append(record)
        
        logger.info(f"Splitting {len(records)} records across {len(dataset_groups)} datasets")
        logger.info(f"Train ratio: {train_ratio}, Random seed: {random_seed}")
        if ood_datasets_set:
            logger.info(f"OOD datasets (all go to test): {sorted(ood_datasets_set)}")
        
        train_records = []
        test_records = []
        
        # Process each dataset independently
        for dataset_id, dataset_records in dataset_groups.items():
            # Check if this is an OOD dataset
            is_ood = dataset_id in ood_datasets_set
            
            if is_ood:
                # OOD dataset: all records go to test
                test_records.extend(dataset_records)
                logger.debug(f"Dataset {dataset_id} [OOD]: 0 train, {len(dataset_records)} test prompts")
                continue
            
            # Group records by prompt for this dataset
            prompt_to_records = defaultdict(list)
            for record in dataset_records:
                prompt_to_records[record.prompt].append(record)
            
            # Get unique prompts and sort by index for consistency
            unique_prompts = list(prompt_to_records.keys())
            unique_prompts.sort(key=lambda p: min(r.record_index for r in prompt_to_records[p]))
            
            # Split prompts into train/test
            n_train = int(len(unique_prompts) * train_ratio)
            indices = list(range(len(unique_prompts)))
            random.shuffle(indices)
            
            train_indices = set(indices[:n_train])
            test_indices = set(indices[n_train:])
            
            # Collect records for train and test prompts
            dataset_train = []
            dataset_test = []
            for idx, prompt in enumerate(unique_prompts):
                records_for_prompt = prompt_to_records[prompt]
                if idx in train_indices:
                    dataset_train.extend(records_for_prompt)
                else:
                    dataset_test.extend(records_for_prompt)
            
            train_records.extend(dataset_train)
            if has_test_datasets is None:
                test_records.extend(dataset_test)
            else:
                if dataset_id in has_test_datasets:
                    test_records.extend(dataset_test)
            
            logger.debug(
                f"Dataset {dataset_id}: {len(unique_prompts)} prompts -> "
                f"{len(train_indices)} train ({len(dataset_train)} records), "
                f"{len(test_indices)} test ({len(dataset_test)} records)"
            )
        
        logger.info(f"Split complete: {len(train_records)} train records, {len(test_records)} test records")
        
        return train_records, test_records
    
    def create_metadata(self) -> BaselineMetadata:
        """
        Create metadata about the loaded baseline data.

        Returns:
            BaselineMetadata object with summary information
        """
        from datetime import datetime

        # Scan all records
        dataset_splits = defaultdict(set)
        models = set()
        total_records = 0

        for record in self.load_records_iter():
            dataset_splits[record.dataset_id].add(record.split)
            models.add(record.model_name)
            total_records += 1

        # Convert sets to lists for JSON serialization
        dataset_splits_dict = {k: sorted(v) for k, v in dataset_splits.items()}

        return BaselineMetadata(
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            total_records=total_records,
            total_datasets=len(dataset_splits),
            total_models=len(models),
            dataset_splits=dataset_splits_dict,
            models=sorted(models),
            cache_format=self.output_config.get('format', 'parquet'),
            config_hash=self._calculate_config_hash()
        )
