from typing import List, Optional, Dict
import os
import pickle
import yaml
import pandas as pd
from pydantic import BaseModel
from fastdfs import RDB, load_rdb
from fastdfs.api import create_rdb
from fastdfs.dataset.meta import RDBColumnDType


def _env_flag_true(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _hf_salt_use_local_only(explicit: Optional[bool]) -> bool:
    """Use Hugging Face Datasets cache only (no Hub/network) when True."""
    if explicit is not None:
        return explicit
    return _env_flag_true("RDBLEARN_HF_SALT_LOCAL_ONLY") or _env_flag_true(
        "HF_DATASETS_OFFLINE"
    )


def _hf_salt_label_columns_by_table(
    task_specs: Dict[str, tuple],
) -> Dict[str, List[str]]:
    """Collect every SALT task target column grouped by RDB table name."""
    by_table: Dict[str, List[str]] = {"sales": [], "items": []}
    seen: Dict[str, set] = {"sales": set(), "items": set()}
    for _task_name, (table_name, _entity_col, target_col, _num_test) in task_specs.items():
        if target_col not in seen[table_name]:
            seen[table_name].add(target_col)
            by_table[table_name].append(target_col)
    return by_table


_HF_SALT_TASK_NAMES = frozenset(
    {
        "sales-office",
        "sales-group",
        "sales-payterms",
        "sales-shipcond",
        "sales-incoterms",
        "item-plant",
        "item-shippoint",
        "item-incoterms",
    }
)

class TaskMetadata(BaseModel):
    key_mappings: Dict[str, str]
    target_col: str
    time_col: Optional[str] = None
    task_type: Optional[str] = None
    evaluation_metric: Optional[str] = None

class Task(BaseModel):
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: Optional[pd.DataFrame] = None
    metadata: TaskMetadata

    class Config:
        arbitrary_types_allowed = True


def _primary_relbench_metric(
    metric_names: List[str],
    task_type: Optional[str] = None,
) -> Optional[str]:
    """Choose the primary RelBench metric for sweep evaluation.

    RelBench exposes multiple metrics per task but does not designate one as primary.
    Prefer ranking (MRR), then ROC-AUC for classification, then regression errors.
    """
    if not metric_names:
        return None

    def pick(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            candidate_lower = candidate.lower()
            for name in metric_names:
                if name.lower() == candidate_lower:
                    return name
        return None

    task_type_lower = (task_type or "").lower()
    is_regression = "regression" in task_type_lower

    if pick("mrr") is not None:
        return pick("mrr")

    if not is_regression:
        roc = pick("roc_auc", "auc")
        if roc is not None:
            return roc
        ap = pick("average_precision")
        if ap is not None:
            return ap
        log_metric = pick("log_loss", "cross_entropy")
        if log_metric is not None:
            return log_metric

    if is_regression:
        for candidate in ("mae", "rmse", "r2"):
            found = pick(candidate)
            if found is not None:
                return found

    return metric_names[0]


class RDBDataset:
    def __init__(self, rdb: RDB, tasks: List[Task]):
        self.rdb = rdb
        self.tasks = {t.name: t for t in tasks}

    def save(self, path: str):
        """Save the dataset to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save RDB
        rdb_path = os.path.join(path, "rdb")
        self.rdb.save(rdb_path)
        
        # Save Tasks
        tasks_dir = os.path.join(path, "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        
        for task_name, task in self.tasks.items():
            task_path = os.path.join(tasks_dir, task_name)
            os.makedirs(task_path, exist_ok=True)
            
            # Save DataFrames
            task.train_df.to_parquet(os.path.join(task_path, "train.parquet"))
            task.test_df.to_parquet(os.path.join(task_path, "test.parquet"))
            if task.val_df is not None:
                task.val_df.to_parquet(os.path.join(task_path, "val.parquet"))
            
            # Save Metadata
            task_info = {
                "name": task.name,
                "metadata": task.metadata.model_dump()
            }
            
            with open(os.path.join(task_path, "metadata.yaml"), "w") as f:
                yaml.dump(task_info, f)

    @classmethod
    def load(cls, path: str):
        """Load the dataset from disk."""
        # Load RDB
        rdb_path = os.path.join(path, "rdb")
        rdb = load_rdb(rdb_path)
        
        # Load Tasks
        tasks = []
        tasks_dir = os.path.join(path, "tasks")
        
        if os.path.exists(tasks_dir):
            for task_name in os.listdir(tasks_dir):
                task_path = os.path.join(tasks_dir, task_name)
                if not os.path.isdir(task_path):
                    continue
                
                # Load Metadata
                metadata_path = os.path.join(task_path, "metadata.yaml")
                if not os.path.exists(metadata_path):
                    continue
                    
                with open(metadata_path, "r") as f:
                    task_info = yaml.safe_load(f)
                
                metadata = TaskMetadata(**task_info["metadata"])
                name = task_info["name"]
                
                # Load DataFrames
                train_df = pd.read_parquet(os.path.join(task_path, "train.parquet"))
                test_df = pd.read_parquet(os.path.join(task_path, "test.parquet"))
                
                val_path = os.path.join(task_path, "val.parquet")
                val_df = pd.read_parquet(val_path) if os.path.exists(val_path) else None
                
                tasks.append(Task(
                    name=name,
                    train_df=train_df,
                    test_df=test_df,
                    val_df=val_df,
                    metadata=metadata
                ))
        
        return cls(rdb=rdb, tasks=tasks)

    @classmethod
    def from_relbench(cls, dataset_name: str, for_task: Optional[str] = None):
        try:
            import relbench.tasks
            from fastdfs.adapter import RelBenchAdapter
        except ImportError as e:
            raise ImportError("relbench and fastdfs must be installed to use from_relbench") from e

        # Load RDB
        adapter = RelBenchAdapter(dataset_name, for_task=for_task)
        rdb = adapter.load()

        # Load Tasks
        tasks = []
        task_names = relbench.tasks.get_task_names(dataset_name)
        
        for task_name in task_names:
            # Load task (download=True ensures it's available/generated)
            rb_task = relbench.tasks.get_task(dataset_name, task_name, download=True)
            
            # Skip link prediction tasks as they are not supported yet
            task_type_val = rb_task.task_type.value if hasattr(rb_task.task_type, "value") else rb_task.task_type
            if task_type_val == "link_prediction":
                continue

            # Get tables (convert to pandas)
            train_df = rb_task.get_table("train", mask_input_cols=False).df
            test_df = rb_task.get_table("test", mask_input_cols=False).df
            val_df = rb_task.get_table("val", mask_input_cols=False).df
            
            # Metadata
            entity_table = rb_task.entity_table
            entity_col = rb_task.entity_col
            
            # Find PK of entity table in RDB
            pk = rdb.get_table_metadata(entity_table).primary_key
            
            key_mappings = {entity_col: f"{entity_table}.{pk}"}
            
            metric_name = None
            if rb_task.metrics:
                metric_names = [m.__name__ for m in rb_task.metrics]
                metric_name = _primary_relbench_metric(
                    metric_names,
                    task_type=task_type_val,
                )

            metadata = TaskMetadata(
                key_mappings=key_mappings,
                target_col=rb_task.target_col,
                time_col=rb_task.time_col,
                task_type=rb_task.task_type.value if hasattr(rb_task.task_type, "value") else rb_task.task_type,
                evaluation_metric=metric_name
            )
            
            tasks.append(Task(
                name=task_name,
                train_df=train_df,
                test_df=test_df,
                val_df=val_df,
                metadata=metadata
            ))
            
        return cls(rdb=rdb, tasks=tasks)

    @classmethod
    def from_4dbinfer(cls, dataset_name: str):
        try:
            from fastdfs.adapter import DBInferAdapter
        except ImportError as e:
            raise ImportError("fastdfs must be installed to use from_4dbinfer") from e

        # Load RDB
        adapter = DBInferAdapter(dataset_name)
        rdb = adapter.load()
        
        # The adapter stores the underlying DBBRDBDataset in self.dataset
        dbb_dataset = adapter.dataset
        
        tasks = []
        for dbb_task in dbb_dataset._tasks:
            task_meta = dbb_task.metadata
            
            # Convert numpy dicts to DataFrames
            train_df = pd.DataFrame(dbb_task.train_set)
            test_df = pd.DataFrame(dbb_task.test_set)
            val_df = pd.DataFrame(dbb_task.validation_set)
            
            # Metadata
            # Find key mappings from both foreign keys and primary key + target_table
            key_mappings = {}
            
            # 1. Add mappings for foreign keys in the task table
            for col_schema in task_meta.columns:
                if col_schema.dtype == "foreign_key":
                    # link_to is in format "table.column"
                    link_to = getattr(col_schema, "link_to", None)
                    if link_to:
                        key_mappings[col_schema.name] = link_to
            
            # 2. Also add primary key + target_table mapping if it exists
            # This handles cases like stackexchange::upvote where task.Id -> Posts.Id
            target_table = getattr(task_meta, "target_table", None)
            if target_table:
                # Find primary key column in task data
                task_pk_col = None
                for col_schema in task_meta.columns:
                    if col_schema.dtype == "primary_key":
                        task_pk_col = col_schema.name
                        break
                
                # Only add if not already mapped via FK
                if task_pk_col and task_pk_col not in key_mappings:
                    # Find primary key of target table in RDB
                    try:
                        table_meta = rdb.get_table_metadata(target_table)
                        table_pk = table_meta.primary_key
                        if table_pk:
                            key_mappings[task_pk_col] = f"{target_table}.{table_pk}"
                    except ValueError:
                        # Target table not found in RDB, skip
                        pass
            
            metadata = TaskMetadata(
                key_mappings=key_mappings,
                target_col=task_meta.target_column,
                time_col=task_meta.time_column,
                task_type=task_meta.task_type.value if hasattr(task_meta.task_type, "value") else task_meta.task_type,
                evaluation_metric=task_meta.evaluation_metric.value if hasattr(task_meta.evaluation_metric, "value") else task_meta.evaluation_metric
            )
            
            tasks.append(Task(
                name=task_meta.name,
                train_df=train_df,
                test_df=test_df,
                val_df=val_df,
                metadata=metadata
            ))
        
        # Apply correction for stackexchange upvote task:
        # Increase timestamp column's value by one second
        # This is needed because the timestamp in the original task table match the ones in RDB, so FastDFS cannot join the features.
        if dataset_name == "stackexchange":
            # The data types of these Id columns are float because it's a mix of NaN and int values.
            # Convert them to string
            def convert_series(series: pd.Series) -> pd.Series:
                series.where(series.isna(), series.astype('Int64').astype(str), inplace=True)

            # Convert all primary keys and foreign keys in all tables using convert_series
            for table_name in rdb.table_names:
                table = rdb.tables[table_name]
                metadata = rdb.get_table_metadata(table_name)
                
                for col_schema in metadata.columns:
                    col_name = col_schema.name
                    if col_schema.dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]:
                        if col_name in table.columns:
                            convert_series(table[col_name])
            
            for task in tasks:
                convert_series(task.train_df['Id'])
                convert_series(task.val_df['Id'])
                convert_series(task.test_df['Id'])
                if task.name == "upvote":
                    time_col = task.metadata.time_col
                    if time_col is not None:
                        # Add one second to the timestamp column in train, test, and val dataframes
                        if time_col in task.train_df.columns:
                            task.train_df[time_col] = task.train_df[time_col] + pd.Timedelta(seconds=1)
                        if time_col in task.test_df.columns:
                            task.test_df[time_col] = task.test_df[time_col] + pd.Timedelta(seconds=1)
                        if task.val_df is not None and time_col in task.val_df.columns:
                            task.val_df[time_col] = task.val_df[time_col] + pd.Timedelta(seconds=1)
            
        return cls(rdb=rdb, tasks=tasks)

    @classmethod
    def from_hf_salt(
        cls,
        for_task: Optional[str] = None,
        *,
        hf_local_only: Optional[bool] = None,
    ):
        """Load Hugging Face SALT as an RDBDataset with train/test classification tasks.

        HF SALT has no validation split; ``val_df`` is ``None`` for every task.
        Evaluation metric is MRR (matches the SALT CLI benchmark).

        Every SALT task target column is removed from the shared RDB (``sales`` and
        ``items`` tables) so Deep Feature Synthesis cannot use another task's label as
        a feature. Labels remain available in each task's ``train_df`` / ``test_df``.

        Args:
            for_task: Optional task name (e.g. ``\"sales-payterms\"``) for validation only.
                All SALT task target columns are always removed from the shared RDB so DFS
                cannot read labels from any task while fitting another.
            hf_local_only: If ``True``, load only from the local Hugging Face Datasets cache.
                If ``None``, enabled when ``RDBLEARN_HF_SALT_LOCAL_ONLY`` or
                ``HF_DATASETS_OFFLINE`` is truthy.
        """
        try:
            from datasets import DownloadMode, load_dataset
            from datasets.download.download_config import DownloadConfig
            from datasets.utils.info_utils import VerificationMode
        except ImportError as e:
            raise ImportError("datasets must be installed to use from_hf_salt") from e

        if for_task is not None and for_task not in _HF_SALT_TASK_NAMES:
            raise ValueError(
                f"Unknown SALT task {for_task!r}. Choose one of: {sorted(_HF_SALT_TASK_NAMES)}"
            )

        local_only = _hf_salt_use_local_only(hf_local_only)
        _load_kw: Dict[str, object] = {}
        if local_only:
            _load_kw = {
                "download_config": DownloadConfig(local_files_only=True),
                "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
                "verification_mode": VerificationMode.NO_CHECKS,
            }

        def get_df(table_name: str, split: str = "train") -> pd.DataFrame:
            return load_dataset(
                "sap-ai-research/SALT",
                table_name,
                split=split,
                **_load_kw,
            ).to_pandas()

        sales_train = get_df("salesdocuments", "train")
        sales_test = get_df("salesdocuments", "test")
        items_train = get_df("salesdocument_items", "train")
        items_test = get_df("salesdocument_items", "test")
        customers = get_df("customers", "train")
        addresses = get_df("addresses", "train")

        sales = pd.concat([sales_train, sales_test], axis=0, ignore_index=True)
        items = pd.concat([items_train, items_test], axis=0, ignore_index=True)

        date = sales["CREATIONDATE"].astype(str)
        time = sales["CREATIONTIME"].astype(str)
        sales["CREATIONDATETIME"] = pd.to_datetime(date + " " + time)
        del sales["CREATIONDATE"]
        del sales["CREATIONTIME"]

        items = pd.merge(
            left=items,
            right=sales[["SALESDOCUMENT", "CREATIONDATETIME"]],
            how="left",
            on="SALESDOCUMENT",
        )

        for df in [sales, items, customers, addresses]:
            if "__index_level_0__" in df.columns:
                del df["__index_level_0__"]
        customers = customers.drop_duplicates(subset=["CUSTOMER"], keep="first").copy()
        addresses = addresses.drop_duplicates(subset=["ADDRESSID"], keep="first").copy()

        sales = sales.rename(
            columns={"INCOTERMSCLASSIFICATION": "HEADERINCOTERMSCLASSIFICATION"}
        )
        items = items.rename(
            columns={"INCOTERMSCLASSIFICATION": "ITEMINCOTERMSCLASSIFICATION"}
        )
        items["ID"] = range(len(items))
        items = items.drop(
            columns=["SHIPTOPARTY", "BILLTOPARTY", "PAYERPARTY"],
            errors="ignore",
        )

        sales_num_test = len(sales_test)
        items_num_test = len(items_test)
        tasks = []
        task_specs = {
            "sales-office": ("sales", "SALESDOCUMENT", "SALESOFFICE", sales_num_test),
            "sales-group": ("sales", "SALESDOCUMENT", "SALESGROUP", sales_num_test),
            "sales-payterms": (
                "sales",
                "SALESDOCUMENT",
                "CUSTOMERPAYMENTTERMS",
                sales_num_test,
            ),
            "sales-shipcond": (
                "sales",
                "SALESDOCUMENT",
                "SHIPPINGCONDITION",
                sales_num_test,
            ),
            "sales-incoterms": (
                "sales",
                "SALESDOCUMENT",
                "HEADERINCOTERMSCLASSIFICATION",
                sales_num_test,
            ),
            "item-plant": ("items", "ID", "PLANT", items_num_test),
            "item-shippoint": ("items", "ID", "SHIPPINGPOINT", items_num_test),
            "item-incoterms": (
                "items",
                "ID",
                "ITEMINCOTERMSCLASSIFICATION",
                items_num_test,
            ),
        }

        for task_name, (table_name, entity_col, target_col, num_test) in task_specs.items():
            table_df = sales if table_name == "sales" else items
            train_df = table_df.iloc[:-num_test].copy()
            test_df = table_df.iloc[-num_test:].copy()

            train_df = train_df[pd.notna(train_df[target_col])].copy()
            test_df = test_df[pd.notna(test_df[target_col])].copy()

            metadata = TaskMetadata(
                key_mappings={entity_col: f"{table_name}.{entity_col}"},
                target_col=target_col,
                time_col="CREATIONDATETIME",
                task_type="classification",
                evaluation_metric="mrr",
            )
            tasks.append(
                Task(
                    name=task_name,
                    train_df=train_df,
                    test_df=test_df,
                    val_df=None,
                    metadata=metadata,
                )
            )

        label_cols = _hf_salt_label_columns_by_table(task_specs)
        sales_rdb = sales.drop(columns=label_cols["sales"], errors="ignore")
        items_rdb = items.drop(columns=label_cols["items"], errors="ignore")

        rdb = create_rdb(
            tables={
                "sales": sales_rdb,
                "items": items_rdb,
                "customers": customers,
                "addresses": addresses,
            },
            name="hf-salt",
            primary_keys={
                "sales": "SALESDOCUMENT",
                "items": "ID",
                "customers": "CUSTOMER",
                "addresses": "ADDRESSID",
            },
            foreign_keys=[
                ("items", "SALESDOCUMENT", "sales", "SALESDOCUMENT"),
                ("items", "SOLDTOPARTY", "customers", "CUSTOMER"),
                ("customers", "ADDRESSID", "addresses", "ADDRESSID"),
            ],
            time_columns={
                "sales": "CREATIONDATETIME",
                "items": "CREATIONDATETIME",
            },
            type_hints={
                "addresses": {
                    "COUNTRY": "category",
                    "REGION": "category",
                },
                "items": {
                    "PRODUCT": "text",
                    "SALESDOCUMENTITEM": "category",
                    "SALESDOCUMENTITEMCATEGORY": "category",
                    "PLANT": "category",
                    "SHIPPINGPOINT": "category",
                    "ITEMINCOTERMSCLASSIFICATION": "category",
                },
                "sales": {
                    "SALESDOCUMENTTYPE": "category",
                    "SALESORGANIZATION": "category",
                    "BILLINGCOMPANYCODE": "category",
                    "TRANSACTIONCURRENCY": "category",
                    "SALESOFFICE": "category",
                    "CUSTOMERPAYMENTTERMS": "category",
                    "SHIPPINGCONDITION": "category",
                    "HEADERINCOTERMSCLASSIFICATION": "category",
                },
            },
        )

        return cls(rdb=rdb, tasks=tasks)
