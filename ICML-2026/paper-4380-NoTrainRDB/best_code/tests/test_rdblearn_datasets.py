import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
from loguru import logger
logger.enable("rdblearn")

# Ensure modules are loaded so we can patch them
try:
    import relbench.tasks
except ImportError:
    from unittest.mock import MagicMock
    mock_relbench = MagicMock()
    sys.modules["relbench"] = mock_relbench
    sys.modules["relbench.tasks"] = mock_relbench.tasks

import fastdfs.adapter

from rdblearn.datasets import (
    RDBDataset,
    Task,
    TaskMetadata,
    _primary_relbench_metric,
    _hf_salt_label_columns_by_table,
)

class TestRDBDataset(unittest.TestCase):
    @patch('rdblearn.datasets.load_rdb')
    def test_save_load(self, mock_load_rdb):
        import tempfile
        import shutil
        import os
        
        # Create a dummy dataset
        mock_rdb = MagicMock()
        metadata = TaskMetadata(
            key_mappings={"id": "table.id"},
            target_col="target"
        )
        task = Task(
            name="task1",
            train_df=pd.DataFrame({'a': [1]}),
            test_df=pd.DataFrame({'a': [2]}),
            metadata=metadata
        )
        dataset = RDBDataset(rdb=mock_rdb, tasks=[task])
        
        # Create temp dir
        temp_dir = tempfile.mkdtemp()
        try:
            # Save
            dataset.save(temp_dir)
            
            # Verify rdb.save called
            mock_rdb.save.assert_called_with(os.path.join(temp_dir, "rdb"))
            
            # Verify tasks directory structure exists
            task_dir = os.path.join(temp_dir, "tasks", "task1")
            self.assertTrue(os.path.exists(os.path.join(task_dir, "metadata.yaml")))
            self.assertTrue(os.path.exists(os.path.join(task_dir, "train.parquet")))
            
            # Mock load_rdb return
            mock_load_rdb.return_value = mock_rdb
            
            # Load
            loaded_dataset = RDBDataset.load(temp_dir)
            
            # Verify
            self.assertEqual(loaded_dataset.rdb, mock_rdb)
            self.assertIn("task1", loaded_dataset.tasks)
            self.assertEqual(loaded_dataset.tasks["task1"].name, "task1")
            
        finally:
            shutil.rmtree(temp_dir)

    @patch('relbench.tasks')
    @patch('fastdfs.adapter.RelBenchAdapter')
    def test_from_relbench(self, mock_adapter_cls, mock_relbench_tasks):
        # Mock RDB
        mock_rdb = MagicMock()
        mock_rdb.get_table_metadata.return_value.primary_key = "id"
        
        # Mock Adapter
        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.load.return_value = mock_rdb
        
        # Mock Tasks
        mock_relbench_tasks.get_task_names.return_value = ["task1"]
        
        mock_task = MagicMock()
        mock_task.get_table.return_value.df = pd.DataFrame({'col': [1, 2]})
        mock_task.entity_table = "users"
        mock_task.entity_col = "user_id"
        mock_task.target_col = "target"
        mock_task.time_col = "timestamp"
        mock_task.task_type.value = "binary_classification"
        mock_task.metrics = [
            MagicMock(__name__="accuracy"),
            MagicMock(__name__="f1"),
            MagicMock(__name__="roc_auc"),
        ]
        
        mock_relbench_tasks.get_task.return_value = mock_task
        
        # Call method
        dataset = RDBDataset.from_relbench("dummy_dataset")
        
        # Assertions
        self.assertEqual(len(dataset.tasks), 1)
        self.assertIn("task1", dataset.tasks)
        task = dataset.tasks["task1"]
        
        self.assertEqual(task.metadata.key_mappings, {"user_id": "users.id"})
        self.assertEqual(task.metadata.target_col, "target")
        self.assertEqual(task.metadata.evaluation_metric, "roc_auc")
        
        mock_adapter_cls.assert_called_with("dummy_dataset", for_task=None)
        mock_relbench_tasks.get_task_names.assert_called_with("dummy_dataset")

    @patch('relbench.tasks')
    @patch('fastdfs.adapter.RelBenchAdapter')
    def test_from_relbench_prefers_mrr(self, mock_adapter_cls, mock_relbench_tasks):
        mock_rdb = MagicMock()
        mock_rdb.get_table_metadata.return_value.primary_key = "id"
        mock_adapter_cls.return_value.load.return_value = mock_rdb

        mock_relbench_tasks.get_task_names.return_value = ["sales-payterms"]
        mock_task = MagicMock()
        mock_task.get_table.return_value.df = pd.DataFrame({'col': [1, 2]})
        mock_task.entity_table = "sales"
        mock_task.entity_col = "sales_id"
        mock_task.target_col = "target"
        mock_task.time_col = "timestamp"
        mock_task.task_type.value = "multiclass_classification"
        mock_task.metrics = [
            MagicMock(__name__="accuracy"),
            MagicMock(__name__="macro_f1"),
            MagicMock(__name__="micro_f1"),
            MagicMock(__name__="mrr"),
        ]
        mock_relbench_tasks.get_task.return_value = mock_task

        dataset = RDBDataset.from_relbench("rel-salt")
        self.assertEqual(
            dataset.tasks["sales-payterms"].metadata.evaluation_metric, "mrr"
        )

    def test_primary_relbench_metric_prefers_roc_auc_over_accuracy(self):
        names = ["accuracy", "average_precision", "f1", "roc_auc"]
        self.assertEqual(
            _primary_relbench_metric(names, task_type="binary_classification"),
            "roc_auc",
        )

    def test_primary_relbench_metric_regression_prefers_mae(self):
        names = ["r2", "mae", "rmse"]
        self.assertEqual(
            _primary_relbench_metric(names, task_type="regression"),
            "mae",
        )

    @patch('relbench.tasks')
    @patch('fastdfs.adapter.RelBenchAdapter')
    def test_from_relbench_prefers_roc_auc(self, mock_adapter_cls, mock_relbench_tasks):
        mock_rdb = MagicMock()
        mock_rdb.get_table_metadata.return_value.primary_key = "id"
        mock_adapter_cls.return_value.load.return_value = mock_rdb

        mock_relbench_tasks.get_task_names.return_value = ["user-churn"]
        mock_task = MagicMock()
        mock_task.get_table.return_value.df = pd.DataFrame({"col": [1, 2]})
        mock_task.entity_table = "users"
        mock_task.entity_col = "user_id"
        mock_task.target_col = "target"
        mock_task.time_col = "timestamp"
        mock_task.task_type.value = "binary_classification"
        mock_task.metrics = [
            MagicMock(__name__="accuracy"),
            MagicMock(__name__="average_precision"),
            MagicMock(__name__="f1"),
            MagicMock(__name__="roc_auc"),
        ]
        mock_relbench_tasks.get_task.return_value = mock_task

        dataset = RDBDataset.from_relbench("rel-ratebeer")
        self.assertEqual(
            dataset.tasks["user-churn"].metadata.evaluation_metric,
            "roc_auc",
        )

    @patch('relbench.tasks')
    @patch('fastdfs.adapter.RelBenchAdapter')
    def test_from_relbench_skip_link_prediction(self, mock_adapter_cls, mock_relbench_tasks):
        # Mock RDB
        mock_rdb = MagicMock()
        
        # Mock Adapter
        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.load.return_value = mock_rdb
        
        # Mock Tasks: one binary classification, one link prediction
        mock_relbench_tasks.get_task_names.return_value = ["task_ok", "task_skip"]
        
        mock_task_ok = MagicMock()
        mock_task_ok.task_type.value = "binary_classification"
        mock_task_ok.get_table.return_value.df = pd.DataFrame({'col': [1]})
        mock_task_ok.entity_table = "users"
        mock_task_ok.entity_col = "user_id"
        mock_task_ok.target_col = "target"
        mock_task_ok.time_col = "timestamp"
        mock_task_ok.metrics = []
        
        mock_task_skip = MagicMock()
        mock_task_skip.task_type.value = "link_prediction"
        
        def get_task_side_effect(dataset_name, task_name, download=True):
            if task_name == "task_ok":
                return mock_task_ok
            return mock_task_skip
            
        mock_relbench_tasks.get_task.side_effect = get_task_side_effect
        
        # Call method
        dataset = RDBDataset.from_relbench("dummy_dataset")
        
        # Assertions: only task_ok should be present
        self.assertEqual(len(dataset.tasks), 1)
        self.assertIn("task_ok", dataset.tasks)
        self.assertNotIn("task_skip", dataset.tasks)

    @patch('fastdfs.adapter.DBInferAdapter')
    def test_from_4dbinfer(self, mock_adapter_cls):
        # Mock RDB
        mock_rdb = MagicMock()
        
        # Mock Adapter
        mock_adapter = mock_adapter_cls.return_value
        mock_adapter.load.return_value = mock_rdb
        
        # Mock DBBRDBDataset
        mock_dbb_dataset = MagicMock()
        mock_adapter.dataset = mock_dbb_dataset
        
        # Mock Task
        mock_dbb_task = MagicMock()
        mock_dbb_task.train_set = {'col': [1]}
        mock_dbb_task.test_set = {'col': [2]}
        mock_dbb_task.validation_set = {'col': [3]}
        
        mock_task_meta = MagicMock()
        mock_task_meta.name = "task1"
        mock_task_meta.target_column = "target"
        mock_task_meta.time_column = "timestamp"
        # Test both Enum-like and string-like task_type/metric
        mock_task_meta.task_type = "classification"
        mock_task_meta.evaluation_metric = MagicMock(value="auroc")
        
        # Mock columns for key_mappings
        mock_col = MagicMock()
        mock_col.name = "user_id"
        mock_col.dtype = "foreign_key"
        mock_col.link_to = "users.id"
        mock_task_meta.columns = [mock_col]
        
        mock_dbb_task.metadata = mock_task_meta
        mock_dbb_dataset._tasks = [mock_dbb_task]
        
        # Call method
        dataset = RDBDataset.from_4dbinfer("dummy_4db")
        
        # Assertions
        self.assertEqual(len(dataset.tasks), 1)
        self.assertIn("task1", dataset.tasks)
        task = dataset.tasks["task1"]
        
        self.assertEqual(task.metadata.key_mappings, {"user_id": "users.id"})
        self.assertEqual(task.metadata.target_col, "target")
        self.assertEqual(task.metadata.task_type, "classification")
        self.assertEqual(task.metadata.evaluation_metric, "auroc")
        
        mock_adapter_cls.assert_called_with("dummy_4db")
        mock_adapter.load.assert_called_once()

    def test_hf_salt_label_columns_by_table(self):
        task_specs = {
            "sales-office": ("sales", "SALESDOCUMENT", "SALESOFFICE", 1),
            "sales-group": ("sales", "SALESDOCUMENT", "SALESGROUP", 1),
            "item-plant": ("items", "ID", "PLANT", 1),
            "item-incoterms": ("items", "ID", "ITEMINCOTERMSCLASSIFICATION", 1),
        }
        label_cols = _hf_salt_label_columns_by_table(task_specs)
        self.assertEqual(
            label_cols["sales"],
            ["SALESOFFICE", "SALESGROUP"],
        )
        self.assertEqual(
            label_cols["items"],
            ["PLANT", "ITEMINCOTERMSCLASSIFICATION"],
        )

    @patch("datasets.load_dataset")
    def test_from_hf_salt_strips_all_task_labels_from_rdb(self, mock_load_dataset):
        sales_cols = [
            "SALESDOCUMENT",
            "CREATIONDATE",
            "CREATIONTIME",
            "SALESOFFICE",
            "SALESGROUP",
            "CUSTOMERPAYMENTTERMS",
            "SHIPPINGCONDITION",
            "INCOTERMSCLASSIFICATION",
            "SALESDOCUMENTTYPE",
        ]
        items_cols = [
            "SALESDOCUMENT",
            "INCOTERMSCLASSIFICATION",
            "PLANT",
            "SHIPPINGPOINT",
            "PRODUCT",
            "SALESDOCUMENTITEM",
        ]
        customers_cols = ["CUSTOMER", "ADDRESSID"]
        addresses_cols = ["ADDRESSID", "COUNTRY", "REGION"]

        def make_row(cols, i):
            row = {c: f"v{i}" for c in cols}
            row["CREATIONDATE"] = "2024-01-01"
            row["CREATIONTIME"] = "12:00:00"
            return row

        def make_split(rows):
            return MagicMock(to_pandas=lambda: pd.DataFrame(rows))

        sales_train = [make_row(sales_cols, i) for i in range(4)]
        sales_test = [make_row(sales_cols, i + 4) for i in range(2)]
        items_train = [make_row(items_cols, i) for i in range(4)]
        items_test = [make_row(items_cols, i + 4) for i in range(2)]
        customers = [{c: "c1" for c in customers_cols}]
        addresses = [{c: "a1" for c in addresses_cols}]

        def load_dataset_side_effect(_repo, table_name, split, **_kwargs):
            if table_name == "salesdocuments":
                return make_split(sales_train if split == "train" else sales_test)
            if table_name == "salesdocument_items":
                return make_split(items_train if split == "train" else items_test)
            if table_name == "customers":
                return make_split(customers)
            if table_name == "addresses":
                return make_split(addresses)
            raise ValueError(f"unexpected table {table_name!r}")

        mock_load_dataset.side_effect = load_dataset_side_effect

        dataset = RDBDataset.from_hf_salt()

        sales_df = dataset.rdb.get_table_dataframe("sales")
        items_df = dataset.rdb.get_table_dataframe("items")
        for col in [
            "SALESOFFICE",
            "SALESGROUP",
            "CUSTOMERPAYMENTTERMS",
            "SHIPPINGCONDITION",
            "HEADERINCOTERMSCLASSIFICATION",
        ]:
            self.assertNotIn(col, sales_df.columns, f"{col} leaked into sales RDB")
        for col in ["PLANT", "SHIPPINGPOINT", "ITEMINCOTERMSCLASSIFICATION"]:
            self.assertNotIn(col, items_df.columns, f"{col} leaked into items RDB")

        payterms_task = dataset.tasks["sales-payterms"]
        self.assertIn("CUSTOMERPAYMENTTERMS", payterms_task.train_df.columns)
        self.assertIn("CUSTOMERPAYMENTTERMS", payterms_task.test_df.columns)

        office_task = dataset.tasks["sales-office"]
        self.assertIn("SALESOFFICE", office_task.train_df.columns)
        self.assertNotIn("CUSTOMERPAYMENTTERMS", sales_df.columns)

    def test_from_hf_salt_for_task_validates_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown SALT task"):
            RDBDataset.from_hf_salt(for_task="not-a-task")

if __name__ == '__main__':
    unittest.main()
