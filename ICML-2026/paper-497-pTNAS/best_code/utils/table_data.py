import os
import pandas as pd
from dataclasses import dataclass
import pickle

# Monkey-patch torch.load for PyTorch >= 2.6 compatibility
import torch as _torch
_original_torch_load_td = _torch.load


def _patched_torch_load_td(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load_td(f, *args, **kwargs)


_torch.load = _patched_torch_load_td

import torch_frame
from torch_frame import stype
from relbench.base import TaskType

from typing import Union, Optional, Dict
from torch_frame.config import (
    ImageEmbedderConfig,
    TextEmbedderConfig,
    TextTokenizerConfig,
)

TextEmbedderCFG = Optional[TextEmbedderConfig]
TextTokenizerCFG = Union[dict[str, TextTokenizerConfig],
                         TextTokenizerConfig, None]
ImageEmbedderCFG = Union[dict[str, ImageEmbedderConfig],
                         ImageEmbedderConfig, None]

DEFAULT_COL_DICT_NAME = "col_types.pkl"
ColTypeDictHint = Union[Dict[str, Dict[str, stype]], Dict[str, stype]]


def load_col_types(
        cache_path: str,
        file_name: str = DEFAULT_COL_DICT_NAME
) -> ColTypeDictHint:
    """
    Load the column types from the cache.
    :param cache_path: The path to the cache file.
    :return: A dictionary of column types.
    """
    path = os.path.join(cache_path, file_name)
    col_type = pickle.load(open(path, "rb"))
    return col_type


def save_col_types(
        cache_path: str,
        col_type_dict: ColTypeDictHint,
        file_name: str = DEFAULT_COL_DICT_NAME
) -> None:
    """
    Save the column types to the cache.
    :param cache_path: The path to the cache file.
    :param col_type: A dictionary of column types.
    """
    path = os.path.join(cache_path, file_name)
    pickle.dump(col_type_dict, open(path, "wb"))


@dataclass
class TableData(object):
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    col_to_stype: Dict[str, stype]
    target_col: str
    task_type: TaskType

    def __post_init__(self):
        self.is_materialize = False

    def materilize(
            self,
            col_to_sep: Optional[dict[str, Optional[str]]] = None,
            col_to_text_embedder_cfg: Optional[TextEmbedderCFG] = None,
            col_to_text_tokenizer_cfg: TextEmbedderCFG = None,
            col_to_image_embedder_cfg: TextTokenizerCFG = None,
            col_to_time_format: ImageEmbedderCFG = None,
    ):
        if self.is_materialize:
            return

        train_dataset = torch_frame.data.Dataset(
            df=self.train_df,
            col_to_stype=self.col_to_stype,
            target_col=self.target_col,
            col_to_sep=col_to_sep,
            col_to_text_embedder_cfg=col_to_text_embedder_cfg,
            col_to_text_tokenizer_cfg=col_to_text_tokenizer_cfg,
            col_to_image_embedder_cfg=col_to_image_embedder_cfg,
            col_to_time_format=col_to_time_format,
        ).materialize()

        self._train_tf = train_dataset.tensor_frame
        self._col_stats = train_dataset.col_stats
        self._val_tf = train_dataset.convert_to_tensor_frame(
            self.val_df)
        self._test_tf = train_dataset.convert_to_tensor_frame(
            self.test_df)

        self.is_materialize = True

    def save_to_dir(
            self,
            dir_path: str
    ):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        train_df_path = os.path.join(dir_path, "train.csv")
        val_df_path = os.path.join(dir_path, "val.csv")
        test_df_path = os.path.join(dir_path, "test.csv")
        self.train_df.to_csv(train_df_path, index=False)
        self.val_df.to_csv(val_df_path, index=False)
        self.test_df.to_csv(test_df_path, index=False)
        save_col_types(dir_path, self.col_to_stype)

        with open(os.path.join(dir_path, "target_col.txt"), "w") as f:
            f.write(self.target_col + "\n")
            f.write(self.task_type.name + "\n")

        # check if is materialize
        if self.is_materialize:
            # save the tensorframe
            train_tf_path = os.path.join(dir_path, "train_tf.pt")
            val_tf_path = os.path.join(dir_path, "val_tf.pt")
            test_tf_path = os.path.join(dir_path, "test_tf.pt")
            torch_frame.save(self.train_tf, self.col_stats, train_tf_path)
            torch_frame.save(self.val_tf, None, path=val_tf_path)
            torch_frame.save(self.test_tf, None, path=test_tf_path)

    @staticmethod
    def load_from_dir(
            dir_path: str,
    ):

        train_df_path = os.path.join(dir_path, "train.csv")
        val_df_path = os.path.join(dir_path, "val.csv")
        test_df_path = os.path.join(dir_path, "test.csv")
        train_df = pd.read_csv(train_df_path, index_col=False)
        val_df = pd.read_csv(val_df_path, index_col=False)
        test_df = pd.read_csv(test_df_path, index_col=False)
        col_to_stype = load_col_types(dir_path)
        with open(os.path.join(dir_path, "target_col.txt"), "r") as f:
            target_col = f.readline().strip()
            task_type = TaskType[f.readline().strip()]

        table_data = TableData(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            col_to_stype=col_to_stype,
            target_col=target_col,
            task_type=task_type,
        )

        # check if there is train_tf.pt or others
        train_tf_path = os.path.join(dir_path, "train_tf.pt")
        val_tf_path = os.path.join(dir_path, "val_tf.pt")
        test_tf_path = os.path.join(dir_path, "test_tf.pt")
        if os.path.exists(train_tf_path):
            assert os.path.exists(val_tf_path)
            assert os.path.exists(test_tf_path)

            table_data.is_materialize = True
            # update the train_tf, val_tf, test_tf, col_stats
            table_data._train_tf, table_data._col_stats = torch_frame.load(
                path=train_tf_path)
            table_data._val_tf, _ = torch_frame.load(path=val_tf_path)
            table_data._test_tf, _ = torch_frame.load(path=test_tf_path)
            print(f" ==> load material dataset from {dir_path}")
        else:
            table_data.is_materialize = False
            print(
                f" ==> load raw dataset from {dir_path}, need material first")
        return table_data

    @property
    def train_tf(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._train_tf

    @property
    def val_tf(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._val_tf

    @property
    def test_tf(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._test_tf

    @property
    def col_stats(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self._col_stats

    @property
    def col_names_dict(self):
        if not self.is_materialize:
            raise ValueError(
                "The tensor frame is not materialized. Please call materilize() first."
            )
        return self.train_tf.col_names_dict
