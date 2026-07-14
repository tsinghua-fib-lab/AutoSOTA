"""Tests."""
import os
import shutil
from os import path
import subprocess
import pytest

from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile

from RCAEval.utility import (
    download_re1ob_dataset,
    download_re1ss_dataset,
    download_re1tt_dataset,
    download_re1_dataset,
    
    download_re2ob_dataset,
    download_re2ss_dataset,
    download_re2tt_dataset,
    download_re2_dataset,
    
    download_re3ob_dataset,
    download_re3ss_dataset,
    download_re3tt_dataset,
    download_re3_dataset,
)


@pytest.mark.parametrize("func", [
    download_re1ob_dataset,
    download_re1ss_dataset,
    download_re1tt_dataset,
    download_re1_dataset,
    # download_re2ob_dataset,
    # download_re2ss_dataset,
    # download_re2tt_dataset,
    # download_re2_dataset,
    download_re3ob_dataset,
    download_re3ss_dataset,
    download_re3tt_dataset,
])
def test_download_dataset(func: Callable):
    """Test download dataset."""
    local_path = tempfile.NamedTemporaryFile().name
    func(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)