# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from pathlib import Path

from typing import Union, Optional

import torch
import torch.nn as nn


from . import vision_transformer as vits

logger = logging.getLogger("dinov3")




