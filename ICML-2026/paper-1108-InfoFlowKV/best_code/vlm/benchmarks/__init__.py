"""Benchmark datasets for VLM evaluation."""

from .base import BaseDataset, get_dataset
from .blink import BlinkDataset
from .realworldqa import RealWorldQADataset
from .mmbench import MMBenchDataset
from .chartqa import ChartQADataset
from .docvqa import DocVQADataset
from .ocrbench import OCRBenchDataset
from .mathvista import MathVistaDataset

__all__ = [
    "BaseDataset",
    "get_dataset",
    "BlinkDataset",
    "RealWorldQADataset",
    "MMBenchDataset",
    "ChartQADataset",
    "DocVQADataset",
    "OCRBenchDataset",
    "MathVistaDataset",
]
