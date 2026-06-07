"""High-level Iris model modules: encoder, decoder, fusion, and task encoding."""

from .encoder import UNet_Encoder
from .decoder import UNet_Decoder
from .task_encoding import TaskEncodingLayer_SubPixel

from .fusion import (
    PriorFusionLayer,
    MultiPriorFusionLayer,
    HierarchyPriorClassifier
)

__all__ = [
    # Encoder/Decoder
    'UNet_Encoder',
    'UNet_Decoder',
    
    # Task encoding
    'TaskEncodingLayer_SubPixel',

    # Fusion
    'PriorFusionLayer',
    'MultiPriorFusionLayer',
    'HierarchyPriorClassifier'
]
