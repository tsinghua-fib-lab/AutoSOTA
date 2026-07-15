"""
Graph construction for simple CNN architectures.
Supports the APPFL CNN model (cnn_dimat.py):
  Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Linear -> ReLU -> Linear
"""

from .base_graph import BIGGraph, NodeType


class CNNGraph(BIGGraph):
    def __init__(self, model):
        super().__init__(model)

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)

        # PREFIX nodes align feature channels within conv or linear blocks.
        # No PREFIX at the conv->linear (flatten) boundary — dimensions are incompatible.
        input_node = self.add_nodes_from_sequence(
            "",
            [
                "conv1",
                "act1",
                "maxpool1",
                NodeType.PREFIX,
                "conv2",
                "act2",
                "maxpool2",
                "fc1",
                "act3",
                NodeType.PREFIX,
                "fc2",
                NodeType.OUTPUT,
            ],
            input_node,
            sep="",
        )

        return self


def cnn(model):
    """Graph for APPFL's DIMAT-compatible CNN."""
    return CNNGraph(model)
