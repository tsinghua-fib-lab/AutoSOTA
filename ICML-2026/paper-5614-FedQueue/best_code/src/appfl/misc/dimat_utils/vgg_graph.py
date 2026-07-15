"""
Graph construction for VGG architectures.
Ported from DIMAT/graphs/vgg_graph.py.
"""

from .base_graph import BIGGraph, NodeType


class VGGGraph(BIGGraph):
    def __init__(self, model, architecture):
        super().__init__(model)
        self.architecture = architecture

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        node_insert = NodeType.PREFIX
        graph = []

        graph_idx = 0
        for arch_idx, elem in enumerate(self.architecture):
            if elem == "M":
                graph.append("features." + str(graph_idx))  # MaxPool2d
                graph_idx += 1
            else:
                graph.append("features." + str(graph_idx))  # Conv2d
                graph.append("features." + str(graph_idx + 1))  # ReLU
                graph.append(node_insert)
                graph_idx += 2
        graph.append("features." + str(graph_idx))  # AvgPool2d
        graph.append("classifier")  # Linear
        graph.append(NodeType.OUTPUT)
        self.add_nodes_from_sequence("", graph, input_node, sep="")

        return self


def vgg11(model):
    architecture = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    return VGGGraph(model, architecture)


def vgg16(model):
    architecture = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]
    return VGGGraph(model, architecture)
