"""
Graph construction for ResNet architectures.
Ported from DIMAT/graphs/resnet_graph.py.
"""

from .base_graph import BIGGraph, NodeType


class ResNetGraph(BIGGraph):
    def __init__(
        self,
        model,
        shortcut_name="shortcut",
        layer_name="layer",
        head_name="linear",
        num_layers=3,
    ):
        super().__init__(model)

        self.shortcut_name = shortcut_name
        self.layer_name = layer_name
        self.num_layers = num_layers
        self.head_name = head_name

    def add_basic_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f".{self.shortcut_name}"
        shortcut_output_node = input_node
        if (
            shortcut_prefix in self.named_modules
            and len(self.get_module(shortcut_prefix)) > 0
        ):
            # There's a break in the skip connection here, so add a new prefix
            input_node = self.add_nodes_from_sequence("", [NodeType.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=["0", "1"],
                input_node=input_node,
            )

        skip_node = self.add_nodes_from_sequence(
            name_prefix=name_prefix,
            list_of_names=[
                "conv1",
                "bn1",
                NodeType.PREFIX,
                "conv2",
                "bn2",
                NodeType.SUM,
            ],
            input_node=input_node,
        )

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_bottleneck_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f".{self.shortcut_name}"
        shortcut_output_node = input_node
        if (
            shortcut_prefix in self.named_modules
            and len(self.get_module(shortcut_prefix)) > 0
        ):
            input_node = self.add_nodes_from_sequence("", [NodeType.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=["0", "1"],
                input_node=input_node,
            )

        skip_node = self.add_nodes_from_sequence(
            name_prefix=name_prefix,
            list_of_names=[
                "conv1",
                "bn1",
                NodeType.PREFIX,
                "conv2",
                "bn2",
                NodeType.PREFIX,
                "conv3",
                "bn3",
                NodeType.SUM,
            ],
            input_node=input_node,
        )

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_layer_nodes(self, name_prefix, input_node):
        source_node = input_node

        for layer_index, block in enumerate(self.get_module(name_prefix)):
            block_class = block.__class__.__name__

            if block_class == "BasicBlock":
                source_node = self.add_basic_block_nodes(
                    name_prefix + f".{layer_index}", source_node
                )
            elif block_class == "Bottleneck":
                source_node = self.add_bottleneck_block_nodes(
                    name_prefix + f".{layer_index}", source_node
                )
            else:
                raise NotImplementedError(block_class)

        return source_node

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        input_node = self.add_nodes_from_sequence(
            "", ["conv1", "bn1"], input_node, sep=""
        )

        for i in range(1, self.num_layers + 1):
            input_node = self.add_layer_nodes(f"{self.layer_name}{i}", input_node)

        input_node = self.add_nodes_from_sequence(
            "",
            [NodeType.PREFIX, "avgpool", self.head_name, NodeType.OUTPUT],
            input_node,
            sep="",
        )

        return self


def resnet20(model):
    return ResNetGraph(model)


def resnet50(model):
    return ResNetGraph(model, shortcut_name="downsample", head_name="fc", num_layers=4)


def resnet18(model):
    return resnet50(model)


def resnet18_appfl(model):
    """Graph for APPFL's ResNet18 (4 layers, 'shortcut' name, 'linear' head)."""
    return ResNetGraph(
        model, shortcut_name="shortcut", head_name="linear", num_layers=4
    )
