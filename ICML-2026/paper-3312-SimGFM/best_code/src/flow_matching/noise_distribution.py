import torch

from src import utils


class NoiseDistribution:

    def __init__(self, model_transition, output_dims, portion_manager):

        self.x_num_classes = output_dims["X"]
        self.e_num_classes = output_dims["E"]
        self.y_num_classes = output_dims["y"]
        self.transition = model_transition

        if model_transition == "uniform":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "absorbfirst":
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[0] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[0] = 1

        elif model_transition == "argmax":
            node_types = portion_manager.get_proportion("node_types").float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = portion_manager.get_proportion("edge_types").float()
            e_marginals = edge_types / torch.sum(edge_types)

            x_max_dim = torch.argmax(x_marginals)
            e_max_dim = torch.argmax(e_marginals)
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[x_max_dim] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "absorbing":
            raise NotImplementedError("absorbing transition is not implemented")

        elif model_transition == "marginal":

            node_types = portion_manager.get_proportion("node_types").float()
            x_limit = node_types / torch.sum(node_types)

            edge_types = portion_manager.get_proportion("edge_types").float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "edge_marginal":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes

            edge_types = portion_manager.get_proportion("edge_types").float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

            node_types = portion_manager.get_proportion("node_types").float()
            x_limit = node_types / torch.sum(node_types)

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # typically dummy
        print(
            f"Limit distribution of the classes | Nodes: {x_limit} | Edges: {e_limit}"
        )
        self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)


    def get_limit_dist(self):
        return self.limit_dist

    def get_noise_dims(self):
        return {
            "X": len(self.limit_dist.X),
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        return X, E, y

    def add_virtual_classes(self, X, E, y=None):
        return new_X, new_E, new_y
