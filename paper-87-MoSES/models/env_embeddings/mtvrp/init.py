import torch
import torch.nn as nn

class MTVRPInitEmbedding(nn.Module):
    """Initial embedding MTVRP.

    Note: this is the same as what MTPOMO and MVMoE use.

    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - durations: duration of the nodes (TW)

    Global features:
        - loc: x, y euclidean coordinates of depot
    """

    def __init__(
        self, embed_dim=128, bias=False, **kw
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MTVRPInitEmbedding, self).__init__()

        # Depot feats (includes global features): x, y, distance, backhaul_class, open_route
        global_feat_dim = 2
        self.project_global_feats = nn.Linear(global_feat_dim, embed_dim, bias=bias)

        # Customer feats: x, y, demand_linehaul, demand_backhaul, time_window_early, time_window_late, durations
        customer_feat_dim = 7
        self.project_customers_feats = nn.Linear(customer_feat_dim, embed_dim, bias=bias)

        self.embed_dim = embed_dim

    def forward(self, td):
        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        global_feats = td["locs"][:, :1, :]

        # Customers (batch, N, 5) -> (batch, N, embed_dim)
        # note that these feats include the depot (but unused) so we exclude the first node
        cust_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ),
            -1,
        )

        # If some features are infinity (e.g. distance limit is inf because of no limit), replace with 0 so that it does not affect the embedding
        global_feats = torch.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.nan_to_num(cust_feats, nan=0.0, posinf=0.0, neginf=0.0)
        global_embeddings = self.project_global_feats(
            global_feats
        )  # [batch, 1, embed_dim]
        cust_embeddings = self.project_customers_feats(
            cust_feats
        )  # [batch, N, embed_dim]
        return torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, N+1, embed_dim]


# Note that this is the most recent version and should be used from now on. Others can be based on this!
class MTVRPInitEmbeddingRouteFinderBase(nn.Module):
    """General Init embedding class

    Args:
        num_global_feats: number of global features
        num_cust_feats: number of customer features
        embed_dim: embedding dimension
        bias: whether to use bias in the linear layers
        posinf_val: value to replace positive infinity values with
    """

    def __init__(
        self, num_global_feats, num_cust_feats, embed_dim=128, bias=False, posinf_val=0.0
    ):
        super(MTVRPInitEmbeddingRouteFinderBase, self).__init__()
        self.project_global_feats = nn.Linear(num_global_feats, embed_dim, bias=bias)
        self.project_customers_feats = nn.Linear(num_cust_feats, embed_dim, bias=bias)
        self.embed_dim = embed_dim
        self.posinf_val = posinf_val

    def _global_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _cust_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def forward(self, td):
        global_feats = self._global_feats(td)
        cust_feats = self._cust_feats(td)

        global_feats = torch.nan_to_num(global_feats, posinf=self.posinf_val)
        cust_feats = torch.nan_to_num(cust_feats, posinf=self.posinf_val)
        global_embeddings = self.project_global_feats(global_feats)
        cust_embeddings = self.project_customers_feats(cust_feats)

        return torch.cat((global_embeddings, cust_embeddings), -2)


class MTVRPInitEmbeddingRouteFinder(MTVRPInitEmbeddingRouteFinderBase):
    """
    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - service_time: service time of the nodes
    Global features:
        - open_route (O)
        - distance_limit (L)
        - (end) time window of depot
        - x, y euclidean coordinates of depot
    The above features are embedded in the depot node as global and get broadcasted via attention.
    This allows the network to learn the relationships between them.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(
            num_global_feats=5,  # x, y, open_route, distance_limit, time_window_depot
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

    def _cust_feats(self, td):
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )


class MTVRPInitEmbeddingM(MTVRPInitEmbeddingRouteFinder):
    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        # Note: here we add the backhaul_class as a feature
        MTVRPInitEmbeddingRouteFinderBase.__init__(
            self,
            num_global_feats=5 + 1,
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        glob_feats = super(MTVRPInitEmbeddingM, self)._global_feats(td)
        is_mixed_backhaul = (td["backhaul_class"] == 2).float()
        return torch.cat([glob_feats, is_mixed_backhaul[..., None]], -1)



class MTVRPPromptEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 embed_dim: int = 128,
                 normalization: str = None,
                 norm_params: bool = True,
                 linear_bias: bool = True,
                 ):
        super(MTVRPPromptEmbedding, self).__init__()
        self.prompt_l1 = nn.Linear(input_dim, embed_dim, bias=linear_bias)
        self.prompt_l2 = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.prompt_l3 = nn.Linear(embed_dim * 2, embed_dim, bias=linear_bias)

        if normalization == None:
            self.normalization = nn.Identity()
        elif normalization == 'layer':
            self.normalization = nn.LayerNorm(embed_dim, elementwise_affine=norm_params)
        else:
            raise NotImplementedError

    def forward(self, x, init_h):
        output = self.prompt_l2(self.normalization(self.prompt_l1(x)))
        output = torch.cat((init_h, output[:, None].expand_as(init_h)), dim=-1)
        output = self.prompt_l3(output)
        return output




class MultiBranchInitEmbedding(nn.Module):
    def __init__(
        self, embed_dim=128, bias=False, posinf_val=0.0,
    ):
        super(MultiBranchInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.posinf_val = posinf_val

        self.project_global_open_route = nn.Linear(1, embed_dim, bias=bias)
        self.project_global_locs = nn.Linear(2, embed_dim, bias=bias)
        self.project_global_distance_limit = nn.Linear(1, embed_dim, bias=bias)
        self.project_global_time_windows = nn.Linear(1, embed_dim, bias=bias)
        self.project_local_locs = nn.Linear(2, embed_dim, bias=bias)
        self.project_local_demand_linehaul = nn.Linear(1, embed_dim, bias=bias)
        self.project_local_demand_backhaul = nn.Linear(1, embed_dim, bias=bias)
        self.project_local_time_windows = nn.Linear(2, embed_dim, bias=bias)
        self.project_local_service_time = nn.Linear(1, embed_dim, bias=bias)

    def _global_feats(self, td):
        # open_route: (batch, )
        # locs: (batch, graph, 2)
        # distance_limit: (batch, )
        # time_windows: (batch, graph, 2)
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

    def _cust_feats(self, td):
        # demand_linehaul: (batch, graph)
        # demand_backhaul: (batch, graph)
        # service_time: (batch, graph)
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )

    def forward(self, td):
        global_feats = self._global_feats(td)
        cust_feats = self._cust_feats(td)

        global_feats = torch.nan_to_num(global_feats, posinf=self.posinf_val)
        cust_feats = torch.nan_to_num(cust_feats, posinf=self.posinf_val)

        global_open_route, global_locs, global_distance_limit, global_time_windows = torch.split(
            global_feats, (1, 2, 1, 1), dim=-1
        )
        local_locs, local_demand_linehaul, local_demand_backhaul, local_time_windows, local_service_time = torch.split(
            cust_feats, (2, 1, 1, 2, 1), dim=-1
        )
        global_open_route_embed = self.project_global_open_route(global_open_route)
        global_locs_embed = self.project_global_locs(global_locs)
        global_distance_limit_embed = self.project_global_distance_limit(global_distance_limit)
        global_time_windows_embed = self.project_global_time_windows(global_time_windows)
        local_locs_embed = self.project_local_locs(local_locs)
        local_demand_linehaul_embed = self.project_local_demand_linehaul(local_demand_linehaul)
        local_demand_backhaul_embed = self.project_local_demand_backhaul(local_demand_backhaul)
        local_time_windows_embed = self.project_local_time_windows(local_time_windows)
        local_service_time_embed = self.project_local_service_time(local_service_time)
        global_embeddings = global_open_route_embed + global_locs_embed + \
                            global_distance_limit_embed + global_time_windows_embed
        local_embeddings = local_locs_embed + local_demand_linehaul_embed + local_demand_backhaul_embed + \
                           local_time_windows_embed + local_service_time_embed

        init_embeddings = torch.cat((global_embeddings, local_embeddings), -2)
        return init_embeddings
