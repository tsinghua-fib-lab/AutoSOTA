from . import *

class MLP(nn.Sequential):

    def __init__(self, idim: int, odim: int, hdim: int=None, norm: bool=True):
        super().__init__()
        hdim = hdim or idim
        self.add_module("input", nn.Linear(idim, hdim, bias=not norm))
        self.add_module("relu", relu:=nn.Sequential())
        if norm: relu.add_module("norm", nn.BatchNorm1d(hdim))
        relu.add_module("activate", nn.ReLU())
        self.add_module("output", nn.Linear(hdim, odim, bias=not norm))


# --------------------------------- EMBEDDING -------------------------------- #

class NodeEmbedding(nn.Module):
    def __init__(self, dim: int, max_dis: int, enc: Optional[nn.Module]):
        super().__init__()
        self.max_dis = max_dis
        self.embed_v = enc and enc(dim)
        self.embed_d = nn.Embedding(max_dis + 1, dim)

    def forward(self, batch):
        x = self.embed_v(batch.x) if self.embed_v else 0
        d = self.embed_d(torch.clamp(batch.d, 0, max=self.max_dis))
        batch.x = x + d
        del batch.d
        return batch


class EdgeEmbedding(nn.Module):
    def __init__(self, dim: int, enc: Optional[nn.Module]):
        super().__init__()
        self.embed = enc and enc(dim)

    def forward(self, message, attrs=None):
        if not self.embed: return F.relu(message)
        return F.relu(message + self.embed(attrs))


# --------------------------------- AGGREGATE -------------------------------- #

def LF(agg: str, aggL: List[str], gin: bool=True):

    """
        Take agg='Lv' for example:
        FFN((1 + eps) * h[u,v] + NormReLU(\sum_{w\in N(u)} ReLU(FFN(h[w,v]) + e_{uw})))
    """
    
    class _(nn.Module):
        def __init__(self, idim: int, odim: int, enc: Optional[nn.Module], bn: bool):
            super().__init__()
            self.enc = EdgeEmbedding(idim, enc)
            self.linear = nn.Linear(idim, idim)
            self.mlp = MLP(idim, odim, norm=bn)
            self.eps = nn.Parameter(torch.zeros(1))

        def forward(self, super, batch):
            dst, *src = batch[f"index_{agg}"]
            attrs = "a" in batch and batch["a"]

            f = lambda agg_f, src_f: torch.index_select(super[agg_f].linear(batch.x), dim=0, index=src_f)
            message = self.enc(sum(map(f, aggL, src)), attrs)
            aggregate = pys.scatter(message, dim=0, index=dst, dim_size=len(batch.x))

            return self.mlp((batch.x * (1. + self.eps) if gin else 0.) + aggregate)

    return _

def G(agg: str, gin: bool=True):

    """
        Take agg='Gv' for example:
        FFN((1 + eps) * h[u,v] + NormReLU(\sum_{w} FFN(h[w,v])))
    """
    
    class _(nn.Module):
        def __init__(self, idim: int, odim: int, bn: bool):
            super().__init__()
            # self.enc = Edge(idim, False)
            self.mlp = MLP(idim, odim, norm=bn)
            self.eps = nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            idx = batch[f"index_{agg}"]
            x = torch.index_select(pys.scatter(batch.x, dim=0, index=idx), dim=0, index=idx)
            return self.mlp((batch.x * (1. + self.eps) if gin else 0.) + x)

    return _

def L(agg: str, gin: bool=True):

    """
        Take agg='Lv' for example:
        FFN((1 + eps) * h[u,v] + NormReLU(\sum_{w\in N(u)} ReLU(FFN(h[w,v]) + e_{uw})))
    """
    
    class _(nn.Module):
        def __init__(self, idim: int, odim: int, enc: Optional[nn.Module], bn: bool):
            super().__init__()
            self.enc = EdgeEmbedding(idim, enc)
            # self.linear = nn.Linear(idim, idim)
            self.mlp = MLP(idim, odim, norm=bn)
            self.eps = nn.Parameter(torch.zeros(1))
            self.conv = gnn.ChebConv(idim, idim, K=5)

        def forward(self, batch):
            # dst, src = batch[f"index_{agg}"]
            attrs = "a" in batch and batch["a"]
 
            # message = self.enc(torch.index_select(self.linear(batch.x), dim=0, index=src), attrs)
            # aggregate = pys.scatter(message, dim=0, index=dst, dim_size=len(batch.x))
            
            aggregate = self.conv(batch.x, batch.edge_index, lambda_max=2.0)
            
            return self.mlp((batch.x * (1. + self.eps) if gin else 0.) + aggregate)

    return _

def LWL(aggL: List[str]=[], aggG: List[str]=[], gin: bool=True):

    class _(nn.ModuleDict):
        def __init__(self, idim: int, odim: int, enc: Optional[nn.Module], bn: bool):
            super().__init__()
            for agg in aggL: self[agg] = L(agg, gin)(idim, odim, enc, bn=bn)
            for agg in aggG: self[agg] = G(agg, gin)(idim, odim, bn=bn)
            self.bn = nn.BatchNorm1d(odim) if bn else nn.Identity()
            
        def forward(self, batch):
            xL = sum(self[agg](batch) for agg in aggL)
            xG = sum(self[agg](batch) for agg in aggG)
            batch.x = F.relu(self.bn(xL + xG))
            return batch

    return _

def FWL(aggL: List[str]=[], aggG: List[str]=[], gin: bool=True):

    class _(nn.ModuleDict):
        def __init__(self, idim: int, odim: int, enc: Optional[nn.Module], bn: bool):
            super().__init__()
            for agg in aggL: self[agg] = LF(agg, aggL, gin)(idim, odim, enc, bn=bn)
            for agg in aggG: self[agg] = G(agg, gin)(idim, odim, bn=bn)
            self.bn = nn.BatchNorm1d(odim) if bn else nn.Identity()

        def forward(self, batch):
            xL = sum(self[agg](self, batch) for agg in aggL)
            xG = sum(self[agg](batch) for agg in aggG)
            batch.x = F.relu(self.bn(xL + xG))
            return batch

    return _


# ---------------------------------- POOLING --------------------------------- #

def Pool(gin: bool=True):
    
    class _(nn.Module):

        def __init__(self, idim: int, odim: int, task: str, bn: bool):
            super().__init__()
            self.task = task
            self.pooling = task != "e" and MLP(idim, idim, norm=bn)
            self.predict = MLP(idim, odim, hdim=2*idim, norm=False)
            self.eps = nn.Parameter(torch.zeros(1))
            self.bn = nn.BatchNorm1d(idim) if bn else nn.Identity()

        def forward(self, batch):
            if self.task == "e":
                x = batch.x
            else:
                x = pys.scatter(batch.x, dim=0, index=batch.index_u)
                if gin:
                    x = torch.index_select(batch.x, dim=0, index=batch.index_d) * (1. + self.eps) + x
                x = F.relu(self.bn(self.pooling(x)))
                if self.task == "g":
                    x = pys.scatter(torch.index_select(x, 0, batch.index_u), batch.batch, dim=0, reduce="mean")

            return self.predict(x)
        
    return _

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

class GNN(nn.Sequential):

    def __init__(self, args, task: str, enc_a: Optional[nn.Module], enc_e: Optional[nn.Module], odim: int, bn: bool=True):
        super().__init__()

        if args.model == "MP": Layer = LWL(aggL=["vL"])
        elif args.model == "Sub": Layer = LWL(aggL=["uL"])
        elif args.model == "L": Layer = LWL(aggL=["uL", "vL"])
        elif args.model == "LF": Layer = FWL(aggL=["uLF", "vLF"])
        # elif args.model == "L-G": Layer = LWL(aggL=["uL", "vL"], aggG=["u", "v"])
        # elif args.model == "LF-G": Layer = FWL(aggL=["uLF", "vLF"], aggG=["u", "v"])
        elif args.model == "Sub-G": Layer = LWL(aggL=["uL"], aggG=["u"])
        elif args.model == "L-G": Layer = LWL(aggL=["uL", "vL"], aggG=["v"])
        elif args.model == "LF-G": Layer = FWL(aggL=["uLF", "vLF"], aggG=["v"])
        else: raise NotImplementedError

        self.add_module("in", NodeEmbedding(idim:=args.dim_embed, args.max_dis, enc_a))

        for i in range(args.num_layer):
            self.add_module(f"A{i}", Layer(idim, idim, enc_e, bn=bn))

        self.add_module("out", Pool(gin=False)(idim, odim, task, bn=bn))
