from . import *


class Subgraph(data.Data):
    def __inc__(self, key, *args, **kwargs):
        if key in ("index_u", "index_v"): return self.num_node
        elif "index" in key: return self.num_nodes
        else: return 0


def subgraph(graph):
    
    node = torch.arange((N:=graph.num_nodes) ** 2).view(size=(N, N))
    adj = pyg.utils.to_dense_adj(graph.edge_index, max_num_nodes=N).squeeze(0)

    spd = torch.where(~torch.eye(N, dtype=bool) & (adj == 0), torch.full_like(adj, float("inf")), adj)
    for k in range(N): spd = torch.minimum(spd, spd[:, [k]] + spd[[k], :])

    attr, (dst, src) = graph.edge_attr, graph.edge_index
    if attr is not None and attr.ndim == 1: attr = attr[:, None]
    assert graph.x.ndim == 2
    
    stack = lambda *x: torch.stack(torch.broadcast_tensors(*x)).flatten(start_dim=1)

    index_uL = stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None])  # [2, N*E]
    index_vL = stack(node[0] + N * dst[:, None], node[0] + N * src[:, None])
    
    edge_index = torch.cat([index_uL, index_vL], dim=1).long()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index, _ = pyg.utils.coalesce(edge_index, None, num_nodes=N * N)
    
    return Subgraph(
        num_node=N,
        num_nodes=N**2,
        x=graph.x[None].repeat_interleave(N, dim=0).flatten(end_dim=1),
        y=graph.y,
        a=attr[:, None].repeat_interleave(N, dim=1).flatten(end_dim=1) if attr is not None else None,

        e=adj.to(int).flatten(end_dim=1),
        d=spd.to(int).flatten(end_dim=1),

        index_d=node[:, 0] + node[0, :],

        index_u=torch.broadcast_to(node[0, :, None], (N, N)).flatten(),
        index_v=torch.broadcast_to(node[0, None, :], (N, N)).flatten(),

        index_uL=index_uL,
        index_vL=index_vL,
        edge_index=edge_index,

        index_uLF=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None], (N * src + dst)[:, None]),
        index_vLF=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None], (N * dst + src)[:, None]),
    )


def eigen_graph(graph):
    
    count_max_node = 30
    count_min_node = 10
   
    N = graph.num_nodes
    pad_length = count_max_node - N

    edge_index_L, edge_weight_L = pyg.utils.get_laplacian(graph.edge_index, normalization='sym', num_nodes=N)
    L = pyg.utils.to_dense_adj(edge_index_L, edge_attr=edge_weight_L, max_num_nodes=N).squeeze(0)
    adj = pyg.utils.to_dense_adj(graph.edge_index, max_num_nodes=N).squeeze(0)

    #############################################
    #              SPD                          #
    #############################################

    spd = torch.where(~torch.eye(N, dtype=bool) & (adj == 0), torch.full_like(adj, float("inf")), adj)
    for k in range(N): spd = torch.minimum(spd, spd[:, [k]] + spd[[k], :])
    
    # S_pad = F.pad(spd, (0, pad_length, 0, pad_length))
    spd = torch.clamp(spd, 0, max=5)
    S_pad = F.pad(spd, (0, pad_length, 0, pad_length)).long()

    E, U = torch.linalg.eigh(L)
    E_pad = F.pad(E, (0, pad_length))
    U_pad = F.pad(U, (0, pad_length, 0, pad_length))

    mask_1d = torch.arange(count_max_node) < N              # [1, 1, 0, 0]
    
    mask_2d = mask_1d[:, None] & mask_1d[None, :]           # [1, 1, 0, 0]
                                                            # [1, 1, 0, 0]
                                                            # [0, 0, 0, 0]
                                                            # [0, 0, 0, 0]
 
    mask_edge = F.pad(adj, (0, pad_length, 0, pad_length))  # [0, 1, 0, 0]
                                                            # [1, 0, 0, 0]
                                                            # [0, 0, 0, 0]
                                                            # [0, 0, 0, 0]

    #############################################
    #              Edge Attr                    #
    #############################################
 
    if graph.edge_attr is not None:
        attr_pad = pyg.utils.to_dense_adj(edge_index=data.edge_index, 
                                edge_attr=data.edge_attr, 
                                max_num_nodes=count_max_node).squeeze(0)
        attr_pad = attr_pad[:, :, None].long()
    else:
        attr_pad = None
        
    #############################################
    #              Label                        #
    #############################################

    if graph.y.shape[0] == N*N:  # edge
        y = graph.y.view(N, N)
        y = F.pad(y, (0, pad_length, 0, pad_length)).float()
    elif graph.y.shape[0] == N:  # node
        y = F.pad(graph.y, (0, pad_length)).float()
    else:                        # graph
        y = graph.y.float()

    #############################################
    #              Node Attr                    #
    #############################################
    
    X_pad = F.pad(graph.x, (0, 0, 0, pad_length)).long()

    # X:    [B, N, N, d]
    # E:    [B, N]
    # U:    [B, N, N]
    # S:    [B, N, N]
    # A:    [B, N, N, d]
    # Y:    [B, N, N] or [B, N] or [B, 1]
    # M1:   [B, N]
    # M2:   [B, N, N]
    # Me:   [B, N, N]
 
    return X_pad, E_pad, U_pad, S_pad, attr_pad, y, \
        mask_1d.long(), mask_2d.long(), mask_edge.long()
