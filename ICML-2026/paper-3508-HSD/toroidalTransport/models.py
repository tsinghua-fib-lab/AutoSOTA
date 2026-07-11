import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import toponetx as tnx

try:
    from torch_geometric.nn import MessagePassing
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from neuraloperator.models import FNO
    NEURAL_OPERATOR_AVAILABLE = True
except ImportError:
    NEURAL_OPERATOR_AVAILABLE = False


class TorchFluxMapper(nn.Module):
    
    def __init__(self, points, faces, device):
        super().__init__()
        
        sc = tnx.SimplicialComplex(faces)
        B1_scipy = sc.incidence_matrix(rank=1, signed=True)
        B2_scipy = sc.incidence_matrix(rank=2, signed=False)
        
        if B1_scipy.shape[1] != len(points):
            B1_scipy = B1_scipy.T
            
        if B2_scipy.shape[0] != B1_scipy.shape[0]: 
            B2_scipy = B2_scipy.T
        
        B1_abs = np.abs(B1_scipy).tocoo()
        indices = torch.from_numpy(np.vstack((B1_abs.row, B1_abs.col))).long()
        values = torch.from_numpy(B1_abs.data).float()
        shape = B1_abs.shape
        self.register_buffer('abs_B1_indices', indices)
        self.register_buffer('abs_B1_values', values)
        self.abs_B1_shape = shape
        
        pts_np = points.astype(np.float32)
        
        edge_vectors_np = B1_scipy @ pts_np
        
        f0 = pts_np[faces[:, 0]]
        f1 = pts_np[faces[:, 1]]
        f2 = pts_np[faces[:, 2]]
        face_normals_raw = np.cross(f1 - f0, f2 - f0)
        face_norms = np.linalg.norm(face_normals_raw, axis=1, keepdims=True) + 1e-9
        face_normals_unit = face_normals_raw / face_norms
        
        edge_normals_sum = np.abs(B2_scipy) @ face_normals_unit
        edge_norms = np.linalg.norm(edge_normals_sum, axis=1, keepdims=True) + 1e-9
        edge_normals_unit = edge_normals_sum / edge_norms
        
        flux_vectors_np = np.cross(edge_normals_unit, edge_vectors_np)
        
        self.register_buffer('flux_vectors', torch.from_numpy(flux_vectors_np).float().to(device))
        
        self.n_edges = shape[0]
        self.n_nodes = shape[1]
        
        print(f"[TorchFluxMapper] Initialized (True Flux Mode): {self.n_nodes} nodes -> {self.n_edges} edges")

    def forward(self, node_vectors):
        B, N, D = node_vectors.shape
        device = node_vectors.device
        
        x_flat = node_vectors.permute(1, 0, 2).reshape(N, -1)
        
        abs_B1 = torch.sparse_coo_tensor(
            self.abs_B1_indices.to(device), 
            self.abs_B1_values.to(device), 
            self.abs_B1_shape, 
            device=device
        )
        
        u_sum = torch.sparse.mm(abs_B1, x_flat)
        u_mid = u_sum.reshape(self.n_edges, B, 3).permute(1, 0, 2) / 2.0
        
        flux = torch.sum(u_mid * self.flux_vectors.unsqueeze(0), dim=-1)
        
        return flux


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, radius=0.1):
        super().__init__()
        self.radius = radius

        self.message_net = nn.Sequential(
            nn.Linear(in_channels * 2 + 3, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

        self._adj_cache = None
        self._pos_cache = None

    def _build_adjacency(self, pos):
        N = pos.shape[0]
        device = pos.device

        pos_diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.norm(pos_diff, dim=-1)

        adj_mask = (dist < self.radius).float()
        adj_mask = adj_mask * (1 - torch.eye(N, device=device))

        degree = adj_mask.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj_mask / degree

        return adj_norm

    def forward(self, x, pos):
        B, C, N = x.shape

        if self._adj_cache is None or self._pos_cache is None or not torch.equal(pos, self._pos_cache):
            self._adj_cache = self._build_adjacency(pos)
            self._pos_cache = pos.clone()

        adj_norm = self._adj_cache

        out_list = []
        for b in range(B):
            x_b = x[b].T
            x_agg = torch.mm(adj_norm, x_b)
            pos_agg = torch.mm(adj_norm, pos)
            rel_pos = pos_agg - pos

            x_combined = torch.cat([x_b, x_agg, rel_pos], dim=-1)
            messages = self.message_net(x_combined)
            out = self.update_net(torch.cat([x_b, messages], dim=-1))
            out_list.append(out.T)

        return torch.stack(out_list, dim=0)


class GNO(nn.Module):
    
    def __init__(self, in_channels, out_channels=1, hidden_channels=32,
                 projection_channels=32, n_layers=2, radius=0.1):
        super().__init__()

        self.lifting = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1)
        )

        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_channels, hidden_channels, radius)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, projection_channels, 1),
            nn.GELU(),
            nn.Conv1d(projection_channels, out_channels, 1)
        )

    def forward(self, x, pos):
        h = self.lifting(x)

        for layer, norm in zip(self.graph_layers, self.norms):
            h_new = layer(h, pos)
            h_new = norm(h_new.transpose(1, 2)).transpose(1, 2)
            h = h + h_new

        out = self.projection(h)
        return out




class FNO3d(nn.Module):
    
    def __init__(self, in_channels, out_channels=1, hidden_channels=8,
                 n_modes=(4, 4, 4), n_layers=3):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.n_layers = n_layers
        
        if NEURAL_OPERATOR_AVAILABLE:
            self.use_neuralop = True
            self.fno = FNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers
            )
        else:
            self.use_neuralop = False
            self.lifting = nn.Conv3d(in_channels, hidden_channels, 1)
            
            self.spectral_layers = nn.ModuleList([
                SpectralConv3dForHSD(hidden_channels, hidden_channels, *n_modes)
                for _ in range(n_layers)
            ])
            
            self.local_layers = nn.ModuleList([
                nn.Conv3d(hidden_channels, hidden_channels, 1)
                for _ in range(n_layers)
            ])
            
            self.norms = nn.ModuleList([
                nn.BatchNorm3d(hidden_channels)
                for _ in range(n_layers)
            ])
            
            self.projection = nn.Sequential(
                nn.Conv3d(hidden_channels, hidden_channels, 1),
                nn.GELU(),
                nn.Conv3d(hidden_channels, out_channels, 1)
            )

    def forward(self, x):
        if self.use_neuralop:
            return self.fno(x)
        else:
            h = self.lifting(x)
            
            for i, (spectral, local, norm) in enumerate(zip(
                self.spectral_layers, self.local_layers, self.norms
            )):
                h1 = spectral(h)
                h2 = local(h)
                h = norm(h1 + h2)
                if i < self.n_layers - 1:
                    h = F.gelu(h)
            
            out = self.projection(h)
            return out


if TORCH_GEOMETRIC_AVAILABLE:
    class MGNBlock(MessagePassing):
        def __init__(self, hidden_dim):
            super().__init__(aggr='add')
            
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )

        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        def message(self, x_i, x_j, edge_attr):
            tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
            return self.edge_mlp(tmp)

        def update(self, aggr_out, x):
            tmp = torch.cat([x, aggr_out], dim=1)
            return self.node_mlp(tmp)


    class LightweightMGN(nn.Module):
        
        def __init__(self, node_in_dim, edge_in_dim, out_dim=1, hidden_dim=64, num_layers=3):
            super().__init__()
            
            self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
            self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
            
            self.layers = nn.ModuleList([
                MGNBlock(hidden_dim) for _ in range(num_layers)
            ])
            
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

        def forward(self, x, edge_index, edge_attr):
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            
            for layer in self.layers:
                delta_x = layer(x, edge_index, edge_attr)
                x = x + delta_x
                
            return self.decoder(x)
else:
    class LightweightMGN(nn.Module):
        
        def __init__(self, node_in_dim, edge_in_dim, out_dim=1, hidden_dim=64, num_layers=3):
            super().__init__()
            
            self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
            self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
            
            self.edge_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ) for _ in range(num_layers)
            ])
            
            self.node_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ) for _ in range(num_layers)
            ])
            
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            
            self.num_layers = num_layers

        def forward(self, x, edge_index, edge_attr):
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            
            src, dst = edge_index[0], edge_index[1]
            
            for i in range(self.num_layers):
                edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
                edge_attr = self.edge_mlps[i](edge_input)
                
                N = x.shape[0]
                aggr = torch.zeros(N, edge_attr.shape[1], device=x.device)
                aggr.index_add_(0, dst, edge_attr)
                
                node_input = torch.cat([x, aggr], dim=1)
                delta_x = self.node_mlps[i](node_input)
                x = x + delta_x
                
            return self.decoder(x)


class DeepONet(nn.Module):
    
    def __init__(self, branch_input_dim, trunk_input_dim, output_dim=1,
                 branch_layers=[128, 128, 64], trunk_layers=[64, 64, 64],
                 basis_dim=64):
        super().__init__()
        
        self.output_dim = output_dim
        self.basis_dim = basis_dim
        
        branch_net = []
        in_dim = branch_input_dim
        for h_dim in branch_layers:
            branch_net.append(nn.Linear(in_dim, h_dim))
            branch_net.append(nn.GELU())
            in_dim = h_dim
        branch_net.append(nn.Linear(in_dim, basis_dim * output_dim))
        self.branch_net = nn.Sequential(*branch_net)
        
        trunk_net = []
        in_dim = trunk_input_dim
        for h_dim in trunk_layers:
            trunk_net.append(nn.Linear(in_dim, h_dim))
            trunk_net.append(nn.GELU())
            in_dim = h_dim
        trunk_net.append(nn.Linear(in_dim, basis_dim * output_dim))
        self.trunk_net = nn.Sequential(*trunk_net)
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, branch_input, trunk_input):
        B = branch_input.shape[0]
        N = trunk_input.shape[0]
        
        branch_out = self.branch_net(branch_input)
        branch_out = branch_out.view(B, self.output_dim, self.basis_dim)
        
        trunk_out = self.trunk_net(trunk_input)
        trunk_out = trunk_out.view(N, self.output_dim, self.basis_dim)
        
        output = torch.einsum('bod,nod->bno', branch_out, trunk_out)
        output = output + self.bias
        
        return output


class GeoFNO(nn.Module):
    
    def __init__(self, modes=8, width=20, layers=4, grid_res=32, out_channels=1):
        super(GeoFNO, self).__init__()
        
        self.grid_res = grid_res
        self.width = width
        self.out_channels = out_channels
        self.n_layers = layers
        self.modes = modes
        
        self.s_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

        self.fc0 = nn.Linear(3 + 1, width)

        if NEURAL_OPERATOR_AVAILABLE:
            self.use_neuralop = True
            self.fno_core = FNO(
                n_modes=(modes, modes, modes),
                hidden_channels=width,
                in_channels=width,
                out_channels=width,
                n_layers=layers
            )
        else:
            self.use_neuralop = False
            self.spectral_convs = nn.ModuleList([
                SpectralConv3dForHSD(width, width, modes, modes, modes)
                for _ in range(layers)
            ])
            self.ws = nn.ModuleList([
                nn.Conv3d(width, width, 1)
                for _ in range(layers)
            ])
            self.norms = nn.ModuleList([
                nn.BatchNorm3d(width)
                for _ in range(layers)
            ])

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def trilinear_splat(self, features, coords):
        B, N, C = features.shape
        device = features.device
        G = self.grid_res
        
        coords_scaled = coords * (G - 1)
        coords_floor = coords_scaled.floor()
        coords_frac = coords_scaled - coords_floor
        
        grid = torch.zeros(B, C, G, G, G, device=device)
        weight_sum = torch.zeros(B, 1, G, G, G, device=device)
        
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    wx = (1 - coords_frac[..., 0]) if dx == 0 else coords_frac[..., 0]
                    wy = (1 - coords_frac[..., 1]) if dy == 0 else coords_frac[..., 1]
                    wz = (1 - coords_frac[..., 2]) if dz == 0 else coords_frac[..., 2]
                    w = wx * wy * wz
                    
                    idx_x = (coords_floor[..., 0] + dx).long().clamp(0, G - 1)
                    idx_y = (coords_floor[..., 1] + dy).long().clamp(0, G - 1)
                    idx_z = (coords_floor[..., 2] + dz).long().clamp(0, G - 1)
                    
                    flat_idx = idx_x * G * G + idx_y * G + idx_z
                    
                    weighted_feat = features * w.unsqueeze(-1)
                    weighted_feat_perm = weighted_feat.permute(0, 2, 1)
                    flat_idx_expanded = flat_idx.unsqueeze(1).expand(-1, C, -1)
                    
                    grid_flat = grid.view(B, C, -1)
                    grid_flat.scatter_add_(2, flat_idx_expanded, weighted_feat_perm)
                    
                    weight_sum_flat = weight_sum.view(B, 1, -1)
                    w_expanded = w.unsqueeze(1)
                    flat_idx_w = flat_idx.unsqueeze(1)
                    weight_sum_flat.scatter_add_(2, flat_idx_w, w_expanded)
        
        grid = grid.view(B, C, G, G, G)
        weight_sum = weight_sum.view(B, 1, G, G, G).clamp(min=1e-6)
        grid = grid / weight_sum
        
        return grid

    def forward(self, x_coords, features):
        B, N, _ = x_coords.shape
        device = x_coords.device
        
        coords_flat = x_coords.reshape(B * N, 3)
        eta = self.s_net(coords_flat)
        eta = eta.reshape(B, N, 3)
        
        x_combined = torch.cat([x_coords, features], dim=-1)
        x_lifted = self.fc0(x_combined)
        
        grid = self.trilinear_splat(x_lifted, eta)
        
        if self.use_neuralop:
            x = self.fno_core(grid)
        else:
            x = grid
            for i, (conv, w, norm) in enumerate(zip(self.spectral_convs, self.ws, self.norms)):
                x1 = conv(x)
                x2 = w(x)
                x = norm(x1 + x2)
                if i < self.n_layers - 1:
                    x = F.gelu(x)
        
        eta_sample = eta * 2 - 1
        eta_sample = eta_sample.view(B, N, 1, 1, 3)
        
        x_sampled = F.grid_sample(x, eta_sample, mode='bilinear',
                                  padding_mode='zeros', align_corners=True)
        x_sampled = x_sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        x_out = self.fc1(x_sampled)
        x_out = F.gelu(x_out)
        x_out = self.fc2(x_out)
        
        return x_out




class SpectralConv3dForHSD(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3dForHSD(nn.Module):
    
    def __init__(self, in_channels, out_channels=1, hidden_channels=8,
                 n_modes=(4, 4, 4), n_layers=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.lifting = nn.Conv3d(in_channels, hidden_channels, 1)

        self.spectral_layers = nn.ModuleList([
            SpectralConv3dForHSD(hidden_channels, hidden_channels, *n_modes)
            for _ in range(n_layers)
        ])

        self.local_layers = nn.ModuleList([
            nn.Conv3d(hidden_channels, hidden_channels, 1)
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.BatchNorm3d(hidden_channels)
            for _ in range(n_layers)
        ])

        self.projection = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, 1)
        )

    def forward(self, x):
        h = self.lifting(x)

        for i, (spectral, local, norm) in enumerate(zip(
            self.spectral_layers, self.local_layers, self.norms
        )):
            h1 = spectral(h)
            h2 = local(h)
            h = norm(h1 + h2)
            if i < self.n_layers - 1:
                h = F.gelu(h)

        out = self.projection(h)
        return out


class PhysicsEncoder(nn.Module):
    
    def __init__(self, Md0, Md1):
        super().__init__()
        self.register_buffer("Md0", torch.from_numpy(Md0.astype(np.float32)))
        self.register_buffer("Mdelta1", torch.from_numpy(Md0.T.astype(np.float32)))
        self.register_buffer("Md1", torch.from_numpy(Md1.astype(np.float32)))
        self.register_buffer("Mdelta2", torch.from_numpy(Md1.T.astype(np.float32)))

    def forward(self, c0, c1, c2):
        c0_induced_from_c1 = torch.matmul(c1, self.Mdelta1.t())
        c1_induced_from_c0 = torch.matmul(c0, self.Md0.t())
        c1_induced_from_c2 = torch.matmul(c2, self.Mdelta2.t())
        c2_induced_from_c1 = torch.matmul(c1, self.Md1.t())

        feat_c0 = torch.cat([c0, c0_induced_from_c1], dim=-1)
        feat_c1 = torch.cat([c1, c1_induced_from_c0, c1_induced_from_c2], dim=-1)
        feat_c2 = torch.cat([c2, c2_induced_from_c1], dim=-1)

        return feat_c0, feat_c1, feat_c2


class SpectralAmp(nn.Module):
    
    def __init__(self, n_modes, power=1.0):
        super().__init__()
        ramp = torch.arange(1, n_modes + 1).float().pow(power)
        self.gain = nn.Parameter(ramp)

    def forward(self, x):
        return x * self.gain.unsqueeze(0)


class PhysicsGMLPLayer(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.W1 = nn.Linear(in_features, out_features)
        self.W2 = nn.Linear(in_features, out_features)
        self.W_out = nn.Linear(out_features, out_features)

    def forward(self, x):
        x_norm = self.norm(x)
        content = self.W1(x_norm)
        gate = self.W2(x_norm)
        gate_act = F.silu(gate) 
        mixed = content * gate_act
        out = self.W_out(mixed)
        
        if x.shape[-1] == out.shape[-1]:
            return x + out
        return out


class SpectralVectorOperator(nn.Module):

    def __init__(self, Md0, Md1, k0, k1, k2, hidden_dims=(128, 64)):
        super().__init__()
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

        self.amp_c0 = SpectralAmp(k0, power=1.0)
        self.amp_c1 = SpectralAmp(k1, power=1.0)
        self.amp_c2 = SpectralAmp(k2, power=1.0)

        self.physics_encoder = PhysicsEncoder(Md0, Md1)

        self.register_buffer("Md0_op", torch.from_numpy(Md0.astype(np.float32)))
        self.register_buffer("Mdelta1_op", torch.from_numpy(Md0.T.astype(np.float32)))
        self.register_buffer("Md1_op", torch.from_numpy(Md1.astype(np.float32)))
        self.register_buffer("Mdelta2_op", torch.from_numpy(Md1.T.astype(np.float32)))

        in_dim = (2 * k0) + (3 * k1) + (2 * k2)
        
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(PhysicsGMLPLayer(last_dim, h))
            last_dim = h
        self.mlp_body = nn.Sequential(*layers)
        
        self.head_vec = nn.Linear(last_dim, k0)
        
        nn.init.orthogonal_(self.head_vec.weight, gain=0.1)

    def forward(self, c0, c1, c2):
        c0_boost = self.amp_c0(c0)
        c1_boost = self.amp_c1(c1)
        c2_boost = self.amp_c2(c2)
        
        feat_c0, feat_c1, feat_c2 = self.physics_encoder(c0_boost, c1_boost, c2_boost)
        feat = torch.cat([feat_c0, feat_c1, feat_c2], dim=-1)
        
        latent = self.mlp_body(feat)
        
        vec_coeffs = self.head_vec(latent).view(-1, 1, self.k0)
        
        return vec_coeffs


class CouplingErrorMLP(nn.Module):

    def __init__(self, k_total, hidden_dim=64):
        super().__init__()
        self.global_proj = nn.Linear(k_total, hidden_dim // 2)
        self.local_proj = nn.Linear(3, hidden_dim // 2)
        
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, pts, c_all_enhanced):
        B = c_all_enhanced.shape[0]
        N = pts.shape[0]
        
        pts_expanded = pts.unsqueeze(0).expand(B, -1, -1)
        
        local_feat = self.local_proj(pts_expanded)
        global_feat = self.global_proj(c_all_enhanced)
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)
        
        in_feat = torch.cat([local_feat, global_feat], dim=-1)
        return self.body(in_feat)


class HSDSpectralVectorFNO(nn.Module):

    def __init__(self, HSD_base_model, data_mgr, Phi0, 
                 fno_modes=(4, 4, 4), fno_hidden=16, fno_layers=2):
        super().__init__()
        
        self.HSD_base_branch = HSD_base_model
        
        self.k0 = HSD_base_model.k0
        self.k1 = HSD_base_model.k1
        self.k2 = HSD_base_model.k2
        
        self.register_buffer("Md0", HSD_base_model.Md0_op.clone())
        self.register_buffer("Mdelta1", HSD_base_model.Md0_op.t().clone())
        self.register_buffer("Md1", HSD_base_model.Mdelta2_op.t().clone())
        self.register_buffer("Mdelta2", HSD_base_model.Mdelta2_op.clone())
        
        self.fno_branch = FNO3dForHSD(
            in_channels=2,
            out_channels=1,
            hidden_channels=fno_hidden,
            n_modes=fno_modes,
            n_layers=fno_layers
        )
        
        k_enhanced_total = (2 * self.k0) + (3 * self.k1) + (2 * self.k2)
        self.mlp_interaction = CouplingErrorMLP(k_total=k_enhanced_total)
        
        self.data_mgr = data_mgr
        
        self.register_buffer('Phi0', Phi0)
        
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, c0, c1, c2, x_spatial):
        vec_coeffs_base = self.HSD_base_branch(c0, c1, c2)
        vec_base = torch.matmul(vec_coeffs_base, self.Phi0.t()).permute(0, 2, 1)
        
        fno_in, _ = self.data_mgr.prepare_fno_batch(x_spatial, None)
        grid_vec = self.fno_branch(fno_in)
        node_vec_fno = self.data_mgr.decode_fno_output(grid_vec)
        
        d0_c0 = torch.matmul(c0, self.Md0.t())
        delta1_c1 = torch.matmul(c1, self.Mdelta1.t())
        d1_c1 = torch.matmul(c1, self.Md1.t())
        delta2_c2 = torch.matmul(c2, self.Mdelta2.t())
        
        feat_c0 = torch.cat([c0, delta1_c1], dim=-1)
        feat_c1 = torch.cat([c1, d0_c0, delta2_c2], dim=-1)
        feat_c2 = torch.cat([c2, d1_c1], dim=-1)
        c_all_enhanced = torch.cat([feat_c0, feat_c1, feat_c2], dim=-1)
        
        node_vec_mlp = self.mlp_interaction(self.data_mgr.pts, c_all_enhanced)
        node_vec_res = node_vec_fno + node_vec_mlp
        
        B = c0.shape[0]
        N = vec_base.shape[1]
        vec_res_flat = node_vec_res.view(B, N, 1).permute(0, 2, 1)
        vec_proj_coeffs = torch.matmul(vec_res_flat, self.Phi0)
        vec_base_component = torch.matmul(vec_proj_coeffs, self.Phi0.t()).permute(0, 2, 1)
        vec_orthogonal = node_vec_res - vec_base_component

        # pred_vec = vec_base + self.res_scale * node_vec_res
        # return pred_vec, vec_base, node_vec_res
        
        pred_vec = vec_base + self.res_scale * vec_orthogonal
        return pred_vec, vec_base, vec_orthogonal




SpectralPhysicsAwareOperator = SpectralVectorOperator
HSDSpectralFNO = HSDSpectralVectorFNO