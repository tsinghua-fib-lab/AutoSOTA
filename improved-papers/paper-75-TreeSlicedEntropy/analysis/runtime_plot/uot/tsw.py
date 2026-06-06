import math
import torch
from power_spherical import PowerSpherical
from fast_pytorch_kmeans import KMeans as FastKMeans

class TWConcurrentLines():
    def __init__(self, 
                 ntrees=1000, 
                 nlines=5, 
                 p=2,
                 delta=2, 
                 mass_division='distance_based', 
                 ftype='linear',
                 d=3,
                 degree=3,
                 radius=2.0,
                 pow_beta=1,
                 device="cuda"):
        """
        Class for computing the Generalized Tree Wasserstein distance between two distributions.
        Args:
            ntrees (int): Number of trees.
            nlines (int): Number of lines per tree.
            p (int): Level of the norm.
            delta (float): Negative inverse of softmax temperature for distance-based mass division.
            mass_division (str): How to divide the mass, one of 'uniform', 'distance_based'.
            ftype (str): Type of defining function ('linear', 'poly', 'circular', 'augmented', 'pow').
            d (int): Dimension of the input space (used if ftype='poly' or ftype='augmented').
            degree (int): Degree of the polynomial (used if ftype='poly').
            radius (float): Radius of the circle (used if ftype='circular').
            pow_beta (float): Contribution between linear and pow (used if ftype='pow').
            uot_ulambda (float): lambda value in close-form for regularized EPT.
            uot_beta (float): beta value in close-form for regularized EPT.
            uot_alpha (float): alpha value in close-form for regularized EPT.
            uot_wr (float): wr value in close-form for regularized EPT. 
            device (str): Device to run the code, follows torch convention (default is "cuda").
        """
        self.ntrees = ntrees
        self.device = device
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.mass_division = mass_division
        
        self.ftype = ftype
        self.d = d
        self.degree = degree
        self.radius = radius
        self.pow_beta = pow_beta

        if self.ftype == 'pow':
            self.mapping = lambda X : X + self.pow_beta * X ** 3
            
            self.dtheta = d
        elif self.ftype == 'poly':
            self.powers = TWConcurrentLines.get_powers(d, degree).to(device)
            self.mapping = lambda X : TWConcurrentLines.poly_features(X, self.powers)

            self.dtheta = self.powers.shape[1]
        else:
            self.dtheta = d

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"
        assert self.ftype in ['linear', 'poly', 'circular', 'pow', 'circular_concentric'], \
            "Invalid ftype. Must be one of 'linear', 'poly', 'circular', 'pow', 'circular_concentric'"

    def __call__(self, X, Y, theta, intercept, total_mass_X=torch.tensor(1.0), total_mass_Y=torch.tensor(1.0)):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm

        return self.compute_tw(X, Y, theta, intercept,total_mass_X, total_mass_Y)

    def compute_tw(self, X, Y, theta, intercept, total_mass_X=torch.tensor(1.0), total_mass_Y=torch.tensor(1.0)):
        if self.ftype == 'poly' or self.ftype == 'pow':
            X = self.mapping(X)
            Y = self.mapping(Y)

        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        mass_X = mass_X * total_mass_X
        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)
        mass_Y = mass_Y * total_mass_Y
        tw = self.tw_concurrent_lines(mass_X, mass_Y, axis_coordinate_X, axis_coordinate_Y)[0]
        return tw

    def tw_concurrent_lines(self, mass_X, mass_Y, axis_coordinate_X, axis_coordinate_Y):
        """
        Args:
            mass_X: (num_trees, num_lines, num_points)
            mass_Y: (num_trees, num_lines, num_points)
            axis_coordinate_X: (num_trees, num_lines, num_points)
            axis_coordinate_Y: (num_trees, num_lines, num_points)
        """

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        mass_XY = torch.cat((mass_X, -mass_Y), dim=2)

        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)

        if self.ftype != 'circular_concentric' and self.ftype != 'circular':
            sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
            sub_mass_target_cumsum = torch.where(coord_sorted > 0, sub_mass_right_cumsum, sub_mass_target_cumsum)

        ### compute edge length
        if self.ftype != 'circular_concentric':

            # add root to the sorted coordinate by insert 0 to the first position <= 0
            root = torch.zeros(num_trees, num_lines, 1, device=self.device) 
            root_indices = torch.searchsorted(coord_sorted, root)
            coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device)
            # distribute other points to the correct position
            edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
            edge_mask.scatter_(2, root_indices, False)
            coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
            # compute edge length
            edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]
        else:
            prepend_tensor = torch.zeros((num_trees, 1, 1), device=coord_sorted.device)
            coord_sorted_with_prepend = torch.cat([prepend_tensor, coord_sorted], dim=-1)
            edge_length = coord_sorted_with_prepend[..., 1:] - coord_sorted_with_prepend[..., :-1]

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])

        # if unbalanced:
        #     uT = torch.sum(mass_X, dim=(-2, -1))
        #     vT = torch.sum(mass_Y, dim=(-2, -1))
            
        #     # higher beta means more regularization and less allowed unbalanced transport
        #     scalar = self.uot_beta * self.uot_ulambda / 2
        #     reg = (self.uot_wr + scalar - self.uot_alpha) * torch.abs(uT - vT) - scalar * (uT + vT)
        #     subtract_mass_sum += reg
        
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)
        return tw, sub_mass_target_cumsum, edge_length

    def project(self, input, theta, intercept):
        N = input.shape[0]
        num_trees = theta.shape[0]
        num_lines = theta.shape[1]

        # all lines has the same point which is root
        input_translated = (input - intercept) #[T,B,D]
        if self.ftype == 'circular':
            axis_coordinate = torch.norm(input_translated.unsqueeze(1) - theta.unsqueeze(2) * self.radius, dim=-1)
        elif self.ftype == 'circular_concentric':
            axis_coordinate = torch.norm(input_translated, dim=-1).unsqueeze(1) # [T,1,B]
        else:
            axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))
        
        if self.mass_division == 'uniform':
            mass_input = mass * torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division =='distance_based':
            if self.ftype == 'circular_concentric':
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate.repeat(1, num_lines, 1), theta)
            else: 
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
            dist = (torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim = -1))
            weight = -self.delta*dist
            mass_input = torch.softmax(weight, dim=-2)/N
        
            # if uot_topk != self.nlines and uot_topk > 0:
            #     # Get top-k indices and create a mask
            #     _, indices = torch.topk(mass_input, uot_topk, dim=-2)
            #     mask = torch.zeros_like(mass_input)
            #     mask.scatter_(dim=-2, index=indices, value=1)
            #     mass_input *= mask
        return mass_input, axis_coordinate

    @staticmethod
    def get_power_generator(dim, degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in TWConcurrentLines.get_power_generator(dim - 1,degree - value):
                    yield (value,) + permutation

    @staticmethod
    def get_powers(dim, degree):
        powers = TWConcurrentLines.get_power_generator(dim, degree)
        return torch.stack([torch.tensor(p) for p in powers], dim=1)         
    
    @staticmethod
    def homopoly(dim, degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return int(
            math.factorial(degree+dim-1) /
            (math.factorial(degree) * math.factorial(dim-1))
        )

    @staticmethod
    def poly_features(input, powers):
        return torch.pow(input.unsqueeze(-1), powers.unsqueeze(0)).prod(dim=1)

def calculate_geometric_median(X, tol=1e-7, max_iter=100, epsilon=1e-6):
    if not isinstance(X, torch.Tensor): raise TypeError("Input must be a torch.Tensor")
    device = X.device; dtype = X.dtype
    if X.dim() != 2: raise ValueError(f"Input tensor X must be 2D, got {X.shape}")
    N, dim = X.shape
    if N == 0: return torch.full((dim,), float('nan'), device=device, dtype=dtype)
    if N == 1: return X[0].clone()
    y = torch.mean(X.to(torch.float32), dim=0); X_compute = X.to(torch.float32)
    for iteration in range(max_iter):
        distances = torch.linalg.norm(X_compute - y.unsqueeze(0), dim=1)
        coincident_mask = distances < epsilon
        if torch.any(coincident_mask):
            coincident_idx = torch.where(coincident_mask)[0][0].item(); point_j = X_compute[coincident_idx]
            other_indices = torch.arange(N, device=device) != coincident_idx
            if not torch.any(other_indices): return point_j.to(dtype)
            other_points = X_compute[other_indices]; diff_vectors = point_j.unsqueeze(0) - other_points
            norms = torch.linalg.norm(diff_vectors, dim=1); valid_norms_mask = norms > epsilon
            if not torch.any(valid_norms_mask): return point_j.to(dtype)
            normalized_vectors = diff_vectors[valid_norms_mask] / norms[valid_norms_mask].unsqueeze(1)
            sum_normalized_vectors = torch.sum(normalized_vectors, dim=0); norm_sum_vectors = torch.linalg.norm(sum_normalized_vectors)
            if norm_sum_vectors <= 1.0 + epsilon: return point_j.to(dtype)
        weights = 1.0 / (distances + epsilon); sum_weights = torch.sum(weights)
        if sum_weights < epsilon: return y.to(dtype)
        weighted_sum = torch.sum(X_compute * weights.unsqueeze(1), dim=0); y_next = weighted_sum / sum_weights
        delta_change = torch.linalg.norm(y_next - y); y = y_next
        if delta_change < tol: return y.to(dtype)
    return y.to(dtype)

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U

def generate_trees_frames(ntrees, nlines, d, mean=0.0, std=1.0, device='cuda', intercept_mode='gaussian',gen_mode='gaussian_raw', X=None, Y=None, kappa=None):
    """
    Generates frames (theta vectors and intercepts) for TW computation.
    Uses fast-pytorch-kmeans for 'cluster_random_path' mode. If a specific cluster
    fails to contain both X and Y points, Gaussian noise is used for THAT cluster's theta.

    Args:
        ntrees (int): Number of trees.
        nlines (int): Number of lines per tree (target number of clusters).
        d (int): Dimension of the space (for theta/intercept in original space).
        mean (float): Mean for Gaussian initialization. Default 0.0.
        std (float): Standard deviation for Gaussian initialization. Default 1.0.
        device (str): Torch device ('cuda', 'cpu').
        gen_mode (str): Method for generating theta:
            'gaussian_raw', 'gaussian_orthogonal', 'random_path', 'cluster_random_path'.
        X (torch.Tensor, optional): Data points for distribution X (N, d). Required for data-driven modes.
        Y (torch.Tensor, optional): Data points for distribution Y (M, d). Required for data-driven modes.
        kappa (float, optional): Concentration parameter for PowerSpherical distribution.
                                   Applied only to vectors from valid clusters in cluster_random_path.
        intercept_mode (str): Method for generating intercept: 'gaussian', 'geometric_median'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (theta, intercept)
            theta: Shape (ntrees, nlines, d)
            intercept: Shape (1, 1, d)
    """
    # --- Input Validation ---
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal', 'random_path', 'cluster_random_path'], f"Invalid gen_mode: {gen_mode}"
    assert intercept_mode in ['gaussian', 'geometric_median'], f"Invalid intercept_mode: {intercept_mode}"

    if gen_mode in ['random_path', 'cluster_random_path'] or intercept_mode == 'geometric_median':
        if X is None or Y is None: raise ValueError(f"X and Y required for mode='{gen_mode}' or intercept='geometric_median'")
        if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
        if not isinstance(Y, torch.Tensor): Y = torch.tensor(Y, device=device)
        X = X.to(device=device, dtype=torch.float32)
        Y = Y.to(device=device, dtype=torch.float32)
        N, d_x = X.shape; M, d_y = Y.shape
        if d_x != d or d_y != d: raise ValueError(f"Dimension mismatch: d={d}, X={X.shape}, Y={Y.shape}")


    # --- Generate Intercept (Always in original space 'd') ---
    intercept = None
    if intercept_mode == 'gaussian':
        intercept = torch.randn(ntrees, 1, d, device=device, dtype=torch.float32) * std + mean
    elif intercept_mode == 'geometric_median':
        XY = torch.cat((X, Y), dim=0)
        with torch.no_grad(): median = calculate_geometric_median(XY.detach())
        if torch.isnan(median).any():
             intercept = torch.randn(ntrees, 1, d, device=device, dtype=torch.float32) * std + mean
        else: intercept = torch.randn(ntrees, 1, d, device=device, dtype=torch.float32) * std + median.reshape(1, 1, d).float()


    # --- Generate Theta (Always in original space 'd') ---
    theta = None
    if gen_mode == 'gaussian_raw':
        theta_init = torch.randn(ntrees, nlines, d, device=device, dtype=torch.float32)
        theta = theta_init / torch.linalg.norm(theta_init, dim=-1, keepdim=True).clamp(min=1e-12)

    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "nlines must be <= d for gaussian_orthogonal"
        theta_init = torch.randn(ntrees, d, nlines, device=device, dtype=torch.float32)
        # Ensure svd_orthogonalize is defined elsewhere
        theta_ortho = svd_orthogonalize(theta_init); theta = theta_ortho.transpose(-2, -1)

    elif gen_mode == 'random_path':
        L = ntrees * nlines
        with torch.no_grad():
            idx_X = torch.randint(0, N, (L,), device=device); idx_Y = torch.randint(0, M, (L,), device=device)
            theta_init = X.detach()[idx_X] - Y.detach()[idx_Y]
        theta = theta_init
        theta_norm = torch.linalg.norm(theta, dim=-1, keepdim=True); valid_mask = (theta_norm > 1e-9).squeeze(-1)
        num_invalid = L - valid_mask.sum()
        if num_invalid > 0:
            print(f"Warning: Found {num_invalid} zero vectors in 'random_path'. Replacing.")
            replacements = torch.randn(num_invalid, d, device=device, dtype=theta.dtype)
            replacements = replacements / torch.linalg.norm(replacements, dim=-1, keepdim=True).clamp(min=1e-12)
            theta[~valid_mask] = replacements
        theta = theta / torch.linalg.norm(theta, dim=-1, keepdim=True).clamp(min=1e-12)
        if kappa is not None and kappa > 0:
             # Ensure PowerSpherical is imported
             ps = PowerSpherical(loc=theta, scale=torch.full((L,), kappa, device=device, dtype=theta.dtype))
             theta = ps.rsample()
        theta = theta.view(ntrees, nlines, d)

    elif gen_mode == 'cluster_random_path':
        if N < 1 or M < 1: # Need at least one point from each
             raise ValueError("Cannot perform clustering with empty X or Y.")

        all_thetas = torch.zeros(ntrees, nlines, d, device=device, dtype=torch.float32)

        with torch.no_grad():
            X_detached = X.detach()
            Y_detached = Y.detach()
            XY = torch.cat((X_detached, Y_detached), dim=0).float()

            # Perform one clustering attempt
            kmeans = FastKMeans(n_clusters=nlines, max_iter=nlines*10, verbose=0, mode='euclidean')
            labels = kmeans.fit_predict(XY)
            labels_X = labels[:N]; labels_Y = labels[N:]
            counts_X = torch.bincount(labels_X, minlength=nlines)
            counts_Y = torch.bincount(labels_Y, minlength=nlines)

            # Pre-calculate indices for sampling later
            indices_X_per_cluster = [torch.where(labels_X == i)[0] for i in range(nlines)]
            indices_Y_per_cluster = [torch.where(labels_Y == i)[0] for i in range(nlines)]

            # Loop through each target line/cluster
            for c in range(nlines):
                is_valid_cluster = counts_X[c] > 0 and counts_Y[c] > 0

                if is_valid_cluster:
                    # --- Generate theta from sampled points in this valid cluster ---
                    indices_c_X = indices_X_per_cluster[c]
                    indices_c_Y = indices_Y_per_cluster[c]
                    num_x_in_c, num_y_in_c = len(indices_c_X), len(indices_c_Y)

                    # Sample ntrees indices with replacement
                    sampled_idx_X = indices_c_X[torch.randint(num_x_in_c, (ntrees,), device=device)]
                    sampled_idx_Y = indices_c_Y[torch.randint(num_y_in_c, (ntrees,), device=device)]

                    # Compute difference vectors for this cluster
                    theta_c = X_detached[sampled_idx_X] - Y_detached[sampled_idx_Y] # Shape (ntrees, d)

                    # Normalize, handle zero vectors within this slice
                    theta_c_norm = torch.linalg.norm(theta_c, dim=-1, keepdim=True)
                    valid_mask_c = (theta_c_norm > 1e-9).squeeze(-1) # Shape (ntrees,)
                    num_invalid_c = ntrees - valid_mask_c.sum()

                    if num_invalid_c > 0 :
                         # print(f"    Warning: Found {num_invalid_c} zero vectors in cluster {c}. Replacing.") # Can be verbose
                         replacements = torch.randn(num_invalid_c, d, device=device, dtype=theta_c.dtype)
                         replacements = replacements / torch.linalg.norm(replacements, dim=-1, keepdim=True).clamp(min=1e-12)
                         theta_c[~valid_mask_c] = replacements # Replace only within this cluster's vectors

                    # Normalize all vectors for this cluster
                    theta_c = theta_c / torch.linalg.norm(theta_c, dim=-1, keepdim=True).clamp(min=1e-12)

                    # Optionally apply PowerSpherical ONLY to valid cluster vectors
                    if kappa is not None and kappa > 0:
                         ps = PowerSpherical(loc=theta_c, scale=torch.full((ntrees,), kappa, device=device, dtype=theta_c.dtype))
                         theta_c = ps.rsample()

                    # Assign results to the output tensor
                    all_thetas[:, c, :] = theta_c

                else:
                    # --- Generate theta using Gaussian noise for this invalid cluster ---
                    theta_c_gauss = torch.randn(ntrees, d, device=device, dtype=torch.float32)
                    # Normalize Gaussian vectors
                    theta_c_gauss = theta_c_gauss / torch.linalg.norm(theta_c_gauss, dim=-1, keepdim=True).clamp(min=1e-12)
                    # Assign results to the output tensor
                    all_thetas[:, c, :] = theta_c_gauss

            # Combine results from all clusters
            theta = all_thetas

    else: # Should not happen
        raise ValueError(f"Unknown gen_mode: {gen_mode}")

    # --- Final Checks ---
    if theta is None: raise RuntimeError(f"Theta generation failed for mode {gen_mode}")
    if theta.shape != (ntrees, nlines, d): raise RuntimeError(f"Theta shape: {theta.shape} != {(ntrees, nlines, d)}")
    if intercept is None: raise RuntimeError("Intercept generation failed.")

    return theta.float(), intercept.float()

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    torch.set_float32_matmul_precision('high')
    # N = 32 * 32
    # M = 32 * 32
    # dn = dm = 128
    # ntrees = 2048
    # nlines = 2
    
    N = 5
    M = 5
    dn = dm = 3
    ntrees = 7
    nlines = 2
    
    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='circular', radius=2))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='circular_concentric'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    dtheta = TWConcurrentLines.homopoly(dn, 3)
    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='poly', d=dn, degree=3))
    theta, intercept = generate_trees_frames(ntrees, nlines, dtheta, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    # theta, intercept = generate_trees_frames(ntrees, nlines, dn, dn, gen_mode="gaussian_orthogonal")
    # X = torch.rand(N, dn).to("cuda")
    # Y = torch.rand(M, dm).to("cuda")
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     tw = TW_obj(X, Y, theta, intercept)
    #     TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn)
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based'))
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, intercept_mode="geometric_median", gen_mode="cluster_random_path", X=X, Y=Y)
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)

    # prof.export_chrome_trace("trace_concurrent.json")
    # with open("profile_result_concurrent.txt", "w") as f:
    #     table_str = prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True)
    #     f.write(table_str)
    #     print(table_str)
