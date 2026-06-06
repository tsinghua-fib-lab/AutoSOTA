import math
import torch

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
                 mapping=None,
                 mapping_optimizer=None,
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
        elif self.ftype == 'augmented':
            self.lr = 1e-3
            self.num_iter = 1
            self.alpha = 1e-3

            self.linear_mapping = torch.nn.Linear(self.d, 64).to(device)
            self.mapping = lambda X : torch.cat((self.linear_mapping(X), X), dim=-1)
            self.mapping_optimizer = torch.optim.Adam(self.linear_mapping.parameters(), lr=self.lr)

            self.dtheta = d + 64
        else:
            self.dtheta = d

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"
        assert self.ftype in ['linear', 'poly', 'circular', 'augmented', 'pow', 'circular_concentric'], \
            "Invalid ftype. Must be one of 'linear', 'poly', 'circular', 'augmented', 'pow', 'circular_concentric'"

    def __call__(self, X, Y, theta, intercept, optimize_mapping=False):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N

        if optimize_mapping:
            self.optimize_mapping(X, Y, theta, intercept)
        else:
            return self.compute_tw(X, Y, theta, intercept)

    def compute_tw(self, X, Y, theta, intercept):
        if self.ftype == 'augmented' or self.ftype == 'poly' or self.ftype == 'pow':
            X = self.mapping(X)
            Y = self.mapping(Y)

        combined_axis_coordinate, mass_XY = self.get_mass_and_coordinate(X, Y, theta, intercept)
        tw = self.tw_concurrent_lines(mass_XY, combined_axis_coordinate)[0]
        return tw

    def optimize_mapping(self, X_original, Y_original, theta, intercept):
        X_detach = X_original.detach()
        Y_detach = Y_original.detach()
        for _ in range(self.num_iter):
            # generate input mapping
            X_mapped = self.mapping(X_detach)
            Y_mapped = self.mapping(Y_detach)

            # compute tsw
            combined_axis_coordinate, mass_XY = self.get_mass_and_coordinate(X_mapped, Y_mapped, theta, intercept)
            negative_tw = -self.tw_concurrent_lines(mass_XY, combined_axis_coordinate)[0]
            # optimize mapping
            reg = self.alpha * (torch.norm(X_mapped, dim=-1).mean() + torch.norm(Y_mapped, dim=-1).mean())
            self.mapping_optimizer.zero_grad()
            (reg + negative_tw).backward()
            self.mapping_optimizer.step()

    def tw_concurrent_lines(self, mass_XY, combined_axis_coordinate):
        """
        Args:
            mass_XY: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, num_lines, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
        
        if self.ftype != 'circular_concentric' and self.ftype != 'circular':
            mask_right = torch.nonzero(coord_sorted > 0, as_tuple=True)
            sub_mass_target_cumsum[mask_right] = sub_mass_right_cumsum[mask_right]

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
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw, sub_mass_target_cumsum, edge_length


    def get_mass_and_coordinate(self, X, Y, theta, intercept):
        # for the last dimension
        # 0, 1, 2, ...., N -1 is of distribution 1
        # N, N + 1, ...., 2N -1 is of distribution 2
        N, dn = X.shape
        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        massXY = torch.cat((mass_X, -mass_Y), dim=2)

        return combined_axis_coordinate, massXY

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
        # print(axis_coordinate.mean(), axis_coordinate.std(), axis_coordinate.max(), axis_coordinate.min())
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division =='distance_based':
            if self.ftype == 'circular_concentric':
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate.repeat(1, num_lines, 1), theta)
            else: 
                input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
            dist = (torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim = -1))
            weight = -self.delta*dist
            mass_input = torch.softmax(weight, dim=-2)/N
        
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

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U

def generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
    # random root as gaussian distribution with given mean and std
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    root = torch.randn(ntrees, 1, d, device=device) * std + mean
    intercept = root
    
    if gen_mode == 'gaussian_raw':
        theta = torch.randn(ntrees, nlines, d, device=device)
        theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
        theta = torch.randn(ntrees, d, nlines, device=device)
        theta = svd_orthogonalize(theta)
        theta = theta.transpose(-2, -1)
    
    return theta, intercept

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
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

    dtheta = dn + 64
    TW_obj = torch.compile(TWConcurrentLines(ntrees=ntrees, mass_division='distance_based', ftype='augmented', d=dn))
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

    # prof.export_chrome_trace("trace_concurrent.json")
    # with open("profile_result_concurrent.txt", "w") as f:
    #     table_str = prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True)
    #     f.write(table_str)
    #     print(table_str)
