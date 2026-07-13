import torch as th

# Define a custom neural network model class
class Weight(th.nn.Module):
    def __init__(self,
                 r_dim: int=2,
                 seed: int=None,
                 initialize: str='default'
        ):
        super(Weight, self).__init__()

        th.manual_seed(seed)

        # Declare a trainable parameter
        if initialize == 'default':
            # self.weight = th.nn.Parameter(th.full((r_dim,1), 1/r_dim, dtype=th.float32))  # Example: a 1D tensor with 1/r_dim values

            # random sampling
            dirichlet_distribution = th.distributions.dirichlet.Dirichlet(th.ones(r_dim, dtype=th.float32)) # flat
            samples = dirichlet_distribution.sample() #[r_dim]
            self.weight = th.nn.Parameter(samples.unsqueeze(dim=-1))
            print("Weight", self.weight)
        else:
            raise NotImplementedError

        self.matrix = th.zeros(r_dim, r_dim)
        for i in range(r_dim):
            for j in range(i + 1):
                self.matrix[i, j] = 1.0 / (i + 1)
        self.intercept = self.matrix[:,0].unsqueeze(dim=-1)  # [r_dim, 1]

        self.zero_vector = th.zeros(r_dim)

    def step(self, lr, grad):
        # Update the weights using the gradient and learning rate
        with th.no_grad():
            naive_weight = self.weight.data.squeeze(dim=-1) - lr * grad # minus because we conduct gradient descent. [r_dim, ]

            ## Now we calculate projection onto unit simplex
            sorted_weight, _ = th.sort(naive_weight, descending=True) # [r_dim, ]
            sorted_weight = sorted_weight.unsqueeze(dim=-1) # [r_dim, 1]

            criterion = sorted_weight - th.matmul(self.matrix, sorted_weight) + self.intercept # [r_dim, 1]
            threshold_idx = th.sum(criterion > 0).item() # range 1 to r_dim
            lmbda = criterion[threshold_idx-1] - sorted_weight[threshold_idx-1] # [1,]

            ### Final result
            self.weight.data = th.max(naive_weight + lmbda, self.zero_vector).unsqueeze(dim=-1) # [r_dim, ] -> [r_dim, 1]

            ### Original method
            # proj_grad = grad - np.mean(grad)*np.ones_like(grad)
            # self.weight.data -= lr * proj_grad.reshape(-1,1)

    def forward(self, input):
        # Use the trainable parameter in the forward pass
        assert len(input.shape) == 2
        output = th.matmul(input, self.weight)
        return output