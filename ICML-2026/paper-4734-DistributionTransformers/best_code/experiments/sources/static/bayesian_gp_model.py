class BayesianGPModel:
            def __init__(self, x_query_shape):
                self.mean = gpytorch.means.ZeroMean()
                self.kernel = gpytorch.kernels.RBFKernel()
                self.lengthscale_prior = dist.InverseGamma(1.0, 2.0)  # Prior for lengthscale
                self.y_pred_prior = dist.MultivariateNormal(
                    hyperparams['loc'].expand(x_query_shape[0]).reshape(x_query_shape[0], 1), 
                    (hyperparams['covariance_matrix']**0.5).expand(x_query_shape[0]).reshape(x_query_shape[0], 1, 1)
                )

            def model(self, Dx, Dy, x_query):
                """Defines the generative model for Bayesian GP."""
                # Sample lengthscale from prior
                lengthscale = pyro.sample("lengthscale", self.lengthscale_prior)
                y_pred = pyro.sample("y_pred", self.y_pred_prior)
                self.kernel.lengthscale = lengthscale.item()

                # Define the GP model
                gpr = BatchGPModel(x_query.view(-1,1, 1), y_pred.view(-1,1), gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([x_query.shape[0]])), self.mean, self.kernel)
                gpr.likelihood.noise = torch.tensor(0.1)
                gpr.eval()
                # Register GP model
                pyro.module("gpr", gpr)

                # Sample function values at x_query
                with pyro.plate("data", Dx.shape[0]):
                    pyro.sample("Dy", gpr.likelihood(gpr(Dx)), obs=Dy)

            def run_mcmc(self, Dx, Dy, x_query, num_samples=500, warmup_steps=200):
                """Runs MCMC to infer the posterior over the length scale and function values."""
                nuts_kernel = NUTS(self.model)
                self.mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
                self.mcmc.run(Dx.expand(x_query.shape[0], Dx.shape[0]).unsqueeze(-1), Dy.expand(x_query.shape[0], Dy.shape[0]), x_query)

            def sample_lengthscale_and_y(self, x_query, num_samples=100):
                """Samples lengthscale and function values y_query at x_query."""
                posterior_samples = self.mcmc.get_samples()
                lengthscale_samples = posterior_samples["lengthscale"]
                y_query_samples = posterior_samples["y_query"]

                return lengthscale_samples[:num_samples], y_query_samples[:num_samples]