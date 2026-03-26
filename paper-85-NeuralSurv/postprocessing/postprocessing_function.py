import jax
import jax.numpy as jnp


class PostProcessingPosteriorFunction:
    def __init__(
        self, rho, Z, phi_samples, glin_fn_samples, batch_size, num_points=None
    ):
        self.rho, self.Z = rho, Z
        self.phi_samples, self.glin_fn_samples = phi_samples, glin_fn_samples
        self.num_points = num_points
        self.batch_size = batch_size
        self.hazard_fn = self.get_hazard_fn()
        self.survival_fn = self.get_survival_fn()

    def compute_glin_function(self, times, x):

        # Compute hazard function
        # 1st dimension: posterior samples
        # 2nd dimension: x
        # 3rd dimension: times
        time_eval = jnp.tile(times, x.shape[0])
        x_eval = jnp.repeat(x, len(times), axis=0)
        samples = self.glin_fn_samples(time_eval, x_eval).squeeze()
        samples = samples.reshape(samples.shape[0], len(x), len(times))

        # Transpose to
        # 1st dimension: x
        # 2nd dimension: times
        # 3rd dimension: posterior samples
        return jnp.transpose(samples, axes=(1, 2, 0))

    def compute_hazard_function(self, times, x):

        N = x.shape[0]
        m = len(times)

        # Compute hazard function
        # 1st dimension: posterior samples
        # 2nd dimension: x
        # 3rd dimension: times
        time_eval = jnp.tile(times, N)
        x_eval = jnp.repeat(x, m, axis=0)
        samples = self.hazard_fn(time_eval, x_eval)
        samples = samples.reshape(samples.shape[0], N, m)

        # Transpose to
        # 1st dimension: x
        # 2nd dimension: times
        # 3rd dimension: posterior samples
        return jnp.transpose(samples, axes=(1, 2, 0))

    def compute_survival_function(self, times, x):

        N = x.shape[0]
        m = len(times)

        # grid over which to evaluate the integrals
        time_grid = jnp.linspace(
            1e-16, jnp.max(times), self.num_points, dtype=jnp.float32
        )
        n = len(time_grid)

        # (t, x) pairs
        time_eval = jnp.tile(time_grid, N)
        x_eval = jnp.repeat(x, n, axis=0)

        # When time_grid <= times
        trap_mask = jnp.tile(time_grid, m) <= jnp.repeat(times, n, axis=0)
        trap_mask = trap_mask.reshape(m, n)
        index_last_true = jnp.sum(trap_mask, axis=1) - 1

        # Generic trapezoid weights
        dt = jnp.diff(time_grid)
        w = jnp.zeros_like(time_grid)
        w = w.at[1:-1].set(0.5 * (dt[1:] + dt[:-1]))
        w = w.at[0].set(0.5 * dt[0])
        w = jnp.tile(w, m).reshape(m, n)
        w = w.at[jnp.arange(m), index_last_true].set(0.5 * dt[-1])
        trap_w = w

        # Compute survival function
        # 1st dimension: posterior samples
        # 2nd dimension: x
        # 3rd dimension: times
        self.N = N
        self.n = n
        samples = self.survival_fn(time_eval, x_eval, trap_mask, trap_w)

        # Transpose to
        # 1st dimension: x
        # 2nd dimension: times
        # 3rd dimension: posterior samples
        return jnp.transpose(samples, axes=(1, 2, 0))

    def get_hazard_fn(self):

        @jax.jit
        def hazard_fn(time_eval, x_eval):
            glin = self.glin_fn_samples(time_eval, x_eval).squeeze()
            phi_repeated = jnp.repeat(self.phi_samples[:, None], len(time_eval), axis=1)
            time_repeated = jnp.repeat(time_eval[None, :], glin.shape[0], axis=0)
            Z_repeated = jnp.repeat(
                self.Z(time_eval, x_eval)[None, :], glin.shape[0], axis=0
            )

            return (
                phi_repeated
                * (time_repeated ** (self.rho - 1))
                / Z_repeated
                * jax.nn.sigmoid(glin)
            )

        return hazard_fn

    def get_survival_fn(self):

        @jax.jit
        def survival_fn(time_eval, x_eval, trap_mask, trap_w):

            samples = self.hazard_fn(time_eval, x_eval)

            # reshape
            samples_reshaped = samples.reshape(samples.shape[0], self.N, self.n)

            # compute integrals
            integral = jnp.einsum("ijt,kt,kt->ijk", samples_reshaped, trap_mask, trap_w)

            return jnp.exp(-integral)

        return survival_fn


class PostProcessingMAPFunction:
    def __init__(self, rho, Z, phi_MAP, glin_fn_MAP, batch_size, num_points=None):
        self.rho, self.Z = rho, Z
        self.phi_MAP, self.glin_fn_MAP = phi_MAP, glin_fn_MAP
        self.num_points = num_points
        self.batch_size = batch_size
        self.hazard_fn = self.get_hazard_fn()
        self.survival_fn = self.get_survival_fn()

    def compute_hazard_function(self, times, x):

        N = x.shape[0]
        m = len(times)

        # Compute hazard function
        # 1st dimension: x
        # 2nd dimension: times
        time_eval = jnp.tile(times, N)
        x_eval = jnp.repeat(x, m, axis=0)
        samples = self.hazard_fn(time_eval, x_eval)
        return samples.reshape(N, m)

    def compute_survival_function(self, times, x):

        N = x.shape[0]
        m = len(times)

        # grid over which to evaluate the integrals
        time_grid = jnp.linspace(
            1e-16, jnp.max(times), self.num_points, dtype=jnp.float32
        )
        n = len(time_grid)

        # (t, x) pairs
        time_eval = jnp.tile(time_grid, N)
        x_eval = jnp.repeat(x, n, axis=0)

        # When time_grid <= times
        trap_mask = jnp.tile(time_grid, m) <= jnp.repeat(times, n, axis=0)
        trap_mask = trap_mask.reshape(m, n)
        index_last_true = jnp.sum(trap_mask, axis=1) - 1

        # Generic trapezoid weights
        dt = jnp.diff(time_grid)
        w = jnp.zeros_like(time_grid)
        w = w.at[1:-1].set(0.5 * (dt[1:] + dt[:-1]))
        w = w.at[0].set(0.5 * dt[0])
        w = jnp.tile(w, m).reshape(m, n)
        w = w.at[jnp.arange(m), index_last_true].set(0.5 * dt[-1])
        trap_w = w

        # Compute survival function
        # 1st dimension: x
        # 2nd dimension: times
        self.N = N
        self.n = n
        return self.survival_fn(time_eval, x_eval, trap_mask, trap_w)

    def get_hazard_fn(self):

        @jax.jit
        def hazard_fn(time_eval, x_eval):
            glin = self.glin_fn_MAP(time_eval, x_eval).squeeze()
            phi_repeated = jnp.repeat(self.phi_MAP, len(time_eval), axis=0)
            Z = self.Z(time_eval, x_eval)

            return (
                phi_repeated * (time_eval ** (self.rho - 1)) / Z * jax.nn.sigmoid(glin)
            )

        return hazard_fn

    def get_survival_fn(self):

        @jax.jit
        def survival_fn(time_eval, x_eval, trap_mask, trap_w):

            samples = self.hazard_fn(time_eval, x_eval)

            # reshape
            samples_reshaped = samples.reshape(self.N, self.n)

            # compute integrals
            integral = jnp.einsum("jt,kt,kt->jk", samples_reshaped, trap_mask, trap_w)

            return jnp.exp(-integral)

        return survival_fn
