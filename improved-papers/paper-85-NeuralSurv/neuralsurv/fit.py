import jax.numpy as jnp
import jax.random as jr

from model.model_utils import (
    from_params_to_theta,
)
from inference.cavi import CAVI
from map.em import EMAlgorithm


class NeuralSurvFit:

    def fit(
        self,
        time,
        event,
        x,
    ):

        # Rescale time
        self.max_time_train = time.max()
        time_rescaled = time / self.max_time_train

        # Get g
        self._get_g()

        # Find Z
        self._find_Z()

        # Find theta MAP
        if hasattr(self, "theta_MAP") and not self.overwrite_em:
            print("theta_MAP has already been found. Using saved value...")
        else:
            print("1. Find theta_MAP")
            self.run_EM(time_rescaled, event, x)
            self.save()

        # Run CAVI
        if hasattr(self, "posterior_params") and not self.overwrite_cavi:
            print("\nCAVI has already been ran. Using saved value...")
        else:
            print("\n2. Run CAVI")
            self.run_CAVI(time_rescaled, event, x)
            self.save()

    def run_EM(self, time, event, x, find_theta_MAP=True):

        theta_init = jnp.array(
            from_params_to_theta(self.model_params_init), dtype=jnp.float32
        )
        phi_init = jnp.array([self.alpha_prior / self.beta_prior], dtype=jnp.float32)
        alpha_prior = jnp.array(self.alpha_prior, dtype=jnp.float32)
        beta_prior = jnp.array(self.beta_prior, dtype=jnp.float32)

        if find_theta_MAP:

            # Get theta MAP in vector and parameters form
            em = EMAlgorithm(
                time=time,
                event=event,
                x=x,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
                rho=self.rho,
                Z=self.Z,
                g=self.g,
                theta_init=theta_init,
                phi_init=phi_init,
                num_points=self.num_points_integral_em,
                batch_size=self.batch_size,
                max_iter=self.max_iter_em,
            )
            self.theta_MAP, self.phi_MAP = em.fit()

            # Keep track of the q-function values
            self.q_function_history = jnp.stack(em.q_function_history)

        else:
            self.theta_MAP, self.phi_MAP = theta_init, phi_init

    def run_CAVI(self, time, event, x):

        # Instantiate the cavi algorithm
        cavi = CAVI(
            time=time,
            event=event,
            x=x,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
            rho=self.rho,
            phi_MAP=self.phi_MAP.squeeze(),
            theta_MAP=self.theta_MAP,
            g=self.g,
            Z=self.Z,
            num_points=self.num_points_integral_cavi,
            batch_size=self.batch_size,
            max_iter=self.max_iter_cavi,
            workdir=self.output_dir,
        )

        # Run cavi
        cavi.fit()

        # Keep track of posterior parameters
        selected_keys = [
            "mu",
            "w_B",
            "alpha",
            "beta",
            "time_all",
            "x_all",
        ]  # Add the attributes you want
        attributes = vars(cavi)
        self.posterior_params = {
            key: attributes[key] for key in selected_keys if key in attributes
        }
