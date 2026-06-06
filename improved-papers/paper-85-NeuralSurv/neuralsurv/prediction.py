import pandas as pd

import jax
import jax.random as jr
import jax.numpy as jnp

from postprocessing.postprocessing_function import (
    PostProcessingPosteriorFunction,
    PostProcessingMAPFunction,
)
from postprocessing.evaluation_metrics import EvaluationMetrics
from postprocessing.draw_posterior_samples import (
    get_posterior_samples_phi,
    get_posterior_samples_glin_fn,
    get_prior_samples_glin_fn,
)


class NeuralSurvPredict:

    def _check_posterior_params_exist(self):
        if not hasattr(self, "posterior_params"):
            raise ValueError("Run neuralsurv.fit() first")

    def get_posterior_samples(self, rng, num_samples):

        # Check model has been fitted
        self._check_posterior_params_exist()

        rng, subrng_phi, subrng_glin_fn = jr.split(rng, 3)
        if not hasattr(self, "phi_samples"):
            self.phi_samples = get_posterior_samples_phi(
                subrng_phi, num_samples, self.posterior_params
            )
        if not hasattr(self, "glin_fn_samples"):
            self.glin_fn_samples = get_posterior_samples_glin_fn(
                subrng_glin_fn,
                num_samples,
                self.batch_size,
                self.g,
                self.theta_MAP,
                self.posterior_params,
            )

    def get_prior_samples(self, rng, num_samples):

        if not hasattr(self, "glin_fn_samples_prior"):
            self.glin_fn_samples_prior = get_prior_samples_glin_fn(
                rng,
                num_samples,
                self.batch_size,
                self.g,
                self.theta_MAP,
                self.posterior_params,
            )

    def predict_hazard_function(self, times, x, aggregate=None, type="posterior"):
        """Obtain posterior samples of hazard function
        aggregate = (0,2): Return summarized posterior samples across patients and posterior samples
        aggregate = 2: Return summarized posterior samples across posterior samples
        aggregate = None: Return posterior samples
        """

        # Rescale times
        times_rescaled = times / self.max_time_train

        # Posterior samples hazard function evaluated at times
        if type == "posterior":
            func = PostProcessingPosteriorFunction(
                rho=self.rho,
                Z=self.Z,
                phi_samples=self.phi_samples,
                glin_fn_samples=self.glin_fn_samples,
                batch_size=self.batch_size,
            )
        elif type == "MAP":
            func = PostProcessingMAPFunction(
                rho=self.rho,
                Z=self.Z,
                phi_MAP=self.phi_MAP,
                glin_fn_MAP=lambda t, x: self.g(t, x, self.theta_MAP),
                batch_size=self.batch_size,
            )

        samples = func.compute_hazard_function(
            times=times_rescaled,
            x=x,
        )

        # Rescale hazard
        samples /= self.max_time_train

        # Summarise posterior
        if type == "posterior" and aggregate == (0, 2):
            # First aggregate across x
            samples = jnp.nanmean(samples, axis=0)

            # Then aggregate accross posterior samples
            median = jnp.nanmedian(samples, axis=1)
            q025 = jnp.nanquantile(samples, 0.025, axis=1)
            q975 = jnp.nanquantile(samples, 0.975, axis=1)
            return {
                "times": times,
                "median": median,
                "q025": q025,
                "q975": q975,
            }
        else:
            return samples

    def predict_survival_function(self, times, x, aggregate=None, type="posterior"):
        """Obtain posterior samples of survival function
        aggregate = (0,2): Return summarized posterior samples across observation index and sample index
        aggregate = 2: Return summarized posterior samples across sample index
        aggregate = None: Return posterior samples"""

        # Rescale times
        times_rescaled = times / self.max_time_train

        # Posterior samples survival function evaluated at times
        if type == "posterior":
            func = PostProcessingPosteriorFunction(
                rho=self.rho,
                Z=self.Z,
                phi_samples=self.phi_samples,
                glin_fn_samples=self.glin_fn_samples,
                num_points=self.num_points_integral_cavi,
                batch_size=self.batch_size,
            )
        elif type == "MAP":
            func = PostProcessingMAPFunction(
                rho=self.rho,
                Z=self.Z,
                phi_MAP=self.phi_MAP,
                glin_fn_MAP=lambda t, x: self.g(t, x, self.theta_MAP),
                num_points=self.num_points_integral_cavi,
                batch_size=self.batch_size,
            )

        samples = func.compute_survival_function(
            times=times_rescaled,
            x=x,
        )

        # Summarise posterior
        if type == "posterior" and aggregate == (0, 2):
            # First aggregate across x
            samples = jnp.nanmean(samples, axis=0)

            # Then aggregate accross posterior samples
            median = jnp.nanmedian(samples, axis=1)
            q025 = jnp.nanquantile(samples, 0.025, axis=1)
            q975 = jnp.nanquantile(samples, 0.975, axis=1)
            return {
                "times": times,
                "median": median,
                "q025": q025,
                "q975": q975,
            }
        else:
            return samples

    def predict_glin_function(self, times, x, aggregate=None, dist="posterior"):
        """Obtain posterior samples of survival function
        aggregate = (0,2): Return summarized posterior samples across observation index and sample index
        aggregate = 2: Return summarized posterior samples across sample index
        aggregate = None: Return posterior samples"""

        # Posterior sample of phi and g
        if dist == "posterior":

            @jax.jit
            def glin_fn(time, x):
                return self.glin_fn_samples(time, x).squeeze()

        elif dist == "prior":

            @jax.jit
            def glin_fn(time, x):
                return self.glin_fn_samples_prior(time, x).squeeze()

        # Rescale times
        times_rescaled = times / self.max_time_train

        # Posterior samples glin function evaluated at times
        func = PostProcessingPosteriorFunction(
            rho=self.rho,
            Z=self.Z,
            phi_samples=self.phi_samples,
            glin_fn_samples=glin_fn,
            num_points=self.num_points_integral_cavi,
            batch_size=self.batch_size,
        )

        samples = func.compute_glin_function(
            times=times_rescaled,
            x=x,
        )

        # Summarise posterior
        if aggregate is not None:
            # First aggregate across x
            samples = jnp.nanmean(samples, axis=0)

            # Then aggregate accross posterior samples
            median = jnp.nanmedian(samples, axis=1)
            q025 = jnp.nanquantile(samples, 0.025, axis=1)
            q975 = jnp.nanquantile(samples, 0.975, axis=1)
            return {
                "times": times_rescaled * self.max_time_train,
                "median": median,
                "q025": q025,
                "q975": q975,
            }
        else:
            return samples

    def compute_evaluation_metrics(
        self,
        time_train,
        event_train,
        time,
        event,
        x,
        plot_dir=None,
        full_postprocessing=False,
    ):
        print("Compute C-index and Brier score")

        # Get risk score posterior samples
        risk = self.predict_hazard_function(time, x)

        # Get survival posterior samples
        survival = self.predict_survival_function(time, x)

        # Get evaluation metrics
        if full_postprocessing:
            evaluation_metrics = EvaluationMetrics(
                risk,
                survival,
                event_train,
                time_train,
                event,
                time,
            )
        else:
            risk = jnp.nanmedian(risk, axis=2)
            survival = jnp.nanmedian(survival, axis=2)
            evaluation_metrics = EvaluationMetrics(
                risk,
                survival,
                event_train,
                time_train,
                event,
                time,
                method="frequentist",
            )

        # Print
        df = pd.DataFrame.from_dict(evaluation_metrics.metrics, orient="index")
        print(df)

        # Save
        if plot_dir is not None:
            df.reset_index().to_csv(plot_dir + "/evaluation_metrics.csv", index=False)
