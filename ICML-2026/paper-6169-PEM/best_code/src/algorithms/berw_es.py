import math
import numpy as np


class BerwES2:
    """
    CMA-style diagonal ES with bootstrap-estimated rank weights.

    This is a derivative-free, black-box optimizer designed for COCO/BBOB minimization.
    It is used as the core engine behind the BERW variants in this repository.
    """

    def __init__(
        self,
        problem,
        max_evals,
        *,
        seed=0,
        popsize=None,
        mu=None,
        n_start=1.5,
        n_end=6.0,
        c_mean=1.0,
        mean_update="average",
        restart_patience=60,
        min_sigma_rel=1e-14,
        max_sigma_rel=0.5,
    ):
        self.problem = problem
        self.max_evals = int(max_evals)
        self.rng = np.random.RandomState(int(seed))

        self.dim = int(problem.dimension)
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.upper = np.asarray(problem.upper_bounds, dtype=float)
        self.span = self.upper - self.lower
        self.span_min = float(np.min(self.span))

        if popsize is None:
            popsize = max(4, 4 + int(3 * math.log(self.dim)))
        self.lambda_ = int(popsize)
        if mu is None:
            mu = self.lambda_ // 2
        self.mu = int(max(1, min(int(mu), self.lambda_)))

        self.n_start = float(n_start)
        self.n_end = float(n_end)
        self.c_mean = float(c_mean)

        if mean_update not in {"average", "weiszfeld"}:
            raise ValueError("mean_update must be 'average' or 'weiszfeld'")
        self.mean_update = mean_update

        self.restart_patience = int(restart_patience)
        self.min_sigma = float(min_sigma_rel) * self.span_min
        self.max_sigma = float(max_sigma_rel) * self.span_min

        self.mean = np.clip(np.asarray(problem.initial_solution, dtype=float), self.lower, self.upper)
        self.sigma = 0.3 * self.span_min

        self.C = np.ones(self.dim, dtype=float)
        self.ps = np.zeros(self.dim, dtype=float)
        self.pc = np.zeros(self.dim, dtype=float)
        self.generation = 0

        self.best_f = float("inf")
        self.best_x = None
        self._no_improve_gens = 0

        self.chiN = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        self._update_strategy_params(mueff=2.0)  # placeholder until first weights

    def _n_t(self):
        if self.max_evals <= 0:
            return self.n_end
        frac = min(1.0, float(self.problem.evaluations) / float(self.max_evals))
        return self.n_start + (self.n_end - self.n_start) * frac

    def _mean_update_mode(self):
        return self.mean_update

    def _compute_weights(self, fvals):
        lam = len(fvals)
        order = np.argsort(fvals)
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        if lam <= 1:
            return np.ones(1, dtype=float), order

        base = 1.0 - (ranks.astype(float) / float(lam - 1))
        base = np.clip(base, 0.0, 1.0)

        n_t = self._n_t()
        w = np.zeros(lam, dtype=float)
        mask = base > 0.0
        w[mask] = np.power(base[mask], n_t)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    def _weiszfeld_update(self, x_points, weights, mean_old):
        eps = 1e-12 * (1.0 + self.span_min)
        diffs = x_points - mean_old[None, :]
        dists = np.linalg.norm(diffs, axis=1) + eps
        w2 = weights / dists
        total = float(np.sum(w2))
        if not np.isfinite(total) or total <= 0.0:
            return mean_old
        return (w2[:, None] * x_points).sum(axis=0) / total

    def _update_strategy_params(self, *, mueff):
        n = float(self.dim)
        mueff = float(max(1.0, mueff))
        self.mueff = mueff

        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs

    def _restart(self):
        self.mean = self.rng.uniform(self.lower, self.upper)
        self.sigma = 0.3 * self.span_min
        self.C[:] = 1.0
        self.ps[:] = 0.0
        self.pc[:] = 0.0
        self._no_improve_gens = 0
        self.generation = 0

    def run(self):
        if self.max_evals <= 0:
            return

        while self.problem.evaluations < self.max_evals and not self.problem.final_target_hit:
            remaining = int(self.max_evals - self.problem.evaluations)
            if remaining <= 0:
                break

            lam = min(self.lambda_, remaining)
            z = self.rng.randn(lam, self.dim)
            y = z * np.sqrt(self.C)[None, :]
            x = self.mean[None, :] + self.sigma * y
            x_eval = np.clip(x, self.lower, self.upper)

            fvals = np.empty(lam, dtype=float)
            for i in range(lam):
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    fvals = fvals[:i]
                    z = z[:i]
                    y = y[:i]
                    x_eval = x_eval[:i]
                    break
                fvals[i] = float(self.problem(x_eval[i]))

            if len(fvals) == 0:
                break

            idx_best = int(np.argmin(fvals))
            if fvals[idx_best] < self.best_f:
                self.best_f = float(fvals[idx_best])
                self.best_x = x_eval[idx_best].copy()
                self._no_improve_gens = 0
            else:
                self._no_improve_gens += 1

            weights_all, order = self._compute_weights(fvals)
            mu = min(self.mu, len(fvals))
            elite = order[:mu]

            w = weights_all[elite]
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                w = np.ones(mu, dtype=float) / float(mu)
            else:
                w = w / w_sum

            mueff = 1.0 / float(np.sum(w * w))
            self._update_strategy_params(mueff=mueff)

            mean_old = self.mean.copy()
            sigma_old = float(self.sigma)

            if self._mean_update_mode() == "average":
                y_w = (w[:, None] * y[elite]).sum(axis=0)
                mean_proposed = mean_old + self.c_mean * sigma_old * y_w
            else:
                x_sel = x_eval[elite]
                mean_target = self._weiszfeld_update(x_sel, w, mean_old)
                mean_proposed = (1.0 - self.c_mean) * mean_old + self.c_mean * mean_target
                y_w = (mean_proposed - mean_old) / max(sigma_old, 1e-30)

            self.mean = np.clip(mean_proposed, self.lower, self.upper)
            y_w = (self.mean - mean_old) / max(sigma_old, 1e-30)

            invsqrtC_y_w = y_w / np.sqrt(np.maximum(self.C, 1e-30))
            self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y_w
            norm_ps = float(np.linalg.norm(self.ps))
            self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
            self.sigma = float(np.clip(self.sigma, self.min_sigma, self.max_sigma))

            hsig_cond = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chiN
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (self.dim + 1.0)) else 0.0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            rank_mu = np.sum(w[:, None] * (y[elite] ** 2), axis=0)
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * rank_mu
            )
            self.C = np.maximum(self.C, 1e-30)

            self.generation += 1

            if self._no_improve_gens >= self.restart_patience:
                self._restart()


class BerwES2NoiseAdaptive(BerwES2):
    """
    BERW base optimizer with noise-adaptive selection temperature and selective reevaluation.

    Core idea (intended as a publishable algorithmic hook):
    - Use a small reevaluation budget to estimate misranking/noise level online.
    - Couple the rank-weight exponent (selection temperature) to the noise estimate:
      higher noise -> softer selection (less elitist), lower noise -> default schedule.

    This differs from classic CMA-ES noise handling (which primarily adapts sigma and the
    number of evaluations per fitness) by explicitly adapting the *selection intensity*
    as a noise-aware smoothing parameter.
    """

    def __init__(
        self,
        problem,
        max_evals,
        *,
        seed=0,
        popsize=None,
        mu=None,
        n_start=1.2,
        n_end=4.0,
        c_mean=1.0,
        mean_update="weiszfeld",
        restart_patience=80,
        min_sigma_rel=1e-14,
        max_sigma_rel=0.5,
        reeval_min=2,
        reeval_max_frac=0.3,
        reeval_extra_per_point=1,
        noise_ema_decay=0.2,
        temp_beta=4.0,
        temp_min_scale=0.25,
    ):
        super().__init__(
            problem,
            max_evals,
            seed=seed,
            popsize=popsize,
            mu=mu,
            n_start=n_start,
            n_end=n_end,
            c_mean=c_mean,
            mean_update=mean_update,
            restart_patience=restart_patience,
            min_sigma_rel=min_sigma_rel,
            max_sigma_rel=max_sigma_rel,
        )
        self.reeval_min = int(max(0, reeval_min))
        self.reeval_max_frac = float(max(0.0, min(1.0, reeval_max_frac)))
        self.reeval_extra_per_point = int(max(0, reeval_extra_per_point))

        self.noise_ema_decay = float(min(1.0, max(0.0, noise_ema_decay)))
        self.noise_ema = 0.0

        self.temp_beta = float(max(0.0, temp_beta))
        self.temp_min_scale = float(min(1.0, max(0.01, temp_min_scale)))
        self.temp_scale = 1.0

    def _effective_n_t(self):
        n_sched = super()._n_t()
        scale = float(np.clip(self.temp_scale, self.temp_min_scale, 1.0))
        return float(max(0.1, n_sched * scale))

    def _compute_weights(self, fvals):
        lam = len(fvals)
        order = np.argsort(fvals)
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        if lam <= 1:
            return np.ones(1, dtype=float), order

        base = 1.0 - (ranks.astype(float) / float(lam - 1))
        base = np.clip(base, 0.0, 1.0)

        n_t = self._effective_n_t()
        w = np.zeros(lam, dtype=float)
        mask = base > 0.0
        w[mask] = np.power(base[mask], n_t)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    @staticmethod
    def _rank_disagreement(f_a, f_b):
        """Average absolute rank change (normalized by lambda)."""
        lam = len(f_a)
        if lam <= 1:
            return 0.0
        order_a = np.argsort(f_a)
        order_b = np.argsort(f_b)
        ranks_a = np.empty(lam, dtype=int)
        ranks_b = np.empty(lam, dtype=int)
        ranks_a[order_a] = np.arange(lam)
        ranks_b[order_b] = np.arange(lam)
        return float(np.mean(np.abs(ranks_a - ranks_b)) / float(lam))

    def _noise_measure(self, f_raw, f_agg, mu):
        """Return a scalar noise proxy in [0, +inf) for temperature adaptation."""
        _ = mu
        return self._rank_disagreement(f_raw, f_agg)

    def _update_temperature(self, noise_level):
        if not np.isfinite(noise_level) or noise_level < 0.0:
            return
        a = self.noise_ema_decay
        self.noise_ema = (1.0 - a) * self.noise_ema + a * float(noise_level)
        # higher noise => smaller scale (softer selection). Only reduce vs schedule.
        self.temp_scale = 1.0 / (1.0 + self.temp_beta * self.noise_ema)
        self.temp_scale = float(np.clip(self.temp_scale, self.temp_min_scale, 1.0))

    def _maybe_record_state(self, *, noise_level, reeval_count):
        """
        Optional internal-state trace for mechanistic evidence.

        If the problem object has a list attribute `_berw_state_trace`, append a row:
        (evals, generation, noise_level, noise_ema, temp_scale, n_sched, n_eff, reeval_count, gate_closed,
         mueff, mueff_target, noise_s0, noise_s1, noise_z_pool_size, noise_z_abs_median,
         noise_z_clip_frac, noise_shape_ks, noise_shape_w1, noise_drift_ks, noise_drift_w1,
         noise_scale_fit_r2, noise_scale_pred_cv, noise_center_split_rel, noise_center_split_cv)
        """

        trace = getattr(self.problem, "_berw_state_trace", None)
        if trace is None:
            return

        noise_s0 = float(getattr(self, "_noise_s0", float("nan")))
        noise_s1 = float(getattr(self, "_noise_s1", float("nan")))
        z_pool = getattr(self, "_noise_z_pool", None)
        if z_pool is None:
            z_pool_size = 0
            z_abs_median = float("nan")
        else:
            z_pool = np.asarray(z_pool, dtype=float)
            z_pool_size = int(getattr(z_pool, "size", 0))
            z_abs_median = float(np.median(np.abs(z_pool))) if z_pool_size > 0 else float("nan")

        noise_z_clip_frac = float(getattr(self, "_noise_diag_z_clip_frac", float("nan")))
        noise_shape_ks = float(getattr(self, "_noise_diag_shape_ks", float("nan")))
        noise_shape_w1 = float(getattr(self, "_noise_diag_shape_w1", float("nan")))
        noise_drift_ks = float(getattr(self, "_noise_diag_drift_ks", float("nan")))
        noise_drift_w1 = float(getattr(self, "_noise_diag_drift_w1", float("nan")))
        noise_scale_fit_r2 = float(getattr(self, "_noise_diag_scale_fit_r2", float("nan")))
        noise_scale_pred_cv = float(getattr(self, "_noise_diag_scale_pred_cv", float("nan")))
        noise_center_split_rel = float(getattr(self, "_noise_diag_center_split_rel", float("nan")))
        noise_center_split_cv = float(getattr(self, "_noise_diag_center_split_cv", float("nan")))

        gate_closed = getattr(self, "gate_closed", None)
        trace.append(
            (
                int(getattr(self.problem, "evaluations", 0)),
                int(self.generation),
                float(noise_level) if noise_level is not None else float("nan"),
                float(self.noise_ema),
                float(self.temp_scale),
                float(BerwES2._n_t(self)),
                float(self._effective_n_t()),
                int(reeval_count),
                int(gate_closed) if gate_closed is not None else -1,
                float(getattr(self, "mueff", float("nan"))),
                float(getattr(self, "mueff_target", float("nan"))),
                noise_s0,
                noise_s1,
                z_pool_size,
                z_abs_median,
                noise_z_clip_frac,
                noise_shape_ks,
                noise_shape_w1,
                noise_drift_ks,
                noise_drift_w1,
                noise_scale_fit_r2,
                noise_scale_pred_cv,
                noise_center_split_rel,
                noise_center_split_cv,
            )
        )

    def _choose_reeval_indices(self, fvals):
        lam = len(fvals)
        if lam <= 0 or self.reeval_extra_per_point <= 0:
            return np.array([], dtype=int)
        max_count = int(math.floor(self.reeval_max_frac * lam))
        max_count = max(0, min(max_count, lam))
        if max_count <= 0 and self.reeval_min <= 0:
            return np.array([], dtype=int)

        # scale reevaluation count with estimated noise (bounded).
        base = max(0, self.reeval_min)
        extra = int(round(self.noise_ema * lam))
        count = base + extra
        count = max(0, min(count, max_count))
        if count <= 0:
            return np.array([], dtype=int)
        order = np.argsort(fvals)
        return order[:count]

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        """
        Perform selective reevaluation and return (fvals_used, noise_level, reeval_count).

        Subclasses may override this method to change:
        - which points are reevaluated,
        - how robust aggregation is done (median, bootstrap, etc),
        - how the noise proxy is computed.
        """

        noise_level = None
        fvals_used = fvals.copy()

        idx_reeval = self._choose_reeval_indices(fvals)
        reeval_count = int(getattr(idx_reeval, "size", 0))
        if idx_reeval.size > 0 and self.reeval_extra_per_point > 0:
            extra_budget = int(self.max_evals - self.problem.evaluations)
            max_points = extra_budget // self.reeval_extra_per_point
            if max_points > 0:
                idx_reeval = idx_reeval[:max_points]
            else:
                idx_reeval = np.array([], dtype=int)
            reeval_count = int(getattr(idx_reeval, "size", 0))

        if idx_reeval.size > 0:
            fvals_agg = fvals.copy()
            for idx in idx_reeval:
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    break
                vals = [float(fvals[idx])]
                for _ in range(self.reeval_extra_per_point):
                    if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                        break
                    vals.append(float(self.problem(x_eval[idx])))
                f_robust = float(np.median(vals))
                fvals_agg[idx] = f_robust
                fvals_used[idx] = f_robust

            mu0 = min(self.mu, len(fvals_used))
            self._last_mu = int(max(1, min(mu0, self._last_lam)))
            noise_level = self._noise_measure(fvals, fvals_agg, mu0)
            self._update_temperature(noise_level)

        return fvals_used, noise_level, reeval_count

    def run(self):
        if self.max_evals <= 0:
            return

        while self.problem.evaluations < self.max_evals and not self.problem.final_target_hit:
            remaining = int(self.max_evals - self.problem.evaluations)
            if remaining <= 0:
                break

            lam = min(self.lambda_, remaining)
            z = self.rng.randn(lam, self.dim)
            y = z * np.sqrt(self.C)[None, :]
            x = self.mean[None, :] + self.sigma * y
            x_eval = np.clip(x, self.lower, self.upper)

            fvals = np.empty(lam, dtype=float)
            for i in range(lam):
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    fvals = fvals[:i]
                    z = z[:i]
                    y = y[:i]
                    x_eval = x_eval[:i]
                    break
                fvals[i] = float(self.problem(x_eval[i]))

            if len(fvals) == 0:
                break

            # For noise-adaptive variants that need access to the actual lambda/mu.
            self._last_lam = int(len(fvals))
            self._last_mu = int(max(1, min(self.mu, self._last_lam)))

            # selective reevaluation for noise estimation + robust aggregation
            fvals_used, noise_level, reeval_count = self._reevaluate_and_aggregate(x_eval, fvals)

            idx_best = int(np.argmin(fvals_used))
            if fvals_used[idx_best] < self.best_f:
                self.best_f = float(fvals_used[idx_best])
                self.best_x = x_eval[idx_best].copy()
                self._no_improve_gens = 0
            else:
                self._no_improve_gens += 1

            weights_all, order = self._compute_weights(fvals_used)
            mu = min(self.mu, len(fvals_used))
            elite = order[:mu]

            w = weights_all[elite]
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                w = np.ones(mu, dtype=float) / float(mu)
            else:
                w = w / w_sum

            mueff = 1.0 / float(np.sum(w * w))
            self._update_strategy_params(mueff=mueff)

            mean_old = self.mean.copy()
            sigma_old = float(self.sigma)

            if self._mean_update_mode() == "average":
                y_w = (w[:, None] * y[elite]).sum(axis=0)
                mean_proposed = mean_old + self.c_mean * sigma_old * y_w
            else:
                x_sel = x_eval[elite]
                mean_target = self._weiszfeld_update(x_sel, w, mean_old)
                mean_proposed = (1.0 - self.c_mean) * mean_old + self.c_mean * mean_target

            self.mean = np.clip(mean_proposed, self.lower, self.upper)
            y_w = (self.mean - mean_old) / max(sigma_old, 1e-30)

            invsqrtC_y_w = y_w / np.sqrt(np.maximum(self.C, 1e-30))
            self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y_w
            norm_ps = float(np.linalg.norm(self.ps))
            self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
            self.sigma = float(np.clip(self.sigma, self.min_sigma, self.max_sigma))

            hsig_cond = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chiN
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (self.dim + 1.0)) else 0.0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            rank_mu = np.sum(w[:, None] * (y[elite] ** 2), axis=0)
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * rank_mu
            )
            self.C = np.maximum(self.C, 1e-30)

            self._maybe_record_state(noise_level=noise_level, reeval_count=reeval_count)

            self.generation += 1

            if self._no_improve_gens >= self.restart_patience:
                self._restart()


class BerwES2NoiseAdaptiveSel(BerwES2NoiseAdaptive):
    """
    Noise-adaptive variant focused on *selection stability*.

    Rationale:
    - Under noise, what matters most for ES updates is whether the elite set (top-mu)
      is correctly identified. Misranking far away from the mu-boundary has limited
      effect on the update direction.
    - Therefore we (i) preferentially reevaluate candidates around the elite boundary,
      and (ii) use the elite-set flip rate as the noise proxy for temperature scaling.
    """

    def __init__(self, *args, reeval_band=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.reeval_band = int(max(1, reeval_band))

    def _choose_reeval_indices(self, fvals):
        lam = len(fvals)
        if lam <= 0 or self.reeval_extra_per_point <= 0:
            return np.array([], dtype=int)

        max_count = int(math.floor(self.reeval_max_frac * lam))
        max_count = max(0, min(max_count, lam))
        if max_count <= 0 and self.reeval_min <= 0:
            return np.array([], dtype=int)

        base = max(0, self.reeval_min)
        extra = int(round(self.noise_ema * lam))
        count = base + extra
        count = max(0, min(count, max_count))
        if count <= 0:
            return np.array([], dtype=int)

        order = np.argsort(fvals)
        mu0 = min(self.mu, lam)
        center = max(0, min(lam - 1, mu0 - 1))
        start = max(0, center - self.reeval_band)
        end = min(lam, center + self.reeval_band + 1)
        band = order[start:end]

        if len(band) >= count:
            return band[:count]
        # If the band is too small (small lambda), fall back to best points.
        rest = order[:count]
        idx = np.unique(np.concatenate([band, rest]))
        return idx[:count]

    def _noise_measure(self, f_raw, f_agg, mu):
        mu = int(max(1, min(mu, len(f_raw))))
        order_a = np.argsort(f_raw)
        order_b = np.argsort(f_agg)
        top_a = set(order_a[:mu].tolist())
        top_b = set(order_b[:mu].tolist())
        overlap = len(top_a.intersection(top_b))
        return float(1.0 - overlap / float(mu))


class BerwES2NoiseReevalBoundary(BerwES2NoiseAdaptiveSel):
    """
    Ablation: boundary reevaluation + robust aggregation, but *no* temperature adaptation.

    Purpose:
    - Separate the effect of (i) spending evaluations on selective reevaluation/median,
      from (ii) the core mechanism: adapting selection intensity (power-lift temperature)
      based on a noise proxy.
    """

    def _update_temperature(self, noise_level):
        # Intentionally do nothing: keep temp_scale=1 and noise_ema fixed.
        _ = noise_level
        return


class BerwES2NoiseAdaptiveSelGate(BerwES2NoiseAdaptiveSel):
    """
    Stability-gated NoiseAdaptiveSel.

    Goal:
    - Fix a key concern: NoiseAdaptiveSel can lose when noise is absent/weak,
      because (i) reevaluation spends budget and (ii) temperature scaling may soften
      selection unnecessarily.

    Mechanism:
    - If the elite-set flip-rate proxy is consistently ~0, we *close the gate*:
        - switch to CMA-style log recombination weights + average mean update
        - disable temperature scaling (temp_scale := 1)
        - skip reevaluation most generations
    - While gated, we run a small periodic *probe* (boundary reevaluation) to detect
      if noise becomes relevant again. If flip-rate exceeds a reopen threshold, we
      reopen the gate and revert to NoiseAdaptiveSel behavior.

    This is intentionally lightweight: it adds a single state-machine layer on top
    of the existing proxy, keeping the core idea centered on "selection-stability
    controls selection intensity" rather than heavy engineering.
    """

    def __init__(
        self,
        *args,
        gate_close_threshold=0.02,
        gate_open_threshold=0.08,
        gate_patience=2,
        gate_warmup_gens=20,
        gate_probe_interval=10,
        gate_probe_count=2,
        gate_close_ema_threshold=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate_close_threshold = float(max(0.0, gate_close_threshold))
        self.gate_open_threshold = float(max(self.gate_close_threshold, gate_open_threshold))
        self.gate_patience = int(max(1, gate_patience))
        self.gate_warmup_gens = int(max(0, gate_warmup_gens))
        self.gate_probe_interval = int(max(1, gate_probe_interval))
        self.gate_probe_count = int(max(1, gate_probe_count))
        self.gate_close_ema_threshold = float(max(0.0, gate_close_ema_threshold))

        # Start gated (conservative): do a small probe at generation 0. This avoids
        # paying the full reevaluation cost in the common near-deterministic regime.
        self.gate_closed = True
        self._low_noise_count = 0

    def _effective_n_t(self):
        n_sched = BerwES2._n_t(self)
        if self.gate_closed:
            return float(max(0.1, n_sched))
        return super()._effective_n_t()

    def _mean_update_mode(self):
        if self.gate_closed:
            return "average"
        return self.mean_update

    def _compute_weights(self, fvals):
        if not self.gate_closed:
            return super()._compute_weights(fvals)

        lam = len(fvals)
        order = np.argsort(fvals)
        if lam <= 1:
            return np.ones(1, dtype=float), order

        mu = int(max(1, min(self.mu, lam)))
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w = np.zeros(lam, dtype=float)
        mask = ranks < mu
        w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
        w = np.maximum(w, 0.0)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    def _choose_reeval_indices(self, fvals):
        if not self.gate_closed:
            return super()._choose_reeval_indices(fvals)

        lam = len(fvals)
        if lam <= 0 or self.reeval_extra_per_point <= 0:
            return np.array([], dtype=int)

        probe_interval = 1 if self.generation < self.gate_warmup_gens else self.gate_probe_interval
        if (self.generation % probe_interval) != 0:
            return np.array([], dtype=int)

        count = int(min(max(1, self.gate_probe_count), lam))

        # Probe around the elite boundary (same rationale as NoiseAdaptiveSel).
        order = np.argsort(fvals)
        mu0 = min(self.mu, lam)
        center = max(0, min(lam - 1, mu0 - 1))
        start = max(0, center - self.reeval_band)
        end = min(lam, center + self.reeval_band + 1)
        band = order[start:end]

        if len(band) >= count:
            return band[:count]
        rest = order[:count]
        idx = np.unique(np.concatenate([band, rest]))
        return idx[:count]

    def _update_temperature(self, noise_level):
        if not np.isfinite(noise_level) or noise_level < 0.0:
            return

        super()._update_temperature(noise_level)

        if self.gate_closed:
            if noise_level > self.gate_open_threshold:
                self.gate_closed = False
                self._low_noise_count = 0
            else:
                # enforce "no adaptation while gated"
                self.temp_scale = 1.0
            return

        # Close the gate only when the *smoothed* proxy is consistently tiny.
        # This avoids spuriously closing under true noise when the elite set
        # happens to be stable for a few generations.
        if self.noise_ema < self.gate_close_ema_threshold:
            self._low_noise_count += 1
        else:
            self._low_noise_count = 0

        if self._low_noise_count >= self.gate_patience:
            self.gate_closed = True
            self.temp_scale = 1.0


class BerwES2NoiseAdaptiveSelESS(BerwES2NoiseAdaptiveSel):
    """
    NoiseAdaptiveSel variant with stability-controlled *effective sample size* (mueff).

    Motivation:
    - Temperature scaling as `n_eff = n_sched * temp_scale` can look like ad-hoc tuning.
    - In ES/CMA updates, selection weights impact the update "noise" through the effective
      number of samples: `mueff = 1/sum_i w_i^2` (Renyi-2 entropy of weights).
    - We use the same selection-stability proxy (elite-set flip rate) but map it to a
      *target mueff* and then solve for the power-lift exponent `n_eff` that achieves it.

    This keeps the core idea centered on "selection stability controls selection intensity",
    but expresses the control law in a more principled quantity (mueff) that appears
    directly in the CMA update equations.
    """

    def __init__(self, *args, ess_min_mueff=1.0, ess_max_mu_frac=1.0, ess_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ess_min_mueff = float(max(1.0, ess_min_mueff))
        self.ess_max_mu_frac = float(np.clip(float(ess_max_mu_frac), 0.0, 1.0))
        self.ess_power = float(max(0.1, ess_power))
        self.mueff_target = float("nan")

    @staticmethod
    def _mueff_power_lift(n, lam, mu):
        lam = int(lam)
        mu = int(mu)
        if mu <= 1 or lam <= 1:
            return 1.0
        ranks = np.arange(mu, dtype=float)
        base = 1.0 - (ranks / float(lam - 1))
        base = np.clip(base, 0.0, 1.0)
        w = np.power(base, float(max(0.0, n)))
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            return float(mu)
        w = w / total
        return 1.0 / float(np.sum(w * w))

    @classmethod
    def _solve_n_for_mueff(cls, target_mueff, lam, mu, n_max):
        lam = int(lam)
        mu = int(mu)
        if mu <= 1 or lam <= 1:
            return float(max(0.0, n_max))

        target = float(np.clip(float(target_mueff), 1.0, float(mu)))
        lo = 0.0
        hi = float(max(0.0, n_max))
        # mueff(lo) >= target >= mueff(hi) is expected (target is between uniform and base).
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            m = cls._mueff_power_lift(mid, lam, mu)
            if m > target:
                lo = mid  # too uniform => increase n
            else:
                hi = mid  # too elitist => decrease n
        return float(hi)

    def _update_temperature(self, noise_level):
        if not np.isfinite(noise_level) or noise_level < 0.0:
            return

        a = self.noise_ema_decay
        self.noise_ema = (1.0 - a) * self.noise_ema + a * float(noise_level)

        n_sched = float(BerwES2._n_t(self))
        lam = int(getattr(self, "_last_lam", self.lambda_))
        mu = int(getattr(self, "_last_mu", self.mu))
        mu = int(max(1, min(mu, lam)))

        if n_sched <= 0.0 or mu <= 1 or lam <= 1:
            self.temp_scale = 1.0
            self.mueff_target = 1.0
            return

        mueff_base = self._mueff_power_lift(n_sched, lam, mu)
        mueff_max = float(mu) * self.ess_max_mu_frac
        mueff_max = float(max(mueff_base, min(float(mu), mueff_max)))

        # Parameter-free control law (principled heuristic):
        # If the elite set has overlap (1 - noise_ema), treat this as a proxy for the
        # fraction of "correct" selected samples. To keep the effective *correct*
        # sample size roughly constant, aim for:
        #     (1 - noise_ema) * mueff_target ≈ mueff_base
        # =>  mueff_target ≈ mueff_base / (1 - noise_ema)
        denom = float(max(1e-12, 1.0 - float(self.noise_ema)))
        target = float(mueff_base / (denom**self.ess_power))
        target = float(np.clip(target, self.ess_min_mueff, mueff_max))
        self.mueff_target = target

        n_eff = self._solve_n_for_mueff(target, lam, mu, n_sched)
        self.temp_scale = float(np.clip(n_eff / n_sched, self.temp_min_scale, 1.0))


class BerwES2NoiseAdaptiveSelGateESS(BerwES2NoiseAdaptiveSelESS):
    """Stability gate + ESS-based selection-intensity control (mueff targeting)."""

    def __init__(
        self,
        *args,
        gate_close_threshold=0.02,
        gate_open_threshold=0.08,
        gate_patience=2,
        gate_warmup_gens=20,
        gate_probe_interval=10,
        gate_probe_count=2,
        gate_close_ema_threshold=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate_close_threshold = float(max(0.0, gate_close_threshold))
        self.gate_open_threshold = float(max(self.gate_close_threshold, gate_open_threshold))
        self.gate_patience = int(max(1, gate_patience))
        self.gate_warmup_gens = int(max(0, gate_warmup_gens))
        self.gate_probe_interval = int(max(1, gate_probe_interval))
        self.gate_probe_count = int(max(1, gate_probe_count))
        self.gate_close_ema_threshold = float(max(0.0, gate_close_ema_threshold))

        self.gate_closed = True
        self._low_noise_count = 0

    def _effective_n_t(self):
        n_sched = BerwES2._n_t(self)
        if self.gate_closed:
            return float(max(0.1, n_sched))
        return super()._effective_n_t()

    def _mean_update_mode(self):
        if self.gate_closed:
            return "average"
        return self.mean_update

    def _compute_weights(self, fvals):
        if not self.gate_closed:
            return super()._compute_weights(fvals)

        lam = len(fvals)
        order = np.argsort(fvals)
        if lam <= 1:
            return np.ones(1, dtype=float), order

        mu = int(max(1, min(self.mu, lam)))
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w = np.zeros(lam, dtype=float)
        mask = ranks < mu
        w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
        w = np.maximum(w, 0.0)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    def _choose_reeval_indices(self, fvals):
        if not self.gate_closed:
            return super()._choose_reeval_indices(fvals)

        lam = len(fvals)
        if lam <= 0 or self.reeval_extra_per_point <= 0:
            return np.array([], dtype=int)

        probe_interval = 1 if self.generation < self.gate_warmup_gens else self.gate_probe_interval
        if (self.generation % probe_interval) != 0:
            return np.array([], dtype=int)

        count = int(min(max(1, self.gate_probe_count), lam))

        order = np.argsort(fvals)
        mu0 = min(self.mu, lam)
        center = max(0, min(lam - 1, mu0 - 1))
        start = max(0, center - self.reeval_band)
        end = min(lam, center + self.reeval_band + 1)
        band = order[start:end]

        if len(band) >= count:
            return band[:count]
        rest = order[:count]
        idx = np.unique(np.concatenate([band, rest]))
        return idx[:count]

    def _update_temperature(self, noise_level):
        if not np.isfinite(noise_level) or noise_level < 0.0:
            return

        super()._update_temperature(noise_level)

        if self.gate_closed:
            if noise_level > self.gate_open_threshold:
                self.gate_closed = False
                self._low_noise_count = 0
            else:
                self.temp_scale = 1.0
            return

        if self.noise_ema < self.gate_close_ema_threshold:
            self._low_noise_count += 1
        else:
            self._low_noise_count = 0

        if self._low_noise_count >= self.gate_patience:
            self.gate_closed = True
            self.temp_scale = 1.0


class BerwES2NoiseAdaptiveSelBootstrapESS(BerwES2NoiseAdaptiveSelESS):
    """
    ESS-control variant that measures selection instability via a simple bootstrap split.

    Compared to NoiseAdaptiveSelESS:
    - We still reevaluate only a small subset (near the elite boundary by default).
    - For each reevaluated point, we split its repeated evaluations into two groups
      (odd/even indices) and compute two robust estimates (median).
    - The noise proxy is the elite-set flip rate between these *two* estimates.

    This yields a more "self-contained" stability signal: it measures how sensitive
    the elite set is to resampling, rather than comparing raw vs aggregated once.
    """

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        noise_level = None
        fvals_used = fvals.copy()

        idx_reeval = self._choose_reeval_indices(fvals)
        reeval_count = int(getattr(idx_reeval, "size", 0))
        if idx_reeval.size > 0 and self.reeval_extra_per_point > 0:
            extra_budget = int(self.max_evals - self.problem.evaluations)
            max_points = extra_budget // self.reeval_extra_per_point
            if max_points > 0:
                idx_reeval = idx_reeval[:max_points]
            else:
                idx_reeval = np.array([], dtype=int)
            reeval_count = int(getattr(idx_reeval, "size", 0))

        if idx_reeval.size > 0:
            f_a = fvals.copy()
            f_b = fvals.copy()
            for idx in idx_reeval:
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    break
                vals = [float(fvals[idx])]
                for _ in range(self.reeval_extra_per_point):
                    if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                        break
                    vals.append(float(self.problem(x_eval[idx])))

                # Used for actual selection/update.
                f_med = float(np.median(vals))
                fvals_used[idx] = f_med

                # Bootstrap split: odd/even indices (at least 1 sample each if possible).
                vals_a = vals[::2]
                vals_b = vals[1::2]
                if not vals_b:
                    vals_b = vals_a
                if not vals_a:
                    vals_a = vals_b
                f_a[idx] = float(np.median(vals_a))
                f_b[idx] = float(np.median(vals_b))

            mu0 = min(self.mu, len(fvals_used))
            self._last_mu = int(max(1, min(mu0, self._last_lam)))
            noise_level = self._noise_measure(f_a, f_b, mu0)
            self._update_temperature(noise_level)

        return fvals_used, noise_level, reeval_count


class BerwES2NoiseAdaptiveSelBootstrapGateESS(BerwES2NoiseAdaptiveSelGateESS):
    """Bootstrap stability signal + ESS control + stability gate."""

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        noise_level = None
        fvals_used = fvals.copy()

        idx_reeval = self._choose_reeval_indices(fvals)
        reeval_count = int(getattr(idx_reeval, "size", 0))
        if idx_reeval.size > 0 and self.reeval_extra_per_point > 0:
            extra_budget = int(self.max_evals - self.problem.evaluations)
            max_points = extra_budget // self.reeval_extra_per_point
            if max_points > 0:
                idx_reeval = idx_reeval[:max_points]
            else:
                idx_reeval = np.array([], dtype=int)
            reeval_count = int(getattr(idx_reeval, "size", 0))

        if idx_reeval.size > 0:
            f_a = fvals.copy()
            f_b = fvals.copy()
            for idx in idx_reeval:
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    break
                vals = [float(fvals[idx])]
                for _ in range(self.reeval_extra_per_point):
                    if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                        break
                    vals.append(float(self.problem(x_eval[idx])))

                f_med = float(np.median(vals))
                fvals_used[idx] = f_med

                vals_a = vals[::2]
                vals_b = vals[1::2]
                if not vals_b:
                    vals_b = vals_a
                if not vals_a:
                    vals_a = vals_b
                f_a[idx] = float(np.median(vals_a))
                f_b[idx] = float(np.median(vals_b))

            mu0 = min(self.mu, len(fvals_used))
            self._last_mu = int(max(1, min(mu0, self._last_lam)))
            noise_level = self._noise_measure(f_a, f_b, mu0)
            self._update_temperature(noise_level)

        return fvals_used, noise_level, reeval_count


class BerwES2NoiseAdaptiveSelBootstrapWeights(BerwES2NoiseAdaptiveSel):
    """
    Bootstrap-expected selection weights.

    Key idea:
    - Use a small number of repeated evaluations for boundary points to build a tiny
      empirical distribution per candidate.
    - Use bootstrap resampling to approximate the distribution of *ranks* under noise,
      and update using the *expected* (truncated) power-lift weights.

    Compared to temperature/ESS control, this is closer to "probabilistic elite membership":
    near the boundary, candidates can receive non-zero weight proportional to how often
    they appear in the top-μ under resampling.
    """

    def __init__(self, *args, bootstrap_samples=32, bootstrap_seed_offset=77889, **kwargs):
        # By default, let the bootstrap do the softening; keep n_t schedule as-is.
        kwargs.setdefault("temp_beta", 0.0)
        super().__init__(*args, **kwargs)

        self.bootstrap_samples = int(max(1, bootstrap_samples))
        seed = int(kwargs.get("seed", 0))
        self._bootstrap_rng = np.random.RandomState((seed + int(bootstrap_seed_offset)) & 0xFFFFFFFF)

        # Per-generation storage: list of np.ndarray of observed f-values for each candidate.
        self._last_point_samples = None
        # Per-generation pooled noise model (relative residuals) for non-reevaluated points.
        self._last_noise_rel_residuals = None
        self._last_noise_abs_floor = 0.0

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        noise_level = None
        fvals_used = fvals.copy()

        lam = int(len(fvals))
        self._last_point_samples = [np.asarray([float(fvals[i])], dtype=float) for i in range(lam)]
        self._last_noise_rel_residuals = None
        self._last_noise_abs_floor = 0.0

        idx_reeval = self._choose_reeval_indices(fvals)
        reeval_count = int(getattr(idx_reeval, "size", 0))
        if idx_reeval.size > 0 and self.reeval_extra_per_point > 0:
            extra_budget = int(self.max_evals - self.problem.evaluations)
            max_points = extra_budget // self.reeval_extra_per_point
            if max_points > 0:
                idx_reeval = idx_reeval[:max_points]
            else:
                idx_reeval = np.array([], dtype=int)
            reeval_count = int(getattr(idx_reeval, "size", 0))

        if idx_reeval.size > 0:
            # Bootstrap split signal for a self-contained stability proxy.
            f_a = fvals.copy()
            f_b = fvals.copy()
            rel_residuals = []
            abs_residual_mags = []

            for idx in idx_reeval:
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    break
                vals = [float(fvals[idx])]
                for _ in range(self.reeval_extra_per_point):
                    if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                        break
                    vals.append(float(self.problem(x_eval[idx])))

                arr = np.asarray(vals, dtype=float)
                self._last_point_samples[int(idx)] = arr

                f_med = float(np.median(arr))
                fvals_used[idx] = f_med

                denom = float(abs(f_med) + 1e-12)
                rel_residuals.extend(((arr - f_med) / denom).tolist())
                abs_residual_mags.extend(np.abs(arr - f_med).tolist())

                vals_a = arr[::2]
                vals_b = arr[1::2]
                if vals_b.size == 0:
                    vals_b = vals_a
                if vals_a.size == 0:
                    vals_a = vals_b
                f_a[idx] = float(np.median(vals_a))
                f_b[idx] = float(np.median(vals_b))

            if rel_residuals:
                self._last_noise_rel_residuals = np.asarray(rel_residuals, dtype=float)
                self._last_noise_abs_floor = float(np.median(np.asarray(abs_residual_mags, dtype=float)))

            mu0 = min(self.mu, len(fvals_used))
            self._last_mu = int(max(1, min(mu0, self._last_lam)))
            noise_level = self._noise_measure(f_a, f_b, mu0)
            self._update_temperature(noise_level)

        return fvals_used, noise_level, reeval_count

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            # Fallback: behave like NoiseAdaptiveSel (deterministic weights on fvals).
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        rel_pool = self._last_noise_rel_residuals
        abs_floor = float(max(0.0, self._last_noise_abs_floor))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if rel_pool is None or rel_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        r = float(rel_pool[int(self._bootstrap_rng.randint(0, int(rel_pool.size)))])
                        scale = float(max(abs(base), abs_floor, 1e-12))
                        f_boot[i] = base + r * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_avg = acc / float(B)
        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        order = np.argsort(-w_avg)  # high expected weight first
        return w_avg, order


class BerwES2NoiseAdaptiveSelBootstrapWeightsTrimmed(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    BootstrapWeights with a robust aggregation across bootstrap replicates.

    Motivation:
    - Under heavy-tailed noise, some bootstrap rank realizations can be extreme outliers.
    - The naive mean over bootstrap weights can then be a high-variance estimator of E[w(rank)].

    This variant uses a coordinate-wise trimmed mean over bootstrap replicates and
    renormalizes the resulting weights. For light-tailed regimes, it reduces to the
    standard mean when `trim_frac=0`.
    """

    def __init__(self, *args, trim_frac=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_frac = float(np.clip(float(trim_frac), 0.0, 0.49))

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        f_boot = np.empty(lam, dtype=float)
        rel_pool = self._last_noise_rel_residuals
        abs_floor = float(max(0.0, self._last_noise_abs_floor))

        W = np.empty((B, lam), dtype=float)
        for b in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if rel_pool is None or rel_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        r = float(rel_pool[int(self._bootstrap_rng.randint(0, int(rel_pool.size)))])
                        scale = float(max(abs(base), abs_floor, 1e-12))
                        f_boot[i] = base + r * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            W[b, :] = w

        trim = float(self.trim_frac)
        k = int(math.floor(trim * float(B)))
        if k <= 0:
            w_avg = np.mean(W, axis=0)
        else:
            W_sorted = np.sort(W, axis=0)
            w_avg = np.mean(W_sorted[k : B - k, :], axis=0)

        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)


class BerwES2NoiseAdaptiveSelBootstrapCMAWeights(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    Bootstrap-expected *CMA-ES log weights* under noisy ranks.

    This is a "drop-in" variant of BootstrapWeights that replaces the rank-weight exponent
    weight function with standard CMA-ES log-weights, then takes the bootstrap
    expectation over noisy rank realizations.

    Motivation:
    - strengthens the claim that "probabilistic elite membership" is weight-function agnostic,
      not tied to a specific power-lift form,
    - provides a cleaner comparison point against CMA-ES baselines on bbob-noisy.
    """

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        rel_pool = self._last_noise_rel_residuals
        abs_floor = float(max(0.0, self._last_noise_abs_floor))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if rel_pool is None or rel_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        r = float(rel_pool[int(self._bootstrap_rng.randint(0, int(rel_pool.size)))])
                        scale = float(max(abs(base), abs_floor, 1e-12))
                        f_boot[i] = base + r * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
                w = np.maximum(w, 0.0)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_avg = acc / float(B)
        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)

    def run(self):
        if self.max_evals <= 0:
            return

        while self.problem.evaluations < self.max_evals and not self.problem.final_target_hit:
            remaining = int(self.max_evals - self.problem.evaluations)
            if remaining <= 0:
                break

            lam = min(self.lambda_, remaining)
            z = self.rng.randn(lam, self.dim)
            y = z * np.sqrt(self.C)[None, :]
            x = self.mean[None, :] + self.sigma * y
            x_eval = np.clip(x, self.lower, self.upper)

            fvals = np.empty(lam, dtype=float)
            for i in range(lam):
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    fvals = fvals[:i]
                    z = z[:i]
                    y = y[:i]
                    x_eval = x_eval[:i]
                    break
                fvals[i] = float(self.problem(x_eval[i]))

            if len(fvals) == 0:
                break

            self._last_lam = int(len(fvals))
            self._last_mu = int(max(1, min(self.mu, self._last_lam)))

            fvals_used, noise_level, reeval_count = self._reevaluate_and_aggregate(x_eval, fvals)

            idx_best = int(np.argmin(fvals_used))
            if fvals_used[idx_best] < self.best_f:
                self.best_f = float(fvals_used[idx_best])
                self.best_x = x_eval[idx_best].copy()
                self._no_improve_gens = 0
            else:
                self._no_improve_gens += 1

            weights_all, _ = self._compute_weights(fvals_used)
            w = np.asarray(weights_all, dtype=float)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                w = np.ones(len(fvals_used), dtype=float) / float(len(fvals_used))
            else:
                w = w / w_sum

            mueff = 1.0 / float(np.sum(w * w))
            self._update_strategy_params(mueff=mueff)

            mean_old = self.mean.copy()
            sigma_old = float(self.sigma)

            if self._mean_update_mode() == "average":
                y_w = (w[:, None] * y).sum(axis=0)
                mean_proposed = mean_old + self.c_mean * sigma_old * y_w
            else:
                mean_target = self._weiszfeld_update(x_eval, w, mean_old)
                mean_proposed = (1.0 - self.c_mean) * mean_old + self.c_mean * mean_target

            self.mean = np.clip(mean_proposed, self.lower, self.upper)
            y_w = (self.mean - mean_old) / max(sigma_old, 1e-30)

            invsqrtC_y_w = y_w / np.sqrt(np.maximum(self.C, 1e-30))
            self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y_w
            norm_ps = float(np.linalg.norm(self.ps))
            self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
            self.sigma = float(np.clip(self.sigma, self.min_sigma, self.max_sigma))

            hsig_cond = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chiN
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (self.dim + 1.0)) else 0.0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            rank_mu = np.sum(w[:, None] * (y**2), axis=0)
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * rank_mu
            )
            self.C = np.maximum(self.C, 1e-30)

            self._maybe_record_state(noise_level=noise_level, reeval_count=reeval_count)

            self.generation += 1

            if self._no_improve_gens >= self.restart_patience:
                self._restart()


class BerwES2NoiseAdaptiveSelBootstrapWeightsGate(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    BootstrapWeights + stability gate (deterministic / low-noise fallback).

    Goal:
    - BootstrapWeights can lose on low-noise regimes (extra overhead + unnecessary smoothing).
    - This gate reverts to CMA-style log recombination weights with average mean update when
      the elite-set stability proxy indicates near-determinism.
    - Periodic probing reevaluates a few boundary points to detect if noise becomes relevant.
    """

    def __init__(
        self,
        *args,
        gate_close_threshold=0.02,
        gate_open_threshold=0.08,
        gate_patience=2,
        gate_warmup_gens=20,
        gate_probe_interval=10,
        gate_probe_count=2,
        gate_close_ema_threshold=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate_close_threshold = float(max(0.0, gate_close_threshold))
        self.gate_open_threshold = float(max(self.gate_close_threshold, gate_open_threshold))
        self.gate_patience = int(max(1, gate_patience))
        self.gate_warmup_gens = int(max(0, gate_warmup_gens))
        self.gate_probe_interval = int(max(1, gate_probe_interval))
        self.gate_probe_count = int(max(1, gate_probe_count))
        self.gate_close_ema_threshold = float(max(0.0, gate_close_ema_threshold))

        self.gate_closed = True
        self._low_noise_count = 0

    def _mean_update_mode(self):
        if self.gate_closed:
            return "average"
        return self.mean_update

    def _compute_weights(self, fvals):
        if not self.gate_closed:
            return super()._compute_weights(fvals)

        lam = len(fvals)
        order = np.argsort(fvals)
        if lam <= 1:
            return np.ones(1, dtype=float), order

        mu = int(max(1, min(self.mu, lam)))
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w = np.zeros(lam, dtype=float)
        mask = ranks < mu
        w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
        w = np.maximum(w, 0.0)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    def _choose_reeval_indices(self, fvals):
        if not self.gate_closed:
            return super()._choose_reeval_indices(fvals)

        lam = len(fvals)
        if lam <= 0 or self.reeval_extra_per_point <= 0:
            return np.array([], dtype=int)

        probe_interval = 1 if self.generation < self.gate_warmup_gens else self.gate_probe_interval
        if (self.generation % probe_interval) != 0:
            return np.array([], dtype=int)

        count = int(min(max(1, self.gate_probe_count), lam))

        order = np.argsort(fvals)
        mu0 = min(self.mu, lam)
        center = max(0, min(lam - 1, mu0 - 1))
        start = max(0, center - self.reeval_band)
        end = min(lam, center + self.reeval_band + 1)
        band = order[start:end]

        if len(band) >= count:
            return band[:count]
        rest = order[:count]
        idx = np.unique(np.concatenate([band, rest]))
        return idx[:count]

    def _update_temperature(self, noise_level):
        if not np.isfinite(noise_level) or noise_level < 0.0:
            return

        super()._update_temperature(noise_level)

        if self.gate_closed:
            if noise_level > self.gate_open_threshold:
                self.gate_closed = False
                self._low_noise_count = 0
            return

        if noise_level < self.gate_close_threshold or self.noise_ema < self.gate_close_ema_threshold:
            self._low_noise_count += 1
        else:
            self._low_noise_count = 0

        if self._low_noise_count >= self.gate_patience:
            self.gate_closed = True


class BerwES2NoiseAdaptiveSelBootstrapWeightsBlend(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    Soft-gated BootstrapWeights: blend deterministic CMA log-weights with probabilistic
    bootstrap-expected weights based on the online noise proxy.

    This avoids a hard state machine and tends to preserve BootstrapWeights' gains on
    noisy regimes while reducing regressions when the proxy indicates near-determinism.
    """

    def __init__(self, *args, blend_beta=6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.blend_beta = float(max(0.0, blend_beta))

    def _compute_weights(self, fvals):
        w_boot, _ = super()._compute_weights(fvals)
        w_boot = np.asarray(w_boot, dtype=float)

        lam = int(len(fvals))
        if lam <= 1:
            return w_boot, np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        order = np.argsort(fvals)
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w_det = np.zeros(lam, dtype=float)
        mask = ranks < mu
        w_det[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
        w_det = np.maximum(w_det, 0.0)
        s_det = float(np.sum(w_det))
        if not np.isfinite(s_det) or s_det <= 0.0:
            w_det[:] = 1.0 / float(lam)
        else:
            w_det = w_det / s_det

        noise = float(np.clip(getattr(self, "noise_ema", 0.0), 0.0, 1.0))
        g = 1.0 - math.exp(-self.blend_beta * noise)
        g = float(np.clip(g, 0.0, 1.0))

        w = (1.0 - g) * w_det + g * w_boot
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            w[:] = 1.0 / float(lam)
        else:
            w = w / s

        return w, np.argsort(-w)


class BerwES2NoiseAdaptiveSelBootstrapWeightsActive(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    BootstrapWeights + uncertainty-aware resampling (single-generation active refinement).

    Mechanism:
    - Use the same bootstrap machinery to estimate top-μ membership probabilities p_i.
    - Spend the limited reevaluation budget on candidates with highest uncertainty p_i(1-p_i),
      rather than only a fixed band around the boundary.

    This turns the resampling step into an (approx.) information-directed refinement of
    probabilistic selection weights.
    """

    def __init__(self, *args, active_bootstrap_samples=16, active_rounds=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_bootstrap_samples = int(max(4, active_bootstrap_samples))
        self.active_rounds = int(max(0, active_rounds))

    def _estimate_topmu_probs(self, fvals_used):
        lam = int(len(fvals_used))
        if lam <= 1:
            return np.ones(lam, dtype=float)

        mu = int(max(1, min(self.mu, lam)))
        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return np.ones(lam, dtype=float) * (float(mu) / float(lam))

        B = int(self.active_bootstrap_samples)
        counts = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)

        rel_pool = self._last_noise_rel_residuals
        abs_floor = float(max(0.0, self._last_noise_abs_floor))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals_used[i])
                    if rel_pool is None or rel_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        r = float(rel_pool[int(self._bootstrap_rng.randint(0, int(rel_pool.size)))])
                        scale = float(max(abs(base), abs_floor, 1e-12))
                        f_boot[i] = base + r * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            counts[order[:mu]] += 1.0

        return counts / float(B)

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        # Start with the base behavior (banded reevaluation + noise proxy update).
        fvals_used, noise_level, reeval_count = super()._reevaluate_and_aggregate(x_eval, fvals)

        if self.active_rounds <= 0 or self.reeval_extra_per_point <= 0:
            return fvals_used, noise_level, reeval_count

        lam = int(len(fvals_used))
        if lam <= 1:
            return fvals_used, noise_level, reeval_count

        # Per-generation cap on number of points we are allowed to reevaluate.
        max_points = int(math.floor(self.reeval_max_frac * lam))
        max_points = max(0, min(max_points, lam))
        if max_points <= 0:
            return fvals_used, noise_level, reeval_count

        # Remaining budget in *points* for this generation (we use 1 extra eval per chosen point).
        remaining_points = max(0, max_points - int(reeval_count))
        if remaining_points <= 0:
            return fvals_used, noise_level, reeval_count

        for _round in range(int(self.active_rounds)):
            if remaining_points <= 0:
                break
            if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                break

            p = self._estimate_topmu_probs(fvals_used)
            u = p * (1.0 - p)

            # Choose most uncertain candidates, avoiding those already heavily reevaluated.
            samples = self._last_point_samples
            if samples is None:
                break
            sample_counts = np.array([int(getattr(s, "size", 0)) for s in samples], dtype=int)
            # Penalize points with more samples to spread refinement.
            scores = u / (1.0 + sample_counts.astype(float))
            order = np.argsort(-scores)

            chosen = []
            for idx in order.tolist():
                if len(chosen) >= remaining_points:
                    break
                if int(sample_counts[idx]) <= 0:
                    continue
                chosen.append(int(idx))

            if not chosen:
                break

            for idx in chosen:
                if remaining_points <= 0:
                    break
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    break
                val = float(self.problem(x_eval[idx]))
                arr = np.asarray(self._last_point_samples[idx], dtype=float)
                arr = np.concatenate([arr, np.asarray([val], dtype=float)])
                self._last_point_samples[idx] = arr
                fvals_used[idx] = float(np.median(arr))
                remaining_points -= 1
                reeval_count += 1

            # Update the pooled noise model from the newly added samples (cheap).
            # This intentionally mirrors the base implementation.
            rel_residuals = []
            abs_residual_mags = []
            for s in self._last_point_samples:
                if s.size <= 1:
                    continue
                m = float(np.median(s))
                denom = float(abs(m) + 1e-12)
                rel_residuals.extend(((s - m) / denom).tolist())
                abs_residual_mags.extend(np.abs(s - m).tolist())
            if rel_residuals:
                self._last_noise_rel_residuals = np.asarray(rel_residuals, dtype=float)
                self._last_noise_abs_floor = float(np.median(np.asarray(abs_residual_mags, dtype=float)))

        return fvals_used, noise_level, reeval_count


class BerwES2NoiseAdaptiveSelBootstrapWeightsHetero(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    BootstrapWeights with a lightweight heteroscedastic noise model:

        |noise| scale ≈ s0 + s1 * |f|

    The model is fit from the (few) reevaluated points each generation and then used
    to synthesize uncertainty for single-evaluated candidates in the bootstrap.

    Motivation:
    - Official bbob-noisy mixes noise types (additive, multiplicative, state-dependent).
    - A pure "relative residual" pool can be miscalibrated when an additive component exists.
    """

    def __init__(
        self,
        *args,
        noise_model_ema=0.25,
        noise_pool_max=512,
        z_clip=10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_model_ema = float(np.clip(float(noise_model_ema), 0.0, 1.0))
        self.noise_pool_max = int(max(64, noise_pool_max))
        self.z_clip = float(max(1.0, z_clip))

        self._noise_s0 = 0.0
        self._noise_s1 = 0.0
        self._noise_z_pool = np.array([], dtype=float)
        self._noise_diag_z_clip_frac = float("nan")
        self._noise_diag_shape_ks = float("nan")
        self._noise_diag_shape_w1 = float("nan")
        self._noise_diag_drift_ks = float("nan")
        self._noise_diag_drift_w1 = float("nan")
        self._noise_diag_scale_fit_r2 = float("nan")
        self._noise_diag_scale_pred_cv = float("nan")
        self._noise_diag_center_split_rel = float("nan")
        self._noise_diag_center_split_cv = float("nan")
        self._noise_prev_gen_z = None

    @staticmethod
    def _ks_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size <= 0 or b.size <= 0:
            return float("nan")
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        grid = np.sort(np.concatenate([a_sorted, b_sorted]))
        cdf_a = np.searchsorted(a_sorted, grid, side="right") / float(a_sorted.size)
        cdf_b = np.searchsorted(b_sorted, grid, side="right") / float(b_sorted.size)
        return float(np.max(np.abs(cdf_a - cdf_b)))

    @staticmethod
    def _w1_distance_1d(a: np.ndarray, b: np.ndarray, *, grid_size: int = 128) -> float:
        """
        1D Wasserstein-1 distance between two empirical samples (quantile form).

        We use a fixed quantile grid for a cheap, stable diagnostic; exact OT is unnecessary here.
        """

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size <= 0 or b.size <= 0:
            return float("nan")
        k = int(max(16, min(int(grid_size), max(int(a.size), int(b.size)))))
        u = (np.arange(k, dtype=float) + 0.5) / float(k)
        qa = np.quantile(a, u)
        qb = np.quantile(b, u)
        return float(np.mean(np.abs(qa - qb)))

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        fvals_used, noise_level, reeval_count = super()._reevaluate_and_aggregate(x_eval, fvals)

        samples = self._last_point_samples
        if samples is None:
            return fvals_used, noise_level, reeval_count

        xs = []
        rs = []
        center_split_pairs = []
        for arr in samples:
            if getattr(arr, "size", 0) <= 1:
                continue
            arr = np.asarray(arr, dtype=float)
            m = float(np.median(arr))
            x = float(abs(m))
            r = arr - m
            xs.extend([x] * int(r.size))
            rs.extend(r.tolist())
            # Split-median diagnostic for centering error (scale-normalized).
            left = arr[::2]
            right = arr[1::2]
            if left.size > 0 and right.size > 0:
                dm = float(abs(np.median(left) - np.median(right)))
                center_split_pairs.append((dm, x))

        if len(rs) < 6:
            return fvals_used, noise_level, reeval_count

        x = np.asarray(xs, dtype=float)
        r = np.asarray(rs, dtype=float)
        y = np.abs(r)

        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        s0_new = float(max(0.0, coef[0]))
        s1_new = float(max(0.0, coef[1]))

        # Scale-fit diagnostics (R^2 on |r| vs (s0+s1|f|)).
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        y_mean = float(np.mean(y))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        if ss_tot > 1e-12:
            self._noise_diag_scale_fit_r2 = float(1.0 - (ss_res / ss_tot))
        else:
            self._noise_diag_scale_fit_r2 = float("nan")

        a = float(self.noise_model_ema)
        self._noise_s0 = (1.0 - a) * float(self._noise_s0) + a * s0_new
        self._noise_s1 = (1.0 - a) * float(self._noise_s1) + a * s1_new

        scale = self._noise_s0 + self._noise_s1 * x
        scale = np.maximum(scale, 1e-12)
        mean_scale = float(np.mean(scale)) if scale.size > 0 else float("nan")
        std_scale = float(np.std(scale)) if scale.size > 0 else float("nan")
        self._noise_diag_scale_pred_cv = float(std_scale / max(1e-12, mean_scale)) if np.isfinite(mean_scale) else float("nan")
        z_raw = r / scale
        self._noise_diag_z_clip_frac = float(np.mean(np.abs(z_raw) > float(self.z_clip))) if z_raw.size > 0 else float("nan")
        z = np.clip(z_raw, -self.z_clip, self.z_clip)

        # Shape-shift diagnostic: KS distance between low-|f| and high-|f| buckets.
        if z.size > 0:
            x_med = float(np.median(x))
            low = z[x <= x_med]
            high = z[x > x_med]
            self._noise_diag_shape_ks = self._ks_distance_1d(low, high)
            self._noise_diag_shape_w1 = self._w1_distance_1d(low, high)
        else:
            self._noise_diag_shape_ks = float("nan")
            self._noise_diag_shape_w1 = float("nan")

        # Drift diagnostic: KS distance vs previous generation's z samples.
        prev = self._noise_prev_gen_z
        if prev is not None:
            self._noise_diag_drift_ks = self._ks_distance_1d(np.asarray(prev, dtype=float), z)
            self._noise_diag_drift_w1 = self._w1_distance_1d(np.asarray(prev, dtype=float), z)
        else:
            self._noise_diag_drift_ks = float("nan")
            self._noise_diag_drift_w1 = float("nan")
        # Keep a small reservoir to avoid unbounded memory.
        if z.size > 256:
            idx = self._bootstrap_rng.choice(z.size, size=256, replace=False)
            self._noise_prev_gen_z = z[idx]
        else:
            self._noise_prev_gen_z = z.astype(float, copy=True)

        if center_split_pairs:
            rels = []
            for dm, x_i in center_split_pairs:
                scale_pred = float(max(1e-12, float(self._noise_s0) + float(self._noise_s1) * float(x_i)))
                rels.append(float(dm) / scale_pred)
            rel_arr = np.asarray(rels, dtype=float)
            rel_arr = rel_arr[np.isfinite(rel_arr)]
            self._noise_diag_center_split_rel = float(np.median(rel_arr)) if rel_arr.size > 0 else float("nan")
            mean_rel = float(np.mean(rel_arr)) if rel_arr.size > 0 else float("nan")
            std_rel = float(np.std(rel_arr)) if rel_arr.size > 0 else float("nan")
            self._noise_diag_center_split_cv = float(std_rel / max(1e-12, mean_rel)) if np.isfinite(mean_rel) else float("nan")
        else:
            self._noise_diag_center_split_rel = float("nan")
            self._noise_diag_center_split_cv = float("nan")

        if z.size > 0:
            pool = self._noise_z_pool
            if pool.size <= 0:
                pool = z.astype(float, copy=True)
            else:
                pool = np.concatenate([pool, z.astype(float, copy=False)])
            if pool.size > self.noise_pool_max:
                idx = self._bootstrap_rng.choice(pool.size, size=self.noise_pool_max, replace=False)
                pool = pool[idx]
            self._noise_z_pool = pool

        return fvals_used, noise_level, reeval_count

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_avg = acc / float(B)
        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroLogW(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    BERW-Hetero with standard CMA-ES logarithmic recombination weights.

    Identical to BerwES2NoiseAdaptiveSelBootstrapWeightsHetero except that each
    bootstrap replicate uses CMA-ES log weights

        w_k = [log(mu + 0.5) - log(rank + 1)]_+   (normalized to sum 1)

    instead of the power-lift weights  w = (1 - rank/(lam-1))^n_eff.

    Everything else -- heteroscedastic noise model, boundary reevaluation,
    residual pool, diagnostics -- is inherited unchanged.

    Purpose: ablation showing the bootstrap-expected-weight mechanism is not
    tied to a specific weight function shape.
    """

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return self._deterministic_log_weights(fvals, mu)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        log_mu_half = np.log(float(mu) + 0.5)

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = log_mu_half - np.log(ranks[mask].astype(float) + 1.0)
                w = np.maximum(w, 0.0)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_avg = acc / float(B)
        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)

    @staticmethod
    def _deterministic_log_weights(fvals, mu):
        """Deterministic CMA log weights (fallback when bootstrap data unavailable)."""
        lam = int(len(fvals))
        order = np.argsort(fvals)
        if lam <= 1:
            return np.ones(lam, dtype=float), order

        mu = int(max(1, min(int(mu), lam)))
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w = np.zeros(lam, dtype=float)
        mask = ranks < mu
        if np.any(mask):
            w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
            w = np.maximum(w, 0.0)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroMisrankingGate(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    Heteroscedastic BootstrapWeights with an *online misranking gate*.

    Key idea:
    - Use the bootstrap resampling itself to estimate how stable the top-μ set is.
    - If the estimated top-μ set is stable (low misranking uncertainty), fall back to
      deterministic CMA log-weights (removes unnecessary smoothing on near-deterministic regimes).
    - If unstable, use probabilistic bootstrap-expected weights.

    Stability proxy:
        overlap = E[|Topμ(ω1) ∩ Topμ(ω2)|] / μ  ≈  (sum_i p_i^2) / μ
    where p_i is the bootstrap-estimated top-μ membership probability.

    We normalize overlap into a [0,1] "stability" score by subtracting the random baseline μ/λ.
    """

    def __init__(
        self,
        *args,
        gate_open_stability=0.85,
        gate_close_stability=0.95,
        gate_ema=0.25,
        gate_patience=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gate_open_stability = float(np.clip(float(gate_open_stability), 0.0, 1.0))
        self.gate_close_stability = float(np.clip(float(gate_close_stability), 0.0, 1.0))
        if self.gate_close_stability < self.gate_open_stability:
            self.gate_close_stability = self.gate_open_stability
        self.gate_ema = float(np.clip(float(gate_ema), 0.0, 1.0))
        self.gate_patience = int(max(1, gate_patience))

        self.gate_closed = True
        self._gate_high_stability_count = 0
        self.gate_stability_ema = 1.0

    def _mean_update_mode(self):
        if self.gate_closed:
            return "average"
        return self.mean_update

    @staticmethod
    def _deterministic_cma_log_weights(fvals, mu):
        lam = int(len(fvals))
        order = np.argsort(fvals)
        if lam <= 1:
            return np.ones(lam, dtype=float), order

        mu = int(max(1, min(int(mu), lam)))
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w = np.zeros(lam, dtype=float)
        mask = ranks < mu
        if np.any(mask):
            w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
            w = np.maximum(w, 0.0)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        counts = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                counts[mask] += 1.0
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_boot = acc / float(B)
        total = float(np.sum(w_boot))
        if not np.isfinite(total) or total <= 0.0:
            w_boot = np.ones(lam, dtype=float) / float(lam)
        else:
            w_boot = w_boot / total

        p = counts / float(B)
        overlap = float(np.sum(p * p) / float(mu)) if mu > 0 else 1.0
        overlap_min = float(mu) / float(lam)
        denom = float(max(1e-12, 1.0 - overlap_min))
        stability = float((overlap - overlap_min) / denom)
        stability = float(np.clip(stability, 0.0, 1.0))

        a = float(self.gate_ema)
        self.gate_stability_ema = (1.0 - a) * float(self.gate_stability_ema) + a * stability

        if self.gate_closed:
            if self.gate_stability_ema < self.gate_open_stability:
                self.gate_closed = False
                self._gate_high_stability_count = 0
            # Closed: deterministic weights.
            return self._deterministic_cma_log_weights(fvals, mu)

        if self.gate_stability_ema > self.gate_close_stability:
            self._gate_high_stability_count += 1
        else:
            self._gate_high_stability_count = 0

        if self._gate_high_stability_count >= self.gate_patience:
            self.gate_closed = True
            self._gate_high_stability_count = 0
            return self._deterministic_cma_log_weights(fvals, mu)

        return w_boot, np.argsort(-w_boot)


class BerwES2NoiseAdaptiveSelBootstrapCMAWeightsHeteroMix(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    Heteroscedastic BootstrapWeights using CMA-ES log-weights, with a continuous
    misranking-aware blend between deterministic and bootstrap-expected weights.

    Goal:
    - Avoid the "two-algorithms stitched together" look of a hard switch.
    - Retain near-CMA behavior when ranks are stable, and smoothly transition to
      probabilistic elite membership when ranks are noisy/misranked.
    """

    def __init__(
        self,
        *args,
        alpha_ema=0.25,
        alpha_mis_lo=0.05,
        alpha_mis_hi=0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_ema = float(np.clip(float(alpha_ema), 0.0, 1.0))
        self.alpha_mis_lo = float(max(0.0, alpha_mis_lo))
        self.alpha_mis_hi = float(max(self.alpha_mis_lo + 1e-12, alpha_mis_hi))

        self.mis_ema = 0.0
        self.mix_alpha = 0.0
        self.mix_stability_ema = 1.0

    @staticmethod
    def _deterministic_cma_log_weights(fvals, mu):
        lam = int(len(fvals))
        order = np.argsort(fvals)
        if lam <= 1:
            return np.ones(lam, dtype=float), order

        mu = int(max(1, min(int(mu), lam)))
        ranks = np.empty(lam, dtype=int)
        ranks[order] = np.arange(lam)

        w = np.zeros(lam, dtype=float)
        mask = ranks < mu
        if np.any(mask):
            w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
            w = np.maximum(w, 0.0)

        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total
        return w, order

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return self._deterministic_cma_log_weights(fvals, mu)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        counts = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                counts[mask] += 1.0
                w[mask] = np.log(float(mu) + 0.5) - np.log(ranks[mask].astype(float) + 1.0)
                w = np.maximum(w, 0.0)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_boot = acc / float(B)
        total = float(np.sum(w_boot))
        if not np.isfinite(total) or total <= 0.0:
            w_boot = np.ones(lam, dtype=float) / float(lam)
        else:
            w_boot = w_boot / total

        p = counts / float(B)
        overlap = float(np.sum(p * p) / float(mu)) if mu > 0 else 1.0
        overlap_min = float(mu) / float(lam)
        denom = float(max(1e-12, 1.0 - overlap_min))
        stability = float((overlap - overlap_min) / denom)
        stability = float(np.clip(stability, 0.0, 1.0))

        mis = float(np.clip(1.0 - stability, 0.0, 1.0))
        a = float(self.alpha_ema)
        self.mis_ema = (1.0 - a) * float(self.mis_ema) + a * mis
        self.mix_stability_ema = (1.0 - a) * float(self.mix_stability_ema) + a * stability

        alpha = (float(self.mis_ema) - float(self.alpha_mis_lo)) / float(self.alpha_mis_hi - self.alpha_mis_lo)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.mix_alpha = alpha

        w_det, _ = self._deterministic_cma_log_weights(fvals, mu)
        w = (1.0 - alpha) * w_det + alpha * w_boot
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            w = np.ones(lam, dtype=float) / float(lam)
        else:
            w = w / total

        return w, np.argsort(-w)

    def run(self):
        if self.max_evals <= 0:
            return

        while self.problem.evaluations < self.max_evals and not self.problem.final_target_hit:
            remaining = int(self.max_evals - self.problem.evaluations)
            if remaining <= 0:
                break

            lam = min(self.lambda_, remaining)
            z = self.rng.randn(lam, self.dim)
            y = z * np.sqrt(self.C)[None, :]
            x = self.mean[None, :] + self.sigma * y
            x_eval = np.clip(x, self.lower, self.upper)

            fvals = np.empty(lam, dtype=float)
            for i in range(lam):
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    fvals = fvals[:i]
                    z = z[:i]
                    y = y[:i]
                    x_eval = x_eval[:i]
                    break
                fvals[i] = float(self.problem(x_eval[i]))

            if len(fvals) == 0:
                break

            self._last_lam = int(len(fvals))
            self._last_mu = int(max(1, min(self.mu, self._last_lam)))

            fvals_used, noise_level, reeval_count = self._reevaluate_and_aggregate(x_eval, fvals)

            idx_best = int(np.argmin(fvals_used))
            if fvals_used[idx_best] < self.best_f:
                self.best_f = float(fvals_used[idx_best])
                self.best_x = x_eval[idx_best].copy()
                self._no_improve_gens = 0
            else:
                self._no_improve_gens += 1

            weights_all, _ = self._compute_weights(fvals_used)
            w = np.asarray(weights_all, dtype=float)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                w = np.ones(len(fvals_used), dtype=float) / float(len(fvals_used))
            else:
                w = w / w_sum

            mueff = 1.0 / float(np.sum(w * w))
            self._update_strategy_params(mueff=mueff)

            mean_old = self.mean.copy()
            sigma_old = float(self.sigma)

            alpha = float(np.clip(float(getattr(self, "mix_alpha", 0.0)), 0.0, 1.0))
            y_w = (w[:, None] * y).sum(axis=0)
            mean_avg = mean_old + self.c_mean * sigma_old * y_w

            mean_target = self._weiszfeld_update(x_eval, w, mean_old)
            mean_weisz = (1.0 - self.c_mean) * mean_old + self.c_mean * mean_target

            mean_proposed = (1.0 - alpha) * mean_avg + alpha * mean_weisz

            self.mean = np.clip(mean_proposed, self.lower, self.upper)
            y_w = (self.mean - mean_old) / max(sigma_old, 1e-30)

            invsqrtC_y_w = y_w / np.sqrt(np.maximum(self.C, 1e-30))
            self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y_w
            norm_ps = float(np.linalg.norm(self.ps))
            self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
            self.sigma = float(np.clip(self.sigma, self.min_sigma, self.max_sigma))

            hsig_cond = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chiN
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (self.dim + 1.0)) else 0.0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            rank_mu = np.sum(w[:, None] * (y**2), axis=0)
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * rank_mu
            )
            self.C = np.maximum(self.C, 1e-30)

            self._maybe_record_state(noise_level=noise_level, reeval_count=reeval_count)

            self.generation += 1

            if self._no_improve_gens >= self.restart_patience:
                self._restart()


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroActiveCov(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    Hetero BootstrapWeights with a misranking-aware *active* diagonal covariance update.

    Motivation:
    - sep-CMA's active covariance (negative weights) is often helpful in low-noise regimes,
      but can be harmful when ranking is unreliable.
    - We estimate ranking stability from the bootstrap itself, and attenuate the negative
      update proportionally to stability.
    """

    def __init__(self, *args, active_ema=0.25, active_power=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_ema = float(np.clip(float(active_ema), 0.0, 1.0))
        self.active_power = float(max(0.0, active_power))
        self.active_stability_ema = 1.0

    @staticmethod
    def _active_rank_weights(lam: int, mu: int) -> np.ndarray:
        """
        Return a CMA-style active weight vector (per-rank) with:
        - positive log-weights for ranks < mu, normalized to sum +1,
        - negative log-weights for ranks >= mu, normalized to sum -1,
        hence sum(weights) == 0.
        """

        lam = int(max(1, lam))
        mu = int(max(1, min(mu, lam)))
        idx = np.arange(1, lam + 1, dtype=float)
        raw = np.log(float(mu) + 0.5) - np.log(idx)
        w_pos = np.maximum(raw[:mu], 0.0)
        s_pos = float(np.sum(w_pos))
        if not np.isfinite(s_pos) or s_pos <= 0.0:
            w_pos = np.ones(mu, dtype=float) / float(mu)
        else:
            w_pos = w_pos / s_pos

        w_neg_raw = np.minimum(raw[mu:], 0.0)
        s_neg = float(np.sum(w_neg_raw))
        if not np.isfinite(s_neg) or abs(s_neg) <= 1e-12:
            w_neg = np.zeros(lam - mu, dtype=float)
        else:
            w_neg = w_neg_raw / abs(s_neg)  # sums to -1
        return np.concatenate([w_pos, w_neg])

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            self.active_stability_ema = 1.0
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            # Fallback: deterministic weights + assume stable.
            self.active_stability_ema = 1.0
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        counts = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                counts[mask] += 1.0
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_boot = acc / float(B)
        total = float(np.sum(w_boot))
        if not np.isfinite(total) or total <= 0.0:
            w_boot = np.ones(lam, dtype=float) / float(lam)
        else:
            w_boot = w_boot / total

        # Bootstrap stability proxy -> EMA.
        p = counts / float(B)
        overlap = float(np.sum(p * p) / float(mu)) if mu > 0 else 1.0
        overlap_min = float(mu) / float(lam)
        denom = float(max(1e-12, 1.0 - overlap_min))
        stability = float((overlap - overlap_min) / denom)
        stability = float(np.clip(stability, 0.0, 1.0))
        a = float(self.active_ema)
        self.active_stability_ema = (1.0 - a) * float(self.active_stability_ema) + a * stability

        return w_boot, np.argsort(-w_boot)

    def run(self):
        if self.max_evals <= 0:
            return

        while self.problem.evaluations < self.max_evals and not self.problem.final_target_hit:
            remaining = int(self.max_evals - self.problem.evaluations)
            if remaining <= 0:
                break

            lam = min(self.lambda_, remaining)
            z = self.rng.randn(lam, self.dim)
            y = z * np.sqrt(self.C)[None, :]
            x = self.mean[None, :] + self.sigma * y
            x_eval = np.clip(x, self.lower, self.upper)

            fvals = np.empty(lam, dtype=float)
            for i in range(lam):
                if self.problem.evaluations >= self.max_evals or self.problem.final_target_hit:
                    fvals = fvals[:i]
                    y = y[:i]
                    x_eval = x_eval[:i]
                    break
                fvals[i] = float(self.problem(x_eval[i]))

            if len(fvals) == 0:
                break

            self._last_lam = int(len(fvals))
            self._last_mu = int(max(1, min(self.mu, self._last_lam)))

            fvals_used, noise_level, reeval_count = self._reevaluate_and_aggregate(x_eval, fvals)

            idx_best = int(np.argmin(fvals_used))
            if fvals_used[idx_best] < self.best_f:
                self.best_f = float(fvals_used[idx_best])
                self.best_x = x_eval[idx_best].copy()
                self._no_improve_gens = 0
            else:
                self._no_improve_gens += 1

            weights_all, _ = self._compute_weights(fvals_used)
            w = np.asarray(weights_all, dtype=float)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0.0:
                w = np.ones(len(fvals_used), dtype=float) / float(len(fvals_used))
            else:
                w = w / w_sum

            mueff = 1.0 / float(np.sum(w * w))
            self._update_strategy_params(mueff=mueff)

            mean_old = self.mean.copy()
            sigma_old = float(self.sigma)

            # Misranking-aware blend of mean update: stable -> average, unstable -> weiszfeld.
            alpha = float(np.clip(1.0 - float(self.active_stability_ema), 0.0, 1.0))
            y_w = (w[:, None] * y).sum(axis=0)
            mean_avg = mean_old + self.c_mean * sigma_old * y_w

            mean_target = self._weiszfeld_update(x_eval, w, mean_old)
            mean_weisz = (1.0 - self.c_mean) * mean_old + self.c_mean * mean_target

            mean_proposed = (1.0 - alpha) * mean_avg + alpha * mean_weisz

            self.mean = np.clip(mean_proposed, self.lower, self.upper)
            y_w = (self.mean - mean_old) / max(sigma_old, 1e-30)

            invsqrtC_y_w = y_w / np.sqrt(np.maximum(self.C, 1e-30))
            self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC_y_w
            norm_ps = float(np.linalg.norm(self.ps))
            self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
            self.sigma = float(np.clip(self.sigma, self.min_sigma, self.max_sigma))

            hsig_cond = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / self.chiN
            hsig = 1.0 if hsig_cond < (1.4 + 2 / (self.dim + 1.0)) else 0.0

            self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            # Active diagonal covariance update: attenuate negative weights when unstable.
            order = np.argsort(fvals_used)
            mu_rank = int(max(1, min(self.mu, len(fvals_used))))
            w_active = self._active_rank_weights(len(fvals_used), mu_rank)
            w_neg = np.minimum(w_active, 0.0)
            eta = float(np.clip(float(self.active_stability_ema) ** float(self.active_power), 0.0, 1.0))
            w_cov_sorted = w[order] + eta * w_neg
            sum_w_cov = float(np.sum(w_cov_sorted))

            y2_sorted = y[order] ** 2
            rank_mu = np.sum(w_cov_sorted[:, None] * y2_sorted, axis=0)

            self.C = (
                (1 - self.c1 - self.cmu * sum_w_cov) * self.C
                + self.c1 * (self.pc**2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * rank_mu
            )
            self.C = np.maximum(self.C, 1e-30)

            self._maybe_record_state(noise_level=noise_level, reeval_count=reeval_count)

            self.generation += 1

            if self._no_improve_gens >= self.restart_patience:
                self._restart()


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroTrimmed(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    Heteroscedastic BootstrapWeights with a robust aggregation across bootstrap replicates.

    This combines:
    - the heteroscedastic noise model (s0+s1|f|) used to synthesize uncertainty for
      single-evaluated candidates, and
    - a coordinate-wise trimmed mean across bootstrap weight vectors to reduce sensitivity
      to rare extreme rank realizations (common under heavy-tailed noise and small pools).
    """

    def __init__(self, *args, trim_frac=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_frac = float(np.clip(float(trim_frac), 0.0, 0.49))

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        W = np.empty((B, lam), dtype=float)
        for b in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            W[b, :] = w

        trim = float(self.trim_frac)
        k = int(math.floor(trim * float(B)))
        if k <= 0:
            w_avg = np.mean(W, axis=0)
        else:
            W_sorted = np.sort(W, axis=0)
            w_avg = np.mean(W_sorted[k : B - k, :], axis=0)

        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroRobust(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    Heteroscedastic BootstrapWeights with robustification for heavy-tailed / small-pool regimes.

    Two stabilizers are applied:
    1) Winsorize the standardized residual pool `z_pool` by clipping the `winsor_k` largest
       absolute values (reduces the impact of a single extreme residual when pool size is small).
    2) Aggregate bootstrap weight vectors with a coordinate-wise trimmed mean (`trim_frac`).
    """

    def __init__(self, *args, trim_frac=0.2, winsor_k=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_frac = float(np.clip(float(trim_frac), 0.0, 0.49))
        self.winsor_k = int(max(0, winsor_k))

    @staticmethod
    def _winsorize_pool(z_pool: np.ndarray, winsor_k: int) -> np.ndarray:
        z_pool = np.asarray(z_pool, dtype=float)
        n = int(z_pool.size)
        k = int(max(0, winsor_k))
        if k <= 0 or n <= (k + 1):
            return z_pool
        abs_sorted = np.sort(np.abs(z_pool))
        tau = float(abs_sorted[max(0, n - k - 1)])
        if not np.isfinite(tau) or tau <= 0.0:
            return z_pool
        return np.clip(z_pool, -tau, tau)

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._winsorize_pool(self._noise_z_pool, self.winsor_k)
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        W = np.empty((B, lam), dtype=float)
        for b in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            W[b, :] = w

        trim = float(self.trim_frac)
        k = int(math.floor(trim * float(B)))
        if k <= 0:
            w_avg = np.mean(W, axis=0)
        else:
            W_sorted = np.sort(W, axis=0)
            w_avg = np.mean(W_sorted[k : B - k, :], axis=0)

        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroTMatch(BerwES2NoiseAdaptiveSelBootstrapWeightsHetero):
    """
    Heteroscedastic BootstrapWeights with a *parametric* noise draw inside the bootstrap.

    Motivation:
    - In heavy-tailed regimes, the per-generation residual pool can be small.
    - A single extreme residual then receives probability mass ≈ 1/|pool| under the
      empirical bootstrap, which can severely overestimate its frequency and destabilize
      expected-rank weights.

    This variant replaces the empirical draw z ~ pool with z ~ Student-t(df), where df is
    selected by matching a robust tail-shape statistic on |z| (quantile ratio), and the
    scale is matched via the median absolute deviation.
    """

    _tmatch_tables = None  # (dfs, abs_medians, q90_over_med)

    def __init__(self, *args, trim_frac=0.2, tmatch_samples=200000, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_frac = float(np.clip(float(trim_frac), 0.0, 0.49))
        self.tmatch_samples = int(max(20000, tmatch_samples))

    @classmethod
    def _get_tmatch_tables(cls, *, n):
        if cls._tmatch_tables is not None:
            return cls._tmatch_tables

        rng = np.random.RandomState(1234567)
        dfs = np.asarray([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0, float("inf")], dtype=float)
        abs_meds = np.empty(dfs.size, dtype=float)
        ratios = np.empty(dfs.size, dtype=float)

        for i, df in enumerate(dfs.tolist()):
            if not np.isfinite(df):
                z = rng.randn(int(n)).astype(float, copy=False)
            else:
                df_f = float(df)
                t = rng.standard_t(df_f, size=int(n)).astype(float, copy=False)
                # Rescale to unit variance (df>2 by construction of candidates).
                t = t / math.sqrt(df_f / max(1e-12, (df_f - 2.0)))
                z = t

            a = np.abs(z)
            med = float(np.median(a))
            q90 = float(np.quantile(a, 0.9))
            abs_meds[i] = med
            ratios[i] = q90 / max(1e-12, med)

        cls._tmatch_tables = (dfs, abs_meds, ratios)
        return cls._tmatch_tables

    def _select_tmatch_df_and_scale(self, z_pool):
        z_pool = np.asarray(z_pool, dtype=float)
        if z_pool.size <= 0:
            return float("inf"), 1.0
        a = np.abs(z_pool[np.isfinite(z_pool)])
        if a.size <= 0:
            return float("inf"), 1.0

        med = float(np.median(a))
        if not np.isfinite(med) or med <= 0.0:
            return float("inf"), 1.0
        q90 = float(np.quantile(a, 0.9))
        ratio = float(q90 / max(1e-12, med))

        dfs, abs_meds, ratios = self._get_tmatch_tables(n=self.tmatch_samples)
        idx = int(np.argmin(np.abs(ratios - ratio)))
        df = float(dfs[idx])
        scale = float(med / max(1e-12, float(abs_meds[idx])))
        return df, scale

    def _draw_z(self, df, scale):
        if not np.isfinite(df):
            return float(scale) * float(self._bootstrap_rng.randn())
        df_f = float(df)
        t = float(self._bootstrap_rng.standard_t(df_f))
        t = t / math.sqrt(df_f / max(1e-12, (df_f - 2.0)))
        return float(scale) * t

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        f_boot = np.empty(lam, dtype=float)

        z_pool = self._noise_z_pool
        df, z_scale = self._select_tmatch_df_and_scale(z_pool)
        s0 = float(max(0.0, self._noise_s0))
        s1 = float(max(0.0, self._noise_s1))

        W = np.empty((B, lam), dtype=float)
        for b in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = self._draw_z(df, z_scale)
                        scale = float(max(1e-12, s0 + s1 * abs(base)))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            W[b, :] = w

        trim = float(self.trim_frac)
        k = int(math.floor(trim * float(B)))
        if k <= 0:
            w_avg = np.mean(W, axis=0)
        else:
            W_sorted = np.sort(W, axis=0)
            w_avg = np.mean(W_sorted[k : B - k, :], axis=0)

        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)


class BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroVar(BerwES2NoiseAdaptiveSelBootstrapWeights):
    """
    BootstrapWeights with a lightweight heteroscedastic *variance* model:

        Var(noise) ≈ v0 + v1 * |f|^2

    Compared to the linear |noise| model, this matches the common additive+multiplicative
    decomposition (variances add) and tends to shrink spurious additive components when
    the noise is close to purely multiplicative.
    """

    def __init__(
        self,
        *args,
        noise_model_ema=0.25,
        noise_pool_max=512,
        z_clip=10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_model_ema = float(np.clip(float(noise_model_ema), 0.0, 1.0))
        self.noise_pool_max = int(max(64, noise_pool_max))
        self.z_clip = float(max(1.0, z_clip))

        self._noise_v0 = 0.0
        self._noise_v1 = 0.0
        self._noise_z_pool = np.array([], dtype=float)

    def _reevaluate_and_aggregate(self, x_eval, fvals):
        fvals_used, noise_level, reeval_count = super()._reevaluate_and_aggregate(x_eval, fvals)

        samples = self._last_point_samples
        if samples is None:
            return fvals_used, noise_level, reeval_count

        xs2 = []
        r2s = []
        rs = []
        x_for_r = []
        for arr in samples:
            if getattr(arr, "size", 0) <= 1:
                continue
            arr = np.asarray(arr, dtype=float)
            m = float(np.median(arr))
            x2 = float(abs(m) ** 2)
            r = arr - m
            xs2.extend([x2] * int(r.size))
            r2s.extend((r * r).tolist())
            rs.extend(r.tolist())
            x_for_r.extend([x2] * int(r.size))

        if len(r2s) < 6:
            return fvals_used, noise_level, reeval_count

        x2 = np.asarray(xs2, dtype=float)
        y = np.asarray(r2s, dtype=float)
        X = np.column_stack([np.ones_like(x2), x2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        v0_new = float(max(0.0, coef[0]))
        v1_new = float(max(0.0, coef[1]))

        a = float(self.noise_model_ema)
        self._noise_v0 = (1.0 - a) * float(self._noise_v0) + a * v0_new
        self._noise_v1 = (1.0 - a) * float(self._noise_v1) + a * v1_new

        v0 = float(max(0.0, self._noise_v0))
        v1 = float(max(0.0, self._noise_v1))
        x2_r = np.asarray(x_for_r, dtype=float)
        scale = np.sqrt(np.maximum(v0 + v1 * x2_r, 1e-24))
        r = np.asarray(rs, dtype=float)
        z = r / scale
        z = np.clip(z, -self.z_clip, self.z_clip)

        if z.size > 0:
            pool = self._noise_z_pool
            if pool.size <= 0:
                pool = z.astype(float, copy=True)
            else:
                pool = np.concatenate([pool, z.astype(float, copy=False)])
            if pool.size > self.noise_pool_max:
                idx = self._bootstrap_rng.choice(pool.size, size=self.noise_pool_max, replace=False)
                pool = pool[idx]
            self._noise_z_pool = pool

        return fvals_used, noise_level, reeval_count

    def _compute_weights(self, fvals):
        lam = int(len(fvals))
        if lam <= 1:
            return np.ones(lam, dtype=float), np.arange(lam, dtype=int)

        mu = int(max(1, min(self.mu, lam)))
        n_eff = float(max(0.0, self._effective_n_t()))

        samples = self._last_point_samples
        if samples is None or len(samples) != lam:
            return super()._compute_weights(fvals)

        B = int(max(1, self.bootstrap_samples))
        acc = np.zeros(lam, dtype=float)
        f_boot = np.empty(lam, dtype=float)
        z_pool = self._noise_z_pool
        v0 = float(max(0.0, self._noise_v0))
        v1 = float(max(0.0, self._noise_v1))

        for _ in range(B):
            for i in range(lam):
                s = samples[i]
                if s.size <= 1:
                    base = float(fvals[i])
                    if z_pool.size <= 0:
                        f_boot[i] = base
                    else:
                        z = float(z_pool[int(self._bootstrap_rng.randint(0, int(z_pool.size)))])
                        scale = float(math.sqrt(max(1e-24, v0 + v1 * (abs(base) ** 2))))
                        f_boot[i] = base + z * scale
                else:
                    f_boot[i] = float(s[int(self._bootstrap_rng.randint(0, int(s.size)))])

            order = np.argsort(f_boot)
            ranks = np.empty(lam, dtype=int)
            ranks[order] = np.arange(lam)

            base = 1.0 - (ranks.astype(float) / float(lam - 1))
            base = np.clip(base, 0.0, 1.0)

            w = np.zeros(lam, dtype=float)
            mask = ranks < mu
            if np.any(mask):
                w[mask] = np.power(base[mask], n_eff)

            total = float(np.sum(w))
            if not np.isfinite(total) or total <= 0.0:
                w[:] = 1.0 / float(lam)
            else:
                w = w / total
            acc += w

        w_avg = acc / float(B)
        total = float(np.sum(w_avg))
        if not np.isfinite(total) or total <= 0.0:
            w_avg = np.ones(lam, dtype=float) / float(lam)
        else:
            w_avg = w_avg / total

        return w_avg, np.argsort(-w_avg)


def my_optimizer(problem, max_evals):
    """COCO/BBOB entry point."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
    ) & 0xFFFFFFFF
    BerwES2(problem, max_evals, seed=seed).run()


def my_optimizer_noise(problem, max_evals):
    """Entry point tuned slightly toward noisy objectives."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 12345
    ) & 0xFFFFFFFF
    BerwES2(
        problem,
        max_evals,
        seed=seed,
        mean_update="weiszfeld",
        n_start=1.2,
        n_end=4.0,
        restart_patience=80,
    ).run()


def my_optimizer_noise_adaptive(problem, max_evals):
    """Entry point: noise-adaptive temperature + selective reevaluation."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 54321
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptive(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel(problem, max_evals):
    """Entry point: selection-stability-based noise proxy + boundary reevaluation."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 98765
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSel(problem, max_evals, seed=seed).run()


def my_optimizer_noise_reeval_boundary(problem, max_evals):
    """Entry point: boundary reevaluation + median aggregation (no temperature adaptation)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 192837
    ) & 0xFFFFFFFF
    BerwES2NoiseReevalBoundary(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_gate(problem, max_evals):
    """Entry point: NoiseAdaptiveSel + stability gate (aims to avoid σ≈0 regression)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 246810
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelGate(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_ess(problem, max_evals):
    """Entry point: selection-stability proxy -> target mueff -> solve n_eff (ESS control)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 112233
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelESS(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_gate_ess(problem, max_evals):
    """Entry point: ESS control + stability gate + deterministic-regime fallback."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 223311
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelGateESS(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_ess(problem, max_evals):
    """Entry point: bootstrap stability proxy -> ESS control (mueff targeting)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 445566
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapESS(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_gate_ess(problem, max_evals):
    """Entry point: bootstrap stability proxy + stability gate + ESS control."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 665544
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapGateESS(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights(problem, max_evals):
    """Entry point: bootstrap-expected probabilistic weights (soft elite membership)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778899
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeights(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_trimmed(problem, max_evals):
    """Entry point: BootstrapWeights with trimmed-mean aggregation across bootstrap replicates."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778907
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsTrimmed(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_cma_weights(problem, max_evals):
    """Entry point: bootstrap-expected CMA-ES log weights under noisy ranks."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778906
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapCMAWeights(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_weights_gate(problem, max_evals):
    """Entry point: BootstrapWeights with a stability gate (low-noise fallback)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778901
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsGate(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_weights_blend(problem, max_evals):
    """Entry point: soft blend of CMA log-weights and BootstrapWeights (noise-EMA gated)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778902
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsBlend(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_weights_active(problem, max_evals):
    """Entry point: BootstrapWeights with uncertainty-aware resampling refinement."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778903
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsActive(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero(problem, max_evals):
    """Entry point: HeteroRobust pop=40 (winsorized z-pool + trimmed bootstrap)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778904
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroRobust(problem, max_evals, seed=seed, popsize=40).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_bs16(problem, max_evals):
    """Entry point: BERW-Hetero with bootstrap_samples=16 (ablation)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778904
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHetero(problem, max_evals, seed=seed, bootstrap_samples=16).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_bs64(problem, max_evals):
    """Entry point: BERW-Hetero with bootstrap_samples=64 (ablation)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778904
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHetero(problem, max_evals, seed=seed, bootstrap_samples=64).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_reeval0(problem, max_evals):
    """Entry point: BERW-Hetero with reeval_extra_per_point=0 (ablation: no reevaluation)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778904
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHetero(problem, max_evals, seed=seed, reeval_extra_per_point=0).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_reeval3(problem, max_evals):
    """Entry point: BERW-Hetero with reeval_extra_per_point=3 (ablation: heavier reevaluation)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778904
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHetero(problem, max_evals, seed=seed, reeval_extra_per_point=3).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_misrank_gate(problem, max_evals):
    """Entry point: Hetero BootstrapWeights with misranking-based stability gate."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778911
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroMisrankingGate(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_logw(problem, max_evals):
    """Entry point: BERW-Hetero with CMA-ES log weights in bootstrap (weight-function ablation)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778904
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroLogW(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_cma_weights_hetero_mix(problem, max_evals):
    """Entry point: Hetero bootstrap-expected CMA log weights with misranking-aware blending."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778912
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapCMAWeightsHeteroMix(problem, max_evals, seed=seed, reeval_min=0).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_activecov(problem, max_evals):
    """Entry point: Hetero BootstrapWeights with misranking-aware active covariance update."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778913
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroActiveCov(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_trimmed(problem, max_evals):
    """Entry point: Hetero BootstrapWeights with trimmed-mean aggregation across bootstrap replicates."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778908
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroTrimmed(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_robust(problem, max_evals):
    """Entry point: Hetero BootstrapWeights with winsorized z-pool + trimmed bootstrap aggregation."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778909
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroRobust(problem, max_evals, seed=seed).run()

def my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero_tmatch(problem, max_evals):
    """Entry point: Hetero BootstrapWeights with t-matched parametric bootstrap draws."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778910
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroTMatch(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_bootstrap_weights_heterovar(problem, max_evals):
    """Entry point: BootstrapWeights with heteroscedastic variance model (v0+v1|f|^2)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 778905
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelBootstrapWeightsHeteroVar(problem, max_evals, seed=seed).run()


def my_optimizer_noise_adaptive_sel_ess2(problem, max_evals):
    """Entry point: ESS control with stronger correction (ess_power=2)."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 332211
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelESS(problem, max_evals, seed=seed, ess_power=2.0).run()


def my_optimizer_noise_adaptive_sel_gate_ess2(problem, max_evals):
    """Entry point: ESS control (ess_power=2) + stability gate + deterministic-regime fallback."""
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 113322
    ) & 0xFFFFFFFF
    BerwES2NoiseAdaptiveSelGateESS(problem, max_evals, seed=seed, ess_power=2.0).run()

def my_optimizer_largescale_soft(problem, max_evals):
    """
    Entry point tuned for large-scale / ill-conditioned settings:
    - softer (less elitist) power-lift schedule to keep mueff larger
    - longer restart patience
    """
    seed = (
        int(getattr(problem, "id_function", 0)) * 1000003
        + int(getattr(problem, "id_instance", 0)) * 1009
        + int(getattr(problem, "dimension", getattr(problem, "n_variables", 0))) * 7
        + 424242
    ) & 0xFFFFFFFF
    BerwES2(problem, max_evals, seed=seed, n_start=1.0, n_end=2.5, restart_patience=120).run()
