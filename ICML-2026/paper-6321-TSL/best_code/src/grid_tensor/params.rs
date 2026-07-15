use serde::Serialize;

use crate::grid_tensor::{refinement::RefinementStrategy, splitting::SplitStrategy};

// ── Refinement ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum RefinementKind {
    L2,
    Huber,
}

/// Parameters for the per-node refinement solver.
///
/// `L2` uses squared-error loss; `Huber` uses a robustified loss with a fixed
/// constant c = 1.345 (≈ 95% efficiency on normal data).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RefinementStrategyParams {
    pub kind: RefinementKind,
    pub alpha: f64,
    /// L2 coupling between u_+ and u_- (objective τ).
    pub tilt_tau: f64,
    /// L1 coupling on (u_+ − u_-) (objective ρ).
    pub tilt_rho: f64,
    /// Prior sample size for parent anchoring (tau_0).
    /// Interpreted as "how many samples worth of confidence in the parent".
    /// Default: 0.0 (no anchoring). Typical values: 10–50.
    pub prior_sample_size: f64,
    pub update_clamp: f64,
}

impl RefinementStrategyParams {
    pub fn get_refinement_strategy(&self) -> RefinementStrategy {
        match self.kind {
            RefinementKind::L2 => RefinementStrategy::L2Refinement {
                alpha: self.alpha,
                tilt_tau: self.tilt_tau,
                tilt_rho: self.tilt_rho,
                prior_sample_size: self.prior_sample_size,
                update_clamp: self.update_clamp,
            },
            RefinementKind::Huber => RefinementStrategy::HuberRefinement {
                alpha: self.alpha,
                c: 1.345,
                tilt_tau: self.tilt_tau,
                tilt_rho: self.tilt_rho,
                prior_sample_size: self.prior_sample_size,
                update_clamp: self.update_clamp,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefinementStrategyParamsBuilder {
    kind: RefinementKind,
    alpha: f64,
    tilt_tau: f64,
    tilt_rho: f64,
    prior_sample_size: f64,
    update_clamp: f64,
}

impl RefinementStrategyParamsBuilder {
    pub fn new() -> Self {
        Self {
            kind: RefinementKind::L2,
            alpha: 0.0,
            tilt_tau: crate::grid_tensor::two_tensor_solver::DEFAULT_TAU,
            tilt_rho: crate::grid_tensor::two_tensor_solver::DEFAULT_RHO,
            prior_sample_size: 0.0,
            update_clamp: f64::INFINITY,
        }
    }

    pub fn l2(mut self) -> Self {
        self.kind = RefinementKind::L2;
        self
    }

    pub fn huber(mut self) -> Self {
        self.kind = RefinementKind::Huber;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// L2 coupling between u_+ and u_- (objective τ).
    pub fn tilt_tau(mut self, tilt_tau: f64) -> Self {
        self.tilt_tau = tilt_tau;
        self
    }

    /// L1 coupling on (u_+ − u_-) (objective ρ).
    pub fn tilt_rho(mut self, tilt_rho: f64) -> Self {
        self.tilt_rho = tilt_rho;
        self
    }

    /// Prior sample size for parent anchoring (tau_0). Default: 0.0 (no anchoring).
    pub fn prior_sample_size(mut self, prior_sample_size: f64) -> Self {
        self.prior_sample_size = prior_sample_size;
        self
    }

    pub fn update_clamp(mut self, update_clamp: f64) -> Self {
        self.update_clamp = update_clamp;
        self
    }

    pub fn build(self) -> RefinementStrategyParams {
        RefinementStrategyParams {
            kind: self.kind,
            alpha: self.alpha,
            tilt_tau: self.tilt_tau,
            tilt_rho: self.tilt_rho,
            prior_sample_size: self.prior_sample_size,
            update_clamp: self.update_clamp,
        }
    }
}

impl Default for RefinementStrategyParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Split strategy ────────────────────────────────────────────────────────────

/// Variants have genuinely different fields, so this stays an enum.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum SplitStrategyParams {
    RandomSplit {
        split_try: usize,
        colsample_bytree: f64,
        min_interval_samples: usize,
        min_split_loss: f64,
        complexity_penalty: f64,
    },
    BestSplit {
        min_interval_samples: usize,
        min_split_loss: f64,
        complexity_penalty: f64,
    },
    TopKSplits {
        top_k: usize,
        must_fill_all_k: bool,
        min_interval_samples: usize,
        min_split_loss: f64,
        complexity_penalty: f64,
    },
}

impl SplitStrategyParams {
    pub fn get_split_strategy(&self) -> SplitStrategy {
        match self {
            SplitStrategyParams::RandomSplit {
                split_try,
                colsample_bytree,
                min_interval_samples,
                min_split_loss,
                complexity_penalty,
            } => SplitStrategy::Random {
                split_try: *split_try,
                colsample_bytree: *colsample_bytree,
                min_interval_samples: *min_interval_samples,
                min_split_loss: *min_split_loss,
                complexity_penalty: *complexity_penalty,
            },
            SplitStrategyParams::BestSplit {
                min_interval_samples,
                min_split_loss,
                complexity_penalty,
            } => SplitStrategy::Best {
                min_interval_samples: *min_interval_samples,
                min_split_loss: *min_split_loss,
                complexity_penalty: *complexity_penalty,
            },
            SplitStrategyParams::TopKSplits {
                top_k,
                must_fill_all_k,
                min_interval_samples,
                min_split_loss,
                complexity_penalty,
            } => SplitStrategy::TopK {
                top_k: *top_k,
                must_fill_all_k: *must_fill_all_k,
                min_interval_samples: *min_interval_samples,
                min_split_loss: *min_split_loss,
                complexity_penalty: *complexity_penalty,
            },
        }
    }
}

/// Flat builder for `SplitStrategyParams`. Call `.random_split()`, `.best_split()`, or
/// `.top_k_splits()` to select the variant; only fields relevant to that variant are used.
#[derive(Debug, Clone)]
pub struct SplitStrategyParamsBuilder {
    variant: SplitVariant,
    split_try: usize,
    colsample_bytree: f64,
    min_interval_samples: usize,
    top_k: usize,
    must_fill_all_k: bool,
    min_split_loss: f64,
    complexity_penalty: f64,
}

#[derive(Debug, Clone)]
enum SplitVariant {
    RandomSplit,
    BestSplit,
    TopKSplits,
}

impl SplitStrategyParamsBuilder {
    pub fn new() -> Self {
        Self {
            variant: SplitVariant::RandomSplit,
            split_try: 10,
            colsample_bytree: 1.0,
            min_interval_samples: 1,
            min_split_loss: 0.0,
            complexity_penalty: 0.0,
            top_k: 5,
            must_fill_all_k: false,
        }
    }

    pub fn random_split(mut self) -> Self {
        self.variant = SplitVariant::RandomSplit;
        self
    }

    pub fn best_split(mut self) -> Self {
        self.variant = SplitVariant::BestSplit;
        self
    }

    pub fn top_k_splits(mut self) -> Self {
        self.variant = SplitVariant::TopKSplits;
        self
    }

    pub fn split_try(mut self, split_try: usize) -> Self {
        self.split_try = split_try;
        self
    }

    pub fn colsample_bytree(mut self, colsample_bytree: f64) -> Self {
        self.colsample_bytree = colsample_bytree;
        self
    }

    pub fn min_interval_samples(mut self, min_interval_samples: usize) -> Self {
        self.min_interval_samples = min_interval_samples;
        self
    }

    pub fn min_split_loss(mut self, min_split_loss: f64) -> Self {
        self.min_split_loss = min_split_loss;
        self
    }

    /// Complexity penalty (λ) for the adaptive merge bonus.
    ///
    /// `bonus = λ · MSE · (log(n)/n + 1/harmonic_mean(n_left, n_right))`
    ///
    /// BIC-inspired and scale-invariant. Larger λ encourages simpler trees.
    /// Default: 0.0. Typical values: 0.5–2.0.
    pub fn complexity_penalty(mut self, lambda: f64) -> Self {
        self.complexity_penalty = lambda;
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn must_fill_all_k(mut self, must_fill_all_k: bool) -> Self {
        self.must_fill_all_k = must_fill_all_k;
        self
    }

    pub fn build(self) -> SplitStrategyParams {
        match self.variant {
            SplitVariant::RandomSplit => SplitStrategyParams::RandomSplit {
                split_try: self.split_try,
                colsample_bytree: self.colsample_bytree,
                min_interval_samples: self.min_interval_samples,
                min_split_loss: self.min_split_loss,
                complexity_penalty: self.complexity_penalty,
            },
            SplitVariant::BestSplit => SplitStrategyParams::BestSplit {
                min_interval_samples: self.min_interval_samples,
                min_split_loss: self.min_split_loss,
                complexity_penalty: self.complexity_penalty,
            },
            SplitVariant::TopKSplits => SplitStrategyParams::TopKSplits {
                top_k: self.top_k,
                must_fill_all_k: self.must_fill_all_k,
                min_interval_samples: self.min_interval_samples,
                min_split_loss: self.min_split_loss,
                complexity_penalty: self.complexity_penalty,
            },
        }
    }
}

impl Default for SplitStrategyParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Grid tensor ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GridTensorParams {
    pub n_iter: usize,
    pub split_strategy_params: SplitStrategyParams,
    pub refinement_strategy_params: RefinementStrategyParams,
    /// Histogram-binning cap for split candidates.
    /// `None` (default): exact — every sorted position is a candidate.
    /// `Some(n)`: only `n` quantile bin boundaries per feature are candidates.
    pub max_bins: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct GridTensorParamsBuilder {
    n_iter: usize,
    split_strategy_params: SplitStrategyParams,
    refinement_strategy_params: RefinementStrategyParams,
    max_bins: Option<u16>,
}

impl GridTensorParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_iter: 25,
            split_strategy_params: SplitStrategyParamsBuilder::new().build(),
            refinement_strategy_params: RefinementStrategyParamsBuilder::new().build(),
            max_bins: None,
        }
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.split_strategy_params = strategy;
        self
    }

    pub fn refinement_strategy(mut self, strategy: RefinementStrategyParams) -> Self {
        self.refinement_strategy_params = strategy;
        self
    }

    pub fn max_bins(mut self, max_bins: Option<u16>) -> Self {
        self.max_bins = max_bins;
        self
    }

    pub fn build(self) -> GridTensorParams {
        GridTensorParams {
            n_iter: self.n_iter,
            split_strategy_params: self.split_strategy_params,
            refinement_strategy_params: self.refinement_strategy_params,
            max_bins: self.max_bins,
        }
    }
}

impl Default for GridTensorParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for GridTensorParams {
    fn default() -> Self {
        GridTensorParamsBuilder::new().build()
    }
}
