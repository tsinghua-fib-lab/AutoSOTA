use ndarray::{Array1, ArrayView2, Axis};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
pub mod action;
pub mod fit;
pub mod histogram_binning;
pub mod identification;

#[cfg(feature = "evo-logging")]
pub mod logging_helpers;
pub mod params;
pub mod reducer;
pub mod refinement;
pub mod splitting;
pub mod state;
pub mod two_tensor_solver;
pub use fit::fit;
#[cfg(feature = "evo-logging")]
pub use logging_helpers::{wrap_raw_events_with_context, RawLoggingEvent};
pub use params::{GridTensorParams, GridTensorParamsBuilder};

// Helper type for serializing interval tuples with null handling
type IntervalSerialized = Vec<Vec<Vec<Option<f64>>>>;

// Custom serialization for intervals that handles infinity as null
fn serialize_intervals<S>(
    intervals: &Vec<Vec<(f64, f64)>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let serialized: IntervalSerialized = intervals
        .iter()
        .map(|axis_intervals| {
            axis_intervals
                .iter()
                .map(|(left, right)| {
                    vec![
                        if *left == f64::NEG_INFINITY {
                            None
                        } else {
                            Some(*left)
                        },
                        if *right == f64::INFINITY {
                            None
                        } else {
                            Some(*right)
                        },
                    ]
                })
                .collect()
        })
        .collect();
    serialized.serialize(serializer)
}

// Custom deserialization for intervals that handles null as infinity
fn deserialize_intervals<'de, D>(deserializer: D) -> Result<Vec<Vec<(f64, f64)>>, D::Error>
where
    D: Deserializer<'de>,
{
    let serialized: IntervalSerialized = Deserialize::deserialize(deserializer)?;
    Ok(serialized
        .into_iter()
        .map(|axis_intervals| {
            axis_intervals
                .into_iter()
                .map(|pair| {
                    (
                        pair[0].unwrap_or(f64::NEG_INFINITY),
                        pair[1].unwrap_or(f64::INFINITY),
                    )
                })
                .collect()
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridTensor {
    pub splits: Vec<Vec<f64>>,
    pub observation_counts: Vec<Vec<usize>>,
    #[serde(
        serialize_with = "serialize_intervals",
        deserialize_with = "deserialize_intervals"
    )]
    pub intervals: Vec<Vec<(f64, f64)>>,
    pub scaling: f64,
    /// Two-tensor fields (mandatory for all grids).
    pub backbone_values: Vec<Vec<f64>>,
    pub tilt_values: Vec<Vec<f64>>,
    pub lambda_plus: f64,
    pub lambda_minus: f64,
}

impl GridTensor {
    pub fn new_two_tensor(
        splits: Vec<Vec<f64>>,
        observation_counts: Vec<Vec<usize>>,
        intervals: Vec<Vec<(f64, f64)>>,
        backbone_values: Vec<Vec<f64>>,
        tilt_values: Vec<Vec<f64>>,
        lambda_plus: f64,
        lambda_minus: f64,
    ) -> Self {
        Self {
            splits,
            observation_counts,
            intervals,
            scaling: 1.0,
            backbone_values,
            tilt_values,
            lambda_plus,
            lambda_minus,
        }
    }

    /// Unscaled `(f_plus, f_minus)` over the per-feature values of one sample.
    #[inline]
    fn fold_two_tensor<I: Iterator<Item = f64>>(&self, values: I) -> (f64, f64) {
        // Two-tensor fields are now mandatory, always use two-tensor path
        let mut f_plus = self.lambda_plus;
        let mut f_minus = self.lambda_minus;
        for (i, val) in values.enumerate() {
            let col_idx = self.splits[i].partition_point(|&split| split <= val);
            let b = self.backbone_values[i][col_idx];
            let d = self.tilt_values[i][col_idx];
            f_plus *= b * d.exp();
            f_minus *= b * (-d).exp();
        }
        (f_plus, f_minus)
    }

    /// Optimized prediction for a single sample
    #[inline]
    pub fn predict_single_unscaled(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(
            x.len(),
            self.splits.len(),
            "Input dimension must match tree grid dimension"
        );

        let (f_plus, f_minus) = self.fold_two_tensor(x.iter().copied());
        f_plus - f_minus
    }

    #[inline]
    pub fn predict_single(&self, x: &[f64]) -> f64 {
        // Two-tensor fields are mandatory, so always return unscaled
        // Scaling is applied at StagePredictor level via scaling_plus/scaling_minus
        self.predict_single_unscaled(x)
    }

    #[inline]
    pub fn predict_unscaled(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let n_rows = x.nrows();
        let mut y_hat = Array1::zeros(n_rows);

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let (f_plus, f_minus) = match row.as_slice() {
                Some(slice) => self.fold_two_tensor(slice.iter().copied()),
                None => self.fold_two_tensor(row.iter().copied()),
            };
            y_hat[i] = f_plus - f_minus;
        }
        y_hat
    }

    /// `(backbone_magnitude, tilt_sum)` over the per-feature values of one sample.
    #[inline]
    fn fold_backbone_and_tilt<I: Iterator<Item = f64>>(&self, values: I) -> (f64, f64) {
        let mut pred = 1.0;
        let mut tilt_sum = 0.0;
        for (i, val) in values.enumerate() {
            let col_idx = self.splits[i].partition_point(|&split| split <= val);
            let b = self.backbone_values[i][col_idx];
            let d = self.tilt_values[i][col_idx];
            tilt_sum += d;
            pred *= b;
        }
        (pred, tilt_sum)
    }

    #[inline]
    pub fn predict_single_backbone_and_tilt(&self, x: &[f64]) -> (f64, f64) {
        self.fold_backbone_and_tilt(x.iter().copied())
    }

    #[inline]
    pub fn predict_backbone_and_tilt(&self, x: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
        let n_rows = x.nrows();
        let mut backbone = Array1::zeros(n_rows);
        let mut tilt = Array1::zeros(n_rows);
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let (b, d) = match row.as_slice() {
                Some(slice) => self.fold_backbone_and_tilt(slice.iter().copied()),
                None => self.fold_backbone_and_tilt(row.iter().copied()),
            };
            backbone[i] = b;
            tilt[i] = d;
        }
        (backbone, tilt)
    }

    /// Compute mean factor for all intervals: backbone * cosh(tilt).
    /// This represents the arithmetic mean of the two exponential factors:
    /// mean_factor = (a₊ + a₋) / 2 = b * cosh(d)
    ///
    /// # Returns
    /// Vector of vectors where each element is the mean factor for that interval.
    pub fn get_mean_factor(&self) -> Vec<Vec<f64>> {
        self.backbone_values
            .iter()
            .zip(self.tilt_values.iter())
            .map(|(backbone_col, tilt_col)| {
                backbone_col
                    .iter()
                    .zip(tilt_col.iter())
                    .map(|(b, d)| b * d.cosh())
                    .collect()
            })
            .collect()
    }
}

impl GridTensor {
    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        // Two-tensor fields are mandatory, so always return unscaled
        // Scaling is applied at StagePredictor level via scaling_plus/scaling_minus
        self.predict_unscaled(x)
    }
}
