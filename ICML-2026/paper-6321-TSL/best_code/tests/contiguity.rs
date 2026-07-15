//! `fit` and `predict` produce identical results regardless of the input
//! array's memory layout: C-contiguous, Fortran-ordered, or a sliced view.

use ndarray::{s, Array1, Array2, Axis, ShapeBuilder};
use tsl::forest::{fit_boosted, params::TSLBoostedParamsBuilder, TSL};

const N: usize = 300;

fn make_data() -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((N, 3));
    for i in 0..N {
        x[[i, 0]] = (i as f64) / N as f64;
        x[[i, 1]] = ((i * 7 % N) as f64) / N as f64;
        x[[i, 2]] = ((i * 13 % N) as f64) / N as f64;
    }
    let y = &x.column(0).to_owned() * 2.0 + &x.column(1).to_owned() - 0.5 * &x.column(2).to_owned();
    (x, y)
}

/// Build a genuinely Fortran-ordered (column-major) copy of `x`, same values.
fn to_fortran(x: &Array2<f64>) -> Array2<f64> {
    let (n, p) = (x.nrows(), x.ncols());
    let mut data = Vec::with_capacity(n * p);
    for j in 0..p {
        for i in 0..n {
            data.push(x[[i, j]]);
        }
    }
    Array2::from_shape_vec((n, p).f(), data).unwrap()
}

/// Embed `x`'s columns into a wider array so that `[.., ..;2]` yields a view with
/// non-contiguous rows (row stride 2) carrying exactly `x`'s values.
fn interleaved(x: &Array2<f64>) -> Array2<f64> {
    let (n, p) = (x.nrows(), x.ncols());
    let mut wide = Array2::zeros((n, p * 2));
    for i in 0..n {
        for j in 0..p {
            wide[[i, 2 * j]] = x[[i, j]];
            wide[[i, 2 * j + 1]] = f64::NAN; // junk in the dropped columns
        }
    }
    wide
}

fn fit(x: ndarray::ArrayView2<f64>, y: ndarray::ArrayView1<f64>) -> TSL {
    let params = TSLBoostedParamsBuilder::new()
        .epochs(3)
        .n_iter(25)
        .n_trees(4)
        .seed(7)
        .build();
    fit_boosted(x, y, &params).1
}

fn assert_close(a: &Array1<f64>, b: &Array1<f64>, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-10,
            "{ctx}: prediction {i} differs: {x} vs {y}"
        );
    }
}

#[test]
fn predict_is_layout_invariant() {
    let (x_c, y) = make_data();
    let model = fit(x_c.view(), y.view());
    let baseline = model.predict(x_c.view());

    // Fortran-ordered input
    let x_f = to_fortran(&x_c);
    assert!(!x_f.is_standard_layout(), "x_f should be Fortran-ordered");
    assert!(
        x_f.index_axis(Axis(0), 0).as_slice().is_none(),
        "F-order rows must be non-contiguous for this test to be meaningful"
    );
    assert_close(&model.predict(x_f.view()), &baseline, "predict on F-order");

    // Column-sliced view (non-contiguous rows)
    let wide = interleaved(&x_c);
    let sliced = wide.slice(s![.., ..;2]);
    assert!(
        sliced.index_axis(Axis(0), 0).as_slice().is_none(),
        "column-sliced rows must be non-contiguous"
    );
    assert_close(&model.predict(sliced), &baseline, "predict on column-sliced");
}

#[test]
fn fit_is_layout_invariant() {
    let (x_c, y) = make_data();
    let baseline = fit(x_c.view(), y.view()).predict(x_c.view());

    // Fortran-ordered input matches C-order.
    let x_f = to_fortran(&x_c);
    let model_f = fit(x_f.view(), y.view());
    assert_close(
        &model_f.predict(x_c.view()),
        &baseline,
        "fit on F-order then predict",
    );

    // Column-sliced view matches C-order.
    let wide = interleaved(&x_c);
    let sliced = wide.slice(s![.., ..;2]);
    let model_s = fit(sliced, y.view());
    assert_close(
        &model_s.predict(x_c.view()),
        &baseline,
        "fit on column-sliced then predict",
    );
}
