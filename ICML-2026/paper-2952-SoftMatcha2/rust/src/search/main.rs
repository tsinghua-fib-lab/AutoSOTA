use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{Array2};
use pyo3::Python;
use std::fs::File;
use crate::search::z_core::compute;
use crate::search::z_bsearch::get_match_range;


// ====================================================================================================================
// Entry point which is called from Python
// <enumerate candidates query>
// ====================================================================================================================
#[pyfunction]
pub fn enumerate_candidates_rs<'py>(
	py: Python<'py>,
	pattern: PyReadonlyArray1<'py, u64>,
	score_matrix: PyReadonlyArray2<'py, f32>,
	rough: PyReadonlyArray1<'py, u64>,
	pair_: PyReadonlyArray1<'py, u64>,
	trio_: PyReadonlyArray1<'py, u64>,
	norm : PyReadonlyArray1<'py, f32>,
	idx_fname : String,
	num_tokens: usize,
	idx_offset: usize,
	pair_cons : usize,
	trio_cons : usize,
	rough_div : usize,
	goal      : usize,
	goal_alpha: f32,
	goal_time : f32,
) -> PyResult<(Bound<'py, PyArray2<u64>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<u64>>, f32)> {

	// 1. numpy -> ndarray view conversion
	let pat   = pattern.as_array();
	let score = score_matrix.as_array();
	let rough = rough.as_array();
	let pair_ = pair_.as_array();
	let trio_ = trio_.as_array();
	let norm  = norm.as_array();
	let file = File::open(idx_fname)?;

	// 2. execution
	let (cand, cand_score, count, last_alpha) = compute(
		pat, score, rough, pair_, trio_, norm, num_tokens, idx_offset,
		pair_cons, trio_cons, rough_div,
		goal, goal_alpha, goal_time, &file
	);

	// 3. conversion to output
	let rows = cand.len();
	let max_len = cand.iter().map(|row| row.len()).max().unwrap_or(0);
	let pad_value: u64 = u64::MAX;
	let mut flat = Vec::with_capacity(rows * max_len);
	for row in &cand {
		flat.extend_from_slice(row);
		if row.len() < max_len {
			let diff = max_len - row.len();
			flat.extend(std::iter::repeat(pad_value).take(diff));
		}
	}
	let arr2 = Array2::from_shape_vec((rows, max_len), flat).map_err(|_| {
		pyo3::exceptions::PyValueError::new_err("failed to build Array2 from candidates")
	})?;

	// 4. rust -> numpy
	let py_cand       = PyArray2::from_owned_array(py, arr2);
	let py_cand_score = PyArray1::from_vec(py, cand_score);
	let py_count      = PyArray1::from_vec(py, count);

	// 5. return
	Ok((py_cand, py_cand_score, py_count, last_alpha))
}



// ====================================================================================================================
// Entry point which is called from Python
// <get match count/range query>
// ====================================================================================================================
#[pyfunction]
pub fn get_match_range_rs<'py>(
	_py: Python<'py>,
	pattern  : PyReadonlyArray1<'py, u64>,
	rough    : PyReadonlyArray1<'py, u64>,
	idx_fname: String,
	rough_div: usize,
	num_tokens: usize,
	idx_offset: usize,
) -> PyResult<(u64, u64)> {

	let pat   = pattern.as_array();
	let rough = rough.as_array();
	let mut pat_vec = vec![0u32; pat.len()];
	for i in 0..pat.len() {
		pat_vec[i] = pat[i] as u32;
	}
	let file = File::open(idx_fname)?;
	let (l, r) = get_match_range(rough, rough_div, &pat_vec, &file, num_tokens, idx_offset);
	Ok((l, r))
}