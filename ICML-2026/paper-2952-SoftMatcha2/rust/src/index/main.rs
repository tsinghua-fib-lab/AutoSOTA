use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use crate::index::phase_1::construct_sa_phase1;
use crate::index::phase_2::construct_sa_phase2;
use crate::index::phase_3::construct_sa_phase3;


// =====================================================================================================================
// Entry point from python
// =====================================================================================================================
#[pyfunction]
pub fn build_sa_rs<'py>(
	_py: Python<'py>,
	token: PyReadonlyArray1<'py, u32>,
	mut freq  : PyReadwriteArray1<'py, u64>,
	mut pair  : PyReadwriteArray1<'py, u64>,
	mut trio  : PyReadwriteArray1<'py, u64>,
	index_path: String,
	max_tokens: u64,
	num_tokens: u64,
	rough_div : u64,
	pair_thres: u64,
	trio_thres: u64,
	chunk_size: u64,
	w_threads : u64,
	num_shard: u64,
	sa_size   : u64
) -> PyResult<()> {

	// 3-1. preparation
	rayon::ThreadPoolBuilder::new()
        .stack_size(32 * 1024 * 1024)
        .build_global()
        .ok();
	let token_view = token.as_array();
	let freq_view  = freq.as_array_mut();
	let pair_view  = pair.as_array_mut();
	let trio_view  = trio.as_array_mut();

	// 3-2. construct suffix array (phase 1)
	let (samples, offset) = construct_sa_phase1(
		token_view, num_tokens as usize, chunk_size as usize
	);

	// 3-3. construct suffix array (phase 2)
	let _res = construct_sa_phase2(
		token_view, samples, offset, index_path,
		max_tokens as usize, num_tokens as usize,
		rough_div as usize, chunk_size as usize,
		w_threads as usize, num_shard as usize,
		sa_size as usize
	)?;

	// 3-4. construct suffix array (phase 3)
	construct_sa_phase3(
		token_view, freq_view,
		pair_view, trio_view,
		num_tokens as usize, pair_thres as usize,
		trio_thres as usize
	);

	// 3-3. return
	Ok(())
}