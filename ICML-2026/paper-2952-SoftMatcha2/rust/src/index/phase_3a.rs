use rayon::prelude::*;
use indicatif::{ProgressBar};
use std::sync::atomic::{AtomicU64, Ordering};


// =====================================================================================================================
// Phase III-a: Making 2-gram & 3-gram table
// 
// Goal:
//   - completing the 2-gram & 3-gram table of top "pair-cons" & "trio-cons" words
// =====================================================================================================================
pub fn phase3a(
	token_slice: &[u32],
	pair_atomic: &[AtomicU64],
	trio_atomic: &[AtomicU64],
	num_tokens : usize,
	pb_chunk   : usize,
	pair_thres : usize,
	trio_thres : usize,
	pb        : &ProgressBar
) {
	(0..num_tokens).into_par_iter().for_each(|i| {
		if i % pb_chunk == 0 { pb.inc(1); }
		let v0 = token_slice[i]     as i64;
		let v1 = token_slice[i + 1] as i64;
		let v2 = token_slice[i + 2] as i64;

		// add to pair_atomic (2-gram)
		if (v0 as usize) < pair_thres && (v1 as usize) < pair_thres {
			let idx = (v0 as usize) * pair_thres + (v1 as usize);
			let word = idx >> 6;
			let bit  = 1u64 << (idx & 63);
			pair_atomic[word].fetch_or(bit, Ordering::Relaxed);
		}

		// add to trio_atomic (3-gram)
		if (v0 as usize) + (v1 as usize) + (v2 as usize) < trio_thres {
			let t = trio_thres as i64;
			let idx = ((t + 2) * (t + 1) * t) / 6 - ((t + 2 - v0) * (t + 1 - v0) * (t + 0 - v0)) / 6
				+ v1 * (t - v0) - v1 * (v1 - 1) / 2
				+ v2;
			let word = idx >> 6;
			let bit  = 1u64 << (idx & 63);
			trio_atomic[word as usize].fetch_or(bit, Ordering::Relaxed);
		}
	});
}