use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use rayon::prelude::*;
use indicatif::{ProgressBar};


// =====================================================================================================================
// Phase III-b: Frequency
// 
// Goal:
//   - calculate the frequency of tokens and save to an array freq[]
// =====================================================================================================================
pub fn phase3b(
	token     : ArrayView1<u32>,
	mut freq  : ArrayViewMut1<u64>,
	num_tokens: usize,
	pb_chunk  : usize,
	pb        : &ProgressBar
) {
	
	let n_threads = rayon::current_num_threads();
	let vocab_size = freq.len();
	let chunk = (num_tokens + n_threads - 1) / n_threads;
	let mut cnt_freq = vec![vec![0usize; vocab_size + 1]; n_threads];
	cnt_freq.par_iter_mut().enumerate().for_each(|(t, cnt_t)| {
		let stt = (t * chunk).min(num_tokens);
		let end = ((t + 1) * chunk).min(num_tokens);
		if stt >= end {
			return;
		}
		for i in stt..end {
			if i % pb_chunk == 0 { pb.inc(1); }
			cnt_t[token[i] as usize] += 1;
		}
	});
	for i in 0..n_threads {
		for j in 0..vocab_size {
			freq[j] += cnt_freq[i][j] as u64;
		}
	}
}