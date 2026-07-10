use numpy::ndarray::{ArrayView1};
use rand::Rng;
use rayon::prelude::*;
use std::time::{Instant};
use std::{thread, time};
use indicatif::{ProgressBar, ProgressStyle};
use crate::helper::compress_build;
use crate::helper::format_eta;


// =====================================================================================================================
// Phase I: Random Sampling
// 
// Goal:
//   - random sample "length-12 consecutive tokens" for sorting.
//   - get offsets for parallelization.
// =====================================================================================================================
pub fn construct_sa_phase1(
	token     : ArrayView1<u32>,
	num_tokens: usize,
	chunk_size: usize,
) -> (Vec<[u64; 5]>, Vec<usize>) {

	let mut rng = rand::rng();
	log::info!("Starting suffix array construction...");

	// 1. get random placements
	let num_samples: usize = (num_tokens / chunk_size) + 1;
	let mut samples: Vec<[u64; 5]> = [].to_vec();
	{
		let mut samples_tmp: Vec<[u64; 5]> = [].to_vec();
		let mult: usize = 1000;
		for _i in 0..(num_samples * mult) {
			let idx: usize = rng.random_range(0..num_tokens);
			samples_tmp.push(compress_build(&token, idx));
		}
		samples_tmp.par_sort_unstable();
		for i in 0..num_samples {
			samples.push(samples_tmp[(i + 1) * samples_tmp.len() / (num_samples + 1)]);
		}
	}
	log::info!("Phase 1 begins.. (~0% >> ~10%)");

	// 2. progress bar
	let pb_chunk = (num_tokens / 10).min(10_000_000).max(1);
	let pb = ProgressBar::new(((num_tokens + pb_chunk - 1) / pb_chunk) as u64);
	pb.set_style(ProgressStyle::with_template(
		"{spinner:.green} {bar:64.blue/blue} {pos}/{len} ETA {msg}").unwrap());
	let timer = Instant::now();
	let mut offset = vec![0usize; num_samples + 2];

	// 3. preparation
	let n_threads = rayon::current_num_threads();
	let chunk: usize = (num_tokens + n_threads - 1) / n_threads;
	let mut cnt_thread = vec![vec![0usize; num_samples + 2]; n_threads];

	// 4. count indptr by threads
	cnt_thread.par_iter_mut().enumerate().for_each(|(t, cnt_t)| {
		let start = t * chunk;
		let end = ((t + 1) * chunk).min(num_tokens);
		if start >= end {
			return;
		}
		for i in start..end {
			if i % pb_chunk == 0 {
				pb.inc(1);
				let elapsed = timer.elapsed().as_secs_f64();
				let eta = (elapsed / pb.position() as f64) * (pb.length().unwrap() - pb.position()) as f64;
				pb.set_message(format_eta(eta as u64));
			}
			let key = compress_build(&token, i);
			let c = match samples.binary_search(&key) {
				Ok(pos) | Err(pos) => pos,
			};
			cnt_t[c] += 1;
		}
	});

	// 5. get cumulative sum
	for c in 0..num_samples+1 {
		for t in 0..n_threads {
			offset[c + 1] += cnt_thread[t][c];
		}
	}
	for c in 0..num_samples+1 {
		offset[c + 1] += offset[c];
	}

	// 6. return
	thread::sleep(time::Duration::from_millis(50));
	pb.inc(pb.length().unwrap() - pb.position());
	thread::sleep(time::Duration::from_millis(50));
	log::info!("Phase 1 Finished (Progress: ~10%)");
	(samples, offset)
}