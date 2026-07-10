use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use std::sync::atomic::{AtomicU64, Ordering};
use indicatif::{ProgressBar, ProgressStyle};
use std::{thread, time};
use crate::index::phase_3a::phase3a;
use crate::index::phase_3b::phase3b;



// =====================================================================================================================
// Phase III: Construct Additional Information
// 
// Goal:
//   - calculate frequency
//   - calculate 2-gram & 3-gram tables to complete metadata.bin
// =====================================================================================================================
pub fn construct_sa_phase3(
	token     : ArrayView1<u32>,
	freq      : ArrayViewMut1<u64>,
	mut pair  : ArrayViewMut1<u64>,
	mut trio  : ArrayViewMut1<u64>,
	num_tokens: usize,
	pair_thres: usize,
	trio_thres: usize,
) {
	
	log::info!("Phase 3 begins.. (~90% >> 100%)");
	let pb_chunk = (num_tokens / 10).min(10_000_000).max(1);
	let pb = ProgressBar::new(((num_tokens + pb_chunk - 1) / pb_chunk) as u64 * 2);
	pb.set_style(ProgressStyle::with_template(
		"{spinner:.green} {bar:64.red/red} {pos}/{len} ETA {eta}").unwrap());

	// 1. preparation
	let token_slice = token.as_slice().expect("token must be contiguous");
	let pair_len = pair.len();
	let trio_len = trio.len();
	let pair_mem: Vec<AtomicU64> = (0..pair_len).map(|_| AtomicU64::new(0)).collect();
	let trio_mem: Vec<AtomicU64> = (0..trio_len).map(|_| AtomicU64::new(0)).collect();
	let pair_atomic: &[AtomicU64] = &pair_mem;
	let trio_atomic: &[AtomicU64] = &trio_mem;

	// 2. get pair/trio
	phase3a(token_slice, pair_atomic, trio_atomic, num_tokens, pb_chunk, pair_thres, trio_thres, &pb);
	let dst_pair: &mut [u64] = pair.as_slice_mut().expect("pair must be contiguous");
	let dst_trio: &mut [u64] = trio.as_slice_mut().expect("trio must be contiguous");
	for (dst, a) in dst_pair.iter_mut().zip(pair_mem.iter()) {
		*dst = a.load(Ordering::Relaxed);
	}
	for (dst, a) in dst_trio.iter_mut().zip(trio_mem.iter()) {
		*dst = a.load(Ordering::Relaxed);
	}

	// 3. get freq
	phase3b(token, freq, num_tokens, pb_chunk, &pb);

	// [4] finish
	thread::sleep(time::Duration::from_millis(50));
	pb.inc(pb.length().unwrap() - pb.position());
	thread::sleep(time::Duration::from_millis(50));
	log::info!("Phase 3 finished (Progress: 100%)");
}