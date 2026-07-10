use numpy::ndarray::{ArrayView1};
use indicatif::{ProgressBar, ProgressStyle};
use std::io;
use std::slice;
use std::{thread, time};
use std::os::unix::fs::FileExt;
use crate::index::memmap::FastMmapVec;
use crate::index::phase_2a::phase2a;
use crate::index::phase_2b::phase2b;


// =====================================================================================================================
// Phase II: Suffix Array Construction
// 
// Goal:
//   - make complete suffix array (including sa.bin, index.bin, and rough.bin)
// =====================================================================================================================
pub fn construct_sa_phase2(
	token     : ArrayView1<u32>,
	samples   : Vec<[u64; 5]>,
	offset    : Vec<usize>,
	input_path: String,
	max_tokens: usize,
	num_tokens: usize,
	rough_div : usize,
	chunk_size: usize,
	w_threads : usize,
	num_shard : usize,
	sa_size   : usize,
) -> io::Result<()> {

	log::info!("Phase 2 begins.. (~10% >> ~90%)");

	// 1. preparation
	let num_samples = samples.len();
	let num_loops = (num_tokens + chunk_size - 1) / chunk_size;
	let pb_val = num_shard * (num_loops + (num_samples + 1));
	let pb = ProgressBar::new(pb_val as u64);
	pb.set_style(ProgressStyle::with_template(
		"{spinner:.green} {bar:64.yellow/yellow} {pos}/{len} ETA {msg}").unwrap());

	// 2. initialize
	let max_rough = (max_tokens + rough_div - 1) / rough_div;
	let sa: FastMmapVec<u8> = FastMmapVec::new(input_path.clone() + "/sa.bin", max_tokens * sa_size)?;
	let rough: FastMmapVec<u64> = FastMmapVec::new(input_path.clone() + "/rough.bin", max_rough * 5)?;
	let mut index: FastMmapVec<u64> = FastMmapVec::new(input_path.clone() + "/index.bin", 0)?;
	let mut index_posi = 0;
	let mut prv_2a_time: f64 = 0.0;
	let mut prv_2b_time: f64 = 0.0;

	// 3. main part
	for shard_id in 0..num_shard {
		let lft_sample = (shard_id + 0) * (num_samples + 1) / num_shard;
		let rgt_sample = (shard_id + 1) * (num_samples + 1) / num_shard;
		let new_sz = offset[rgt_sample] - offset[lft_sample];
		let _ = index.resize((index_posi + new_sz) * 4)?;

		// 4-0. get eta
		let eta_2b: f64;
		if lft_sample == 0 {
			eta_2b = -1.0;
		}
		else {
			eta_2b = prv_2b_time * ((num_samples + 1 - lft_sample) as f64) / (lft_sample as f64);
		}

		// 4-1. initial sorting
		phase2a(
			token, &samples, &offset, lft_sample, rgt_sample, index_posi,
			sa.as_file(),
			index.as_file(),
			num_tokens, chunk_size,
			w_threads, sa_size, &pb,
			&mut prv_2a_time,
			eta_2b,
			shard_id,
			num_shard
		);

		// 4-2. final sorting
		let eta_2a: f64 = prv_2a_time * ((num_shard - 1 - shard_id) as f64) / ((shard_id + 1) as f64);
		let ret = phase2b(
			&offset, lft_sample, rgt_sample, index_posi,
			sa.as_file(),
			index.as_file(),
			rough.as_file(),
			rough_div, w_threads, sa_size, num_shard, &pb,
			&mut prv_2b_time,
			eta_2a,
			num_samples
		);
		index_posi = ret;
	}

	// 5. last add
	let last_rough = ((num_tokens + rough_div - 1) / rough_div) as u64;
	let last_data = [u64::MAX, u64::MAX, u64::MAX, u64::MAX, index_posi as u64];
	unsafe {
		let data_u8 = slice::from_raw_parts(last_data.as_ptr() as *const u8, 40);
		rough.as_file().write_all_at(data_u8, last_rough * 40).expect("rough write failed");
	}

	// 6. return
	thread::sleep(time::Duration::from_millis(50));
	pb.inc(pb.length().unwrap() - pb.position());
	thread::sleep(time::Duration::from_millis(50));
	let _ = index.resize(index_posi * 4);
	log::info!("Phase 2 finished (Progress: ~90%)");
	Ok({})
}