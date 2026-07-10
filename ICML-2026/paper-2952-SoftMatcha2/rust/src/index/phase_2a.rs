use numpy::ndarray::{ArrayView1};
use std::slice;
use std::fs::File;
use rayon::prelude::*;
use std::time::{Instant};
use indicatif::{ProgressBar};
use std::os::unix::fs::FileExt;
use crate::helper::format_eta;
use crate::helper::compress_build;


// =====================================================================================================================
// Phase II-a: Rough Suffix Array Construction
// 
// Goal:
//   - make rough suffix array
//   - after this, we only have to sort ranges [0, x1), [x1, x2), [x2, x3), ..., for some (x1, x2, ...).
// =====================================================================================================================
pub fn phase2a(
	token     : ArrayView1<u32>,
	samples   : &Vec<[u64; 5]>,
	offset    : &Vec<usize>,
	lft_sample: usize,
	rgt_sample: usize,
	index_posi: usize,
	file_sa   : &File,
	file_id   : &File,
	num_tokens: usize,
	chunk_size: usize,
	w_threads : usize,
	sa_size   : usize,
	pb        : &ProgressBar,
	prv_2a    : &mut f64,
	eta_2b    : f64,
	shard_id  : usize,
	num_shard : usize
) {
	let n_threads = rayon::current_num_threads();
	let num_samples = samples.len();
	let minus_offset = offset[lft_sample] - index_posi;
	let timer = Instant::now();
	let num_loops = (num_tokens + chunk_size - 1) / chunk_size;

	// 1. preparation
	let mut fill = vec![0usize; num_samples + 1];
	let mut res = vec![0 as u32; chunk_size];
	let mut sub_sa = vec![0 as u8; chunk_size * sa_size];
	let mut sub_id = vec![0 as u64; chunk_size * 4];

	// 2. parallelization
	for loop_id in 0..num_loops {
		let stt: usize = ((loop_id + 0) * chunk_size).min(num_tokens);
		let end: usize = ((loop_id + 1) * chunk_size).min(num_tokens);
		let chunk: usize = (end - stt + n_threads - 1) / n_threads;
		let mut sub_sum: Vec<Vec<usize>> = vec![vec![0usize; num_samples + 2]; n_threads + 1];
		let res_ptr = res.as_mut_ptr() as usize;
		let sa_ptr = sub_sa.as_mut_ptr() as usize;
		let id_ptr = sub_id.as_mut_ptr() as usize;

		// <2-1> calculate res
		let sub_cnt: Vec<Vec<usize>> = (0..n_threads).into_par_iter().map(|t| {
			let sub_stt: usize = (stt + (t + 0) * chunk).min(end);
			let sub_end: usize = (stt + (t + 1) * chunk).min(end);
			let res_ptr = res_ptr as *mut u32;
			let mut cnt: Vec<usize> = vec![0usize; num_samples + 2];
			unsafe {
				for i in sub_stt..sub_end {
					let key = compress_build(&token, i);
					let c = match samples.binary_search(&key) {
						Ok(pos) | Err(pos) => pos,
					};
					*res_ptr.add(i - stt) = c as u32;
					cnt[c] += 1;
				}
			}
			cnt
		}).collect();

		// <2-2> calculate sub_sum
		let mut cur_sum = 0;
		for c in 0..num_samples+2 {
			for t in 0..n_threads {
				sub_sum[t][c] = cur_sum;
				cur_sum += sub_cnt[t][c];
			}
		}

		// <2-3> update sub_sa & sub_id
		(0..n_threads).into_par_iter().for_each(|t| {
			let sub_stt: usize = (stt + (t + 0) * chunk).min(end);
			let sub_end: usize = (stt + (t + 1) * chunk).min(end);
			let sa_ptr = sa_ptr as *mut u8;
			let id_ptr = id_ptr as *mut u64;
			let mut sub_fill: Vec<usize> = vec![0usize; num_samples + 1];
			unsafe {
				for i in sub_stt..sub_end {
					let c = res[i - stt] as usize;
					let u = sub_sum[t][c] + sub_fill[c];
					if lft_sample <= c && c < rgt_sample {
						let key = compress_build(&token, i);
						for k in 0..sa_size {
							*sa_ptr.add(u * sa_size + k) = ((i >> (k * 8)) & 255) as u8;
						}
						*id_ptr.add(4 * u + 0) = key[0];
						*id_ptr.add(4 * u + 1) = key[1];
						*id_ptr.add(4 * u + 2) = key[2];
						*id_ptr.add(4 * u + 3) = key[3];
						sub_fill[c] += 1;
					}
				}
			}
		});

		// <2-4> write to disk
		let wchunk = ((rgt_sample - lft_sample) + w_threads - 1) / w_threads;
		(0..w_threads).into_par_iter().for_each(|t| {
			let sub_stt: usize = (lft_sample + (t + 0) * wchunk).min(rgt_sample);
			let sub_end: usize = (lft_sample + (t + 1) * wchunk).min(rgt_sample);
			for c in sub_stt..sub_end {
				let id_size: usize = 4;
                let sa_posi_byte: u64 = 1 * (sa_size as u64) * ((offset[c] + fill[c]) as u64);
                let id_posi_byte: u64 = 8 * (id_size as u64) * ((offset[c] + fill[c] - minus_offset) as u64);
                let len_items = sub_sum[0][c + 1] - sub_sum[0][c];
                if len_items == 0 { continue; }
                let sa_src_start: usize = sub_sum[0][c] * sa_size;
                let id_src_start: usize = sub_sum[0][c] * id_size;
                let sa_src_end: usize   = sub_sum[0][c + 1] * sa_size;
                let id_src_end: usize   = sub_sum[0][c + 1] * id_size;
                let sa_buf = &sub_sa[sa_src_start..sa_src_end];
                let id_buf = &sub_id[id_src_start..id_src_end];

				// direct file I/O
                let id_buf_u8 = unsafe {
                    slice::from_raw_parts(
                        id_buf.as_ptr() as *const u8,
                        id_buf.len() * 8
                    )
                };
                file_sa.write_all_at(sa_buf, sa_posi_byte).expect("write failed sa");
                file_id.write_all_at(id_buf_u8, id_posi_byte).expect("write failed idx");
			}
		});

		// <2-5> update fill
		for c in 0..num_samples+1 {
			fill[c] += sub_sum[0][c + 1] - sub_sum[0][c];
		}

		// <2-6> update timer
		pb.inc(1);
		let elapsed = timer.elapsed().as_secs_f64();
		let processed = shard_id * num_loops + (loop_id + 1);
		let avg_secs = (*prv_2a + elapsed) / (processed as f64);
		let mut eta = avg_secs * ((num_shard * num_loops - processed) as f64) + eta_2b;
		if eta_2b < -0.5 {
			eta = avg_secs * ((pb.length().unwrap() - pb.position()) as f64);
		}
		pb.set_message(format_eta(eta as u64));
	}

	// update time
	*prv_2a += timer.elapsed().as_secs_f64();
}