use std::slice;
use std::fs::File;
use rayon::prelude::*;
use std::time::{Instant};
use indicatif::{ProgressBar};
use std::os::unix::fs::FileExt;
use crate::helper::format_eta;


// =====================================================================================================================
// Phase II-b: Rigor Suffix Array Construction
// 
// Goal:
//   - make complete suffix array by sorting some small ranges [l, r)
//   - after this, the suffix array is completed
// =====================================================================================================================
pub fn phase2b(
	offset    : &Vec<usize>,
	lft_sample: usize,
	rgt_sample: usize,
	index_posi: usize,
	file_sa   : &File,  // sa
    file_id   : &File,  // index
    file_ro   : &File,  // rough
	rough_div : usize,
	w_threads : usize,
	sa_size   : usize,
	num_shard : usize,
	pb        : &ProgressBar,
	prv_2b    : &mut f64,
	eta_2a    : f64,
	num_samples: usize
) -> usize {

	let n_threads = rayon::current_num_threads();
	let mut current_posi = index_posi;
	let timer = Instant::now();

	// get max count
	let mut max_count = 0;
	for c in lft_sample..rgt_sample {
		max_count = max_count.max(offset[c + 1] - offset[c]);
	}
	let mut recs: Vec<([u64; 4], u64)> = vec![([0, 0, 0, 0], 0); max_count];
	let mut idrg: Vec<u64> = vec![0; max_count * 4];
	
	// loop begins
	for c in lft_sample..rgt_sample {
		let start: usize = offset[c] - offset[lft_sample] + index_posi;
		let count: usize = offset[c + 1] - offset[c];

		// 1. preparation
		let chunk = (count + n_threads - 1) / n_threads;
		let recs_ptr = recs.as_mut_ptr() as usize;
		let idrg_ptr = idrg.as_mut_ptr() as usize;

		// 2. sorting register
		(0..n_threads).into_par_iter().for_each(|t| {
			let stt = ((t + 0) * chunk).min(count);
			let end = ((t + 1) * chunk).min(count);
			if stt < end {
				let recs_ptr = recs_ptr as *mut ([u64; 4], u64);
				let subchunk = (end - stt).min(1_048_576);
				let mut local_id_buf = vec![0u64; subchunk * 4];
				let mut local_sa_buf = vec![0u8; subchunk * sa_size];
				
				// register by subchunk
				for v in 0..(end-stt+subchunk-1)/subchunk {
					let stt_ = (stt + (v + 0) * subchunk).min(end);
					let end_ = (stt + (v + 1) * subchunk).min(end);

					// <2-a> read from file (index)
					let id_start_idx = (start + stt_) * 4;
					let id_end_idx   = (start + end_) * 4;
					let local_id_u8 = unsafe {
						slice::from_raw_parts_mut(
							local_id_buf[0 .. id_end_idx - id_start_idx].as_mut_ptr() as *mut u8,
							local_id_buf[0 .. id_end_idx - id_start_idx].len() * 8
						)
					};
					let id_offset = (id_start_idx * 8) as u64;
					let n1 = file_id.read_at(local_id_u8, id_offset).expect("read index failed");
					if n1 != local_id_u8.len() {
						println!("! short read index");
						panic!("Short read at index! req={} got={}", local_id_u8.len(), n1);
					}

					// <2-b> read from file (sa)
					let sa_start_idx = (offset[c] + stt_) * sa_size;
					let sa_end_idx   = (offset[c] + end_) * sa_size;
					let sa_offset = sa_start_idx as u64;
					let n2 = file_sa.read_at(
						&mut local_sa_buf[0 .. sa_end_idx - sa_start_idx], sa_offset
					).expect("read sa failed");
					if n2 != sa_end_idx - sa_start_idx {
						println!("! short read sa");
						panic!("Short read at sa! req={} got={}", local_sa_buf.len(), n2);
					}

					// <2-c> register to array recs, which we sort later
					unsafe {
						for idx in 0..(end_-stt_) {
							let key = [
								local_id_buf[idx * 4 + 0],
								local_id_buf[idx * 4 + 1],
								local_id_buf[idx * 4 + 2],
								local_id_buf[idx * 4 + 3]
							];
							let mut sa_val: u64 = 0;
							for j in 0..sa_size {
								sa_val += (1 << (j * 8)) * (local_sa_buf[idx * sa_size + j] as u64);
							}
							*recs_ptr.add(stt_ + idx) = (key, sa_val);
						}
					}
				}
			}
		});

		// 3. sorting
		recs[0..count].par_sort_unstable_by(|(ka, sa_a), (kb, sa_b)| {
			ka.cmp(kb).then_with(|| sa_a.cmp(sa_b))
		});

		// 4. write initialize (step 1)
		let cnt_list: Vec<usize> = (0..n_threads).into_par_iter().map(|t| {
			let stt = ((t + 0) * chunk).min(count);
			let end = ((t + 1) * chunk).min(count);
			let mut cnt: usize = 0;
			for i in stt..end {
				if i == 0 || (offset[c] + i) & (rough_div - 1) == 0 || recs[i - 1].0 != recs[i].0 {
					cnt += 1;
				}
			}
			cnt
		}).collect();

		// 5. write initialize (step 2)
		let rough_stt = (offset[c] + rough_div - 1) / rough_div;
		let rough_end = (offset[c + 1] + rough_div - 1) / rough_div;
		let mut rough_buf = vec![0u64; (rough_end - rough_stt) * 5];
		let rough_ptr = rough_buf.as_mut_ptr() as usize;
		(0..n_threads).into_par_iter().for_each(|t| {
			let stt = ((t + 0) * chunk).min(count);
			let end = ((t + 1) * chunk).min(count);
			let idrg_ptr = idrg_ptr as *mut u64;
			let rough_ptr = rough_ptr as *mut u64;
			let mut stt_place = 0;
			for i in 0..t {
				stt_place += cnt_list[i];
			}
			unsafe {
				for i in stt..end {
					// write the sorted information to rough
					if (offset[c] + i) & (rough_div - 1) == 0 {
						let pos = (offset[c] + i) / rough_div;
						*rough_ptr.add((pos - rough_stt) * 5 + 0) = recs[i].0[0] as u64;
						*rough_ptr.add((pos - rough_stt) * 5 + 1) = recs[i].0[1] as u64;
						*rough_ptr.add((pos - rough_stt) * 5 + 2) = recs[i].0[2] as u64;
						*rough_ptr.add((pos - rough_stt) * 5 + 3) = recs[i].0[3] as u64;
						*rough_ptr.add((pos - rough_stt) * 5 + 4) = (current_posi + stt_place) as u64;
					}

					// write the sorted information to idrg (buffer)
					if i == 0 || (offset[c] + i) & (rough_div - 1) == 0 || recs[i - 1].0 != recs[i].0 {
						*idrg_ptr.add(stt_place * 4 + 0) = recs[i].0[0] as u64;
						*idrg_ptr.add(stt_place * 4 + 1) = recs[i].0[1] as u64;
						*idrg_ptr.add(stt_place * 4 + 2) = recs[i].0[2] as u64;
						*idrg_ptr.add(stt_place * 4 + 3) = recs[i].0[3] as u64 + ((offset[c] + i) & (rough_div - 1)) as u64;
						stt_place += 1;
					}
				}
			}
		});

		// 6. write to rough array (rough.bin)
		if rough_end - rough_stt > 0 {
			unsafe {
            	let offset_bytes = (rough_stt * 5 * 8) as u64;
				let data_u8 = slice::from_raw_parts(rough_buf.as_ptr() as *const u8, 40 * (rough_end - rough_stt));
				file_ro.write_all_at(data_u8, offset_bytes).expect("rough write failed");
			}
		}
		let final_length: usize = cnt_list.iter().sum();

		// 7. write to disk
		let chunk1 = (count + w_threads - 1) / w_threads;
		let chunk2 = (final_length + w_threads - 1) / w_threads;
		(0..w_threads).into_par_iter().for_each(|t| {
			let stt = ((t + 0) * chunk1).min(count);
			let end = ((t + 1) * chunk1).min(count);
			let stt2 = ((t + 0) * chunk2).min(final_length);
			let end2 = ((t + 1) * chunk2).min(final_length);

			// <7-1> write sa
			if end - stt > 0 {
				let mut sa_buf = vec![0u8; (end - stt) * sa_size];
				for i in 0..end-stt {
					let sa_val = recs[stt + i].1;
					for j in 0..sa_size {
						sa_buf[i * sa_size + j] = ((sa_val >> (8 * j)) & 255) as u8;
					}
				}
                let offset_bytes = ((offset[c] + stt) * sa_size) as u64;
                file_sa.write_all_at(&sa_buf, offset_bytes).expect("sa write failed");
			}

			// <7-2> write index
			if end2 - stt2 > 0 {
				let id_src_slice = &idrg[stt2 * 4 .. end2 * 4];
                let id_src_u8 = unsafe {
                    slice::from_raw_parts(
                        id_src_slice.as_ptr() as *const u8,
                        id_src_slice.len() * 8
                    )
                };
                let offset_bytes = ((current_posi + stt2) * 4 * 8) as u64;
                file_id.write_all_at(id_src_u8, offset_bytes).expect("index write failed");
			}
		});
		current_posi += final_length;

		// 8. update timer
		pb.inc(num_shard as u64);
		let elapsed = timer.elapsed().as_secs_f64();
		let avg_time = (*prv_2b + elapsed) / ((c + 1) as f64);
		let eta = eta_2a + avg_time * ((num_samples - c) as f64);
		pb.set_message(format_eta(eta as u64));
	}

	// return
	*prv_2b += timer.elapsed().as_secs_f64();
	current_posi
}