use numpy::ndarray::{ArrayView1};
use std::collections::HashMap;
use rayon::prelude::*;
use std::cmp;
use crate::search::z_bsearch::get_match_exists_main;
use crate::search::z_bsearch::get_hash;
use crate::search::z_check::check_valid;
use crate::helper::bsearch;
use crate::helper::compress;
use crate::helper::softmin;
use crate::helper::read_from_file;
use crate::helper::get_array4;
use crate::helper::retrieve_value;
use crate::helper::get_upper_convex;
use crate::helper::check_subsequence;
use std::fs::File;
use std::time::{Instant};


// ====================================================================================================================
// The function to search all patterns where similarity is ">=thres"
//     - Input
//          * map      : a dictionary which records pairs of (pattern, whether match exists or not)
//          * pair_    : the existance of 2-gram
//          * trio_    : the existance of 3-gram
//          * rough    : positions 0, 128, 256, ... of index list
//          * match_sim: a 2D vector which records the list of similarity for each pattern position
//          * inst_mult: a 1D vector which records the insertion cost
//          * delt_mult: a 1D vector which records the delection cost
//          * pair_cons: the size of table (top-"pair_cons" words should be considered)
//          * trio_cons: the size of table (top-"trio_cons" words should be considered)
//          * thres    : the threshold of similarity
// 
//     - Output
//          * list of (pattern, similarity, #hits) for each pattern
// ====================================================================================================================
pub fn enumerate_algorithm(
	map  : &mut HashMap<u64, (u32, usize)>,
	pair_: ArrayView1<u64>,
	trio_: ArrayView1<u64>,
	rough: ArrayView1<u64>,
	match_sim : Vec<Vec<(f32, usize)>>,
	inst_mult : Vec<(f32, usize)>,
	delt_mult : Vec<f32>,
	num_tokens: usize,
	idx_length: usize,
	pair_cons : usize,
	trio_cons : usize,
	rough_div : usize,
	thres     : f32,
	cp_limit  : usize,
	file      : &File,
	timer     : &Instant,
	goal_time : f32
) -> (Vec<Vec<u32>>, Vec<f32>, u64) {

	// 1. preparation
	//    - current_cand: the current candidates of (similarity, multiplier, pattern)
	//    - lft_        : the leftmost position of pattern which already searched
	//    - rgt_        : the rightmost position of pattern which already searched
	//    - num_adds    : how many additions to current_cand occurs?
	//    - num_threads : the number of threads in parallelization
	let pat_len: usize = delt_mult.len();
	let mut current_cand: Vec<(f32, f32, Vec<u32>, usize)> = Vec::new();
	let mut num_adds: u64 = 0;
	let num_threads = rayon::current_num_threads();

	// initially, both similarity & multiplier is 1
	// this is because the initial sequence is empty
	current_cand.push((1.00, 1.00, Vec::new(), 0));

	// 2. search begins
	for idx in 0..pat_len {
		let max_inst = 4; // maximum number of insertions for each step
		let n_parallel = num_threads * 2;
		let chunk_size = cmp::max(1, (current_cand.len() + n_parallel - 1) / n_parallel);

		// 3. paralellization begins
		let results: Vec<(Vec<(f32, f32, Vec<u32>, usize)>, Vec<(u64, (u32, usize))>)> = current_cand
			.par_chunks(chunk_size).map(|chunk| {
				
				// (a) put the candidates in the previous loop
				//     - local_cand[i]: the candidates where "i" insertions are performed in this step
				//     - i = max_inst is special; which records the final candidates
				let mut local_cand: Vec<Vec<(f32, f32, Vec<u32>, usize)>> = vec![Vec::new(); max_inst + 1];
				let mut local_decided: Vec<Vec<(f32, f32, Vec<u32>, usize)>> = vec![Vec::new(); max_inst + 1];
				for item in chunk {
					local_cand[0].push(item.clone());
				}
				let mut local_map_adds: Vec<(u64, (u32, usize))> = Vec::new();

				// iter = the current number of insertion
				for iter in 0..max_inst {
					if local_cand[iter].len() == 0 && iter != max_inst - 1 {
						continue;
					}
					let work = std::mem::take(&mut local_cand[iter]);
					
					// (b) enumerate all possible ways of matching/insert/delete
					for (current_sim, current_mult, current_seq, start_pos) in work.iter() {
						let mut seq_buf: Vec<u32> = Vec::with_capacity(current_seq.len() + 1);
						let mut cand_next: Vec<(u32, i32)> = Vec::new();
						if thres < 0.95 && 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
							break;
						}

						// <b-0> if #matches is already small, get "candidates for next tokens"
						// with this method, the runtime decreases by ~15%
						if true {
							let target_pos: usize = idx_length.min(*start_pos + 50);
							let stt_bytes = (*start_pos as usize) * 32;
							let end_bytes = (target_pos as usize) * 32;
							let buf = read_from_file(&file, stt_bytes, end_bytes);
							let target: [u64; 4] = get_array4(&buf, target_pos - 1 - *start_pos);
							let cur_seqhash: [u64; 4] = compress(&current_seq, 1);

							// search
							if target >= cur_seqhash {
								for i in *start_pos..target_pos {
									let tgt: [u64; 4] = get_array4(&buf, i - *start_pos);
									if tgt >= cur_seqhash {
										break;
									}
									let cnd = retrieve_value(&tgt, current_seq.len()) as u32;
									cand_next.push((cnd, (i - start_pos) as i32));
								}
							}
							cand_next.sort_unstable();
						}

						// <b-1> matching
						if current_seq.len() < 12 {
							for (sim, k) in &match_sim[idx] {
								if softmin(*current_sim, *sim) * current_mult < thres {
									break;
								}
								let v = bsearch(&cand_next, *k as u32);
								if cand_next.len() == 0 || v != -1 {
									seq_buf.clear();
									seq_buf.extend_from_slice(current_seq);
									seq_buf.push(*k as u32);
									if cand_next.len() == 0 {
										if check_valid(&seq_buf, pair_, trio_, pair_cons, trio_cons) == true {
											let nex_sim: f32 = softmin(*current_sim, *sim);
											local_cand[max_inst].push((nex_sim, *current_mult, seq_buf.clone(), *start_pos));
										}
									}
									else {
										let nex_sim: f32 = softmin(*current_sim, *sim);
										local_decided[max_inst].push((nex_sim, *current_mult, seq_buf.clone(), *start_pos + (v as usize)));
										local_map_adds.push((get_hash(&seq_buf), (2, *start_pos + (v as usize))));
									}
								}
							}
						}

						// <b-2> deletion
						if current_sim * current_mult * delt_mult[idx] >= thres {
							let nex_mult = current_mult * delt_mult[idx];
							local_cand[max_inst].push((*current_sim, nex_mult, current_seq.clone(), *start_pos));
						}

						// <b-3> insert
						if iter <= max_inst - 2 && current_seq.len() >= 1 && current_seq.len() < 12 {
							for (mult, k) in &inst_mult {
								if current_sim * current_mult * mult < thres {
									break;
								}
								let v = bsearch(&cand_next, *k as u32);
								if cand_next.len() == 0 || v != -1 {
									seq_buf.clear();
									seq_buf.extend_from_slice(current_seq);
									seq_buf.push(*k as u32);
									if cand_next.len() == 0 {
										if check_valid(&seq_buf, pair_, trio_, pair_cons, trio_cons) == true {
											let nex_mult = current_mult * mult;
											local_cand[iter + 1].push((*current_sim, nex_mult, seq_buf.clone(), *start_pos));
										}
									}
									else {
										let nex_mult = current_mult * mult;
										local_decided[iter + 1].push((*current_sim, nex_mult, seq_buf.clone(), *start_pos + (v as usize)));
										local_map_adds.push((get_hash(&seq_buf), (2, *start_pos + (v as usize))));
									}
								}
							}
						}
					}

					// (c) sort the array local_cand
					local_cand[iter + 1].sort_unstable_by(|a, b| a.2.cmp(&b.2));
					if thres < 0.95 && 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
						break;
					}

					// (d) get "unique" sequences of local_cand
					//    - uniq: (sequence, lower_bound in local_cand, upper_bound)
					//    - prv_: the previous position which we need for calculation
					let mut uniq: Vec<(Vec<u32>, usize, usize)> = Vec::new();
					let mut prv_ = 0;
					let cand_len = local_cand[iter + 1].len();
					if cand_len >= 1 {
						for i in 0..cand_len {
							if i + 1 == cand_len || local_cand[iter + 1][i].2 != local_cand[iter + 1][i + 1].2 {
								uniq.push((local_cand[iter + 1][i].2.clone(), prv_, i + 1));
								prv_ = i + 1;
							}
						}
					}
					if thres < 0.95 && 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
						break;
					}

					// (e) check whether the pattern exists in the corpus
					let checked_results: Vec<(usize, usize, usize, usize)>
						= uniq.iter().map(|(seq, ordl, ordr)| {
						let (flg, posi) = get_match_exists_main(
							rough, rough_div, map, seq, &file, num_tokens, idx_length, &timer, goal_time
						);
						(*ordl, *ordr, flg, posi)}).collect();
					if thres < 0.95 && 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
						break;
					}

					// (f) update
					let mut next_work: Vec<(f32, f32, Vec<u32>, usize)> = Vec::new();
					for (ordl, ordr, flg, posi) in checked_results {
						let h = get_hash(&local_cand[iter + 1][ordl].2);
						let v = ((flg as u32) & 1) + 1;
						if v == 2 {
							for cx in ordl..ordr {
								let tmp_ = local_cand[iter + 1][cx].clone();
								next_work.push((tmp_.0, tmp_.1, tmp_.2, posi));
							}
						}
						if flg >= 2 {
							local_map_adds.push((h, (v, posi)));
						}
					}
					for vec_ in local_decided[iter + 1].iter() {
						next_work.push(vec_.clone());
					}
					local_cand[iter + 1] = next_work;
					if thres < 0.95 && 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
						break;
					}
				}

				// (g) return the following two:
				//     - the candidates which should be proceed to the next step
				//     - the dictionary between "sequence" and "whether exists or not in the corpus"
				(local_cand[max_inst].clone(), local_map_adds)
			})
			.collect();

		// check timer
		if thres < 0.95 && 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
			return (vec![], vec![869120.0], cp_limit as u64);
		}

		
		// ============================================================================================================
		// 4. non-parallel phase
		// (a) allocate memory
		let mut raw_next_candidates: Vec<(f32, f32, Vec<u32>, usize)> = Vec::new();
		let total_cand_size: usize = results.iter().map(|(c, _)| c.len()).sum();
		raw_next_candidates.reserve(total_cand_size);

		// (b) merge candidates (for each threads)
		for (mut cands, adds) in results {
			raw_next_candidates.append(&mut cands);
			for (h, v) in adds {
				// add to dictionary
				if !map.contains_key(&h) {
					num_adds += 1;
					map.insert(h, v);
				}
			}
		}
		if num_adds as usize >= cp_limit {
			return (Vec::new(), Vec::new(), cp_limit as u64);
		}

		// (c) update current_cand for the next step
		//     - we delete the candidates where (similarity, multiplier) is worse than other
		//     - e.g. we delete (0.2, 0.6) if there is an (0.5, 0.8)
		raw_next_candidates.par_sort_unstable_by(|a, b| a.2.cmp(&b.2));
		current_cand = Vec::new();
		let mut prev_ = 0;
		let len = raw_next_candidates.len();
		if len > 0 {
			for i in 0..len {
				if i + 1 == len || raw_next_candidates[i].2 != raw_next_candidates[i + 1].2 {
					let mut tmp_: Vec<(f32, f32)> = Vec::new();
					for j in prev_..(i + 1) {
						tmp_.push((raw_next_candidates[j].0, raw_next_candidates[j].1));
					}
					tmp_ = get_upper_convex(tmp_);
					for (sim, mult) in tmp_ {
						current_cand.push((sim, mult, raw_next_candidates[i].2.clone(), raw_next_candidates[i].3));
					}
					prev_ = i + 1;
				}
			}
		}
	}


	// 5. enumerate final pattern candidates
	current_cand.par_sort_unstable_by(|a, b| (b.0 * b.1).partial_cmp(&(a.0 * a.1)).unwrap());
	let mut fin_cand: Vec<Vec<u32>> = Vec::new();
	let mut fin_cost: Vec<f32> = Vec::new();
	for (sim, mult, seq, _) in current_cand.iter() {
		let mut flg: bool = true;

		// exclude subsequence (e.g., "the sun" when "sun" already exists)
		for seq2 in fin_cand.iter() {
			if check_subsequence(seq.clone(), seq2.clone()) == true {
				flg = false;
				break;
			}
		}

		// addition
		if seq.len() == 0 {
			flg = false;
		}
		if flg == true {
			fin_cand.push(seq.clone());
			fin_cost.push(sim * mult);
		}
	}

	// 6. return
	(fin_cand, fin_cost, num_adds)
}