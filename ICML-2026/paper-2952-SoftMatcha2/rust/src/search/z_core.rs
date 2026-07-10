use numpy::ndarray::{ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::fs::File;
use std::time::{Instant};
use rayon::prelude::*;
use crate::search::z_bsearch::get_match_range;
use crate::search::z_enumerate::enumerate_algorithm;


// ====================================================================================================================
// Enumerate top-"goal" patterns
//     - Input
//          * pat  : pattern tokens, length P
//          * score: similarity index for each (position in pattern, word id), P*V matrix
//          * rough: positions 0, 128, 256, ... of index list
//          * pair_: the existance of 2-gram
//          * trio_: the existance of 3-gram
//          * norm : norm list for each word, length V
// 
//     - Input (Goal)
//          * goal      : number of patterns we output (default = 20)
//          * goal_alpha: minimum threshold we output  (default = 0.45)
//          * goal_time : maximum runtime we allow     (default = 10)
// 
//     - Output
//          * for each candidates, (pattern, similarity score, #match count)
//          * we also output minimum similarity we searched
// ====================================================================================================================
pub fn compute(
	pat  : ArrayView1<u64>,
	score: ArrayView2<f32>,
	rough: ArrayView1<u64>,
	pair_: ArrayView1<u64>,
	trio_: ArrayView1<u64>,
	norm : ArrayView1<f32>,
	num_tokens: usize,
	idx_length: usize,
	pair_cons : usize,
	trio_cons : usize,
	rough_div : usize,
	goal      : usize,
	goal_alpha: f32,
	goal_time : f32,
	file: &File
) -> (Vec<Vec<u64>>, Vec<f32>, Vec<u64>, f32) {

	// 1. preparation
	//   - inst_rank: the cost of insert/delete (the lower the value, the heavier the cost)
	//   - map      : the map that records "exists in corpus or not" for each pattern sequence
	//   - alpha    : the list of thresholds of similarity
	let inst_rank: usize = 50;
	let pat_len: usize = score.nrows();
	let mut map: HashMap<u64, (u32, usize)> = HashMap::new();
	let alpha: Vec<f32> = vec![
		0.99,
		0.80, 0.70, 0.60, 0.58, 0.57,
		0.56, 0.55, 0.54, 0.53, 0.52,
		0.51, 0.50, 0.49, 0.48, 0.47,
		0.46, 0.45, 0.44, 0.43, 0.42,
		0.41, 0.40, 0.39, 0.38, 0.37,
		0.36, 0.35, 0.34, 0.33, 0.32,
		0.31, 0.30, 0.29, 0.28, 0.27,
		0.26, 0.25, 0.24, 0.23, 0.22,
		0.21, 0.20
	];
	let timer: Instant = Instant::now();


	// 2. calculate cost for match & insert & delete
	//   - match_sim[i]: the list of similarity when matched with the i-th token in the given pattern
	//   - inst_mult   : the list of cost (multiplier) when we add a new token
	//   - delt_mult[i]: the cost (multiplier) when we delete the i-th token in the given pattern
	let match_sim: Vec<Vec<(f32, usize)>>;
	let mut inst_mult: Vec<(f32, usize)> = Vec::new();
	let mut delt_mult: Vec<f32> = vec![0.0; pat_len];
	{
		// (a) calculate match_sim
		match_sim = (0..pat_len).into_par_iter().map(|i| {
			let mut row_: Vec<(f32, usize)> = Vec::new();
			for j in 0..(score.ncols() - 1) {
				let sim = score[(i, j)].max(0.0);
				if sim >= alpha[alpha.len() - 1].max(goal_alpha) {
					row_.push((sim, j));
				}
			}
			row_.par_sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
			row_
		}).collect();

		// (b) get insertion coefficient (step 1)
		let mut inst_tmp: Vec<(f32, usize)> = Vec::new();
		let mut temp: Vec<f32> = Vec::new();
		for i in 0..(score.ncols()/50) {
			temp.push(norm[i]);
		}
		temp.par_sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
		let qual: f32;
		if inst_rank * inst_rank + 100 < temp.len() {
			qual = temp[inst_rank * inst_rank + 100];
		}
		else {
			qual = 1.0e9;
		}

		// (c) get insertion coefficient (step 2)
		for i in 0..(score.ncols() - 1) {
			if norm[i] <= qual {
				inst_tmp.push((norm[i], i));
			}
		}
		inst_tmp.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
		let mut actual_inst_rank: usize = inst_rank;
		for i in 1..inst_rank {
			if inst_tmp[i].0 >= 1.0e10 {
				actual_inst_rank = i - 1;
				break;
			}
		}
		let coefficient = inst_tmp[actual_inst_rank].0 * (pat_len as f32) / 5.0;

		// (d) calculate inst_mult: cost for insertion
		for i in 0..(score.ncols() - 1).min(inst_tmp.len()) {
			let mult: f32 = (-norm[inst_tmp[i].1] / coefficient).exp();
			if mult < alpha[alpha.len() - 1].max(goal_alpha) {
				break;
			}
			inst_mult.push((mult, inst_tmp[i].1));
		}

		// (e) calculate delt_mult: cost for deletion
		for i in 0..pat_len {
			if (pat[i] as usize) != score.ncols() - 1 {
				let mult: f32 = (-norm[pat[i] as usize] / coefficient).exp();
				delt_mult[i] = mult;
			}
			else {
				delt_mult[i] = 0.0;
			}
		}
	}

	
	// 3. preparation for search
	//   - return_pat     : the returned pattern which is similar to pat[]
	//   - return_score   : the similarity for each returned pattern
	//   - time_complexity: current number of searched patterns
	let mut return_pat: Vec<Vec<u32>> = Vec::new();
	let mut return_score: Vec<f32> = Vec::new();
	let mut last_alpha: f32 = 1.01;
	let mut time_complexity: u64 = 0;

	// 4. search begins
	for (idx_, thres) in alpha.iter().enumerate() {
		// (a) search
		let (ans, ans_score, num_search) = enumerate_algorithm(
			&mut map, pair_, trio_, rough,
			match_sim.clone(),
			inst_mult.clone(),
			delt_mult.clone(),
			num_tokens, idx_length, pair_cons, trio_cons, rough_div, *thres,
			500_000_000,
			&file, &timer, goal_time
		);
		time_complexity += num_search;

		// (b) check timeout
		let mut end = false;
		if ans_score == vec![869120.0] {
			end = true;
		}
		else {
			return_pat = ans.clone();
			return_score = ans_score.clone();
			last_alpha = *thres;
		}

		// (c) check reaching goal
		if ans.to_vec().len() >= (goal as usize) || idx_ == alpha.len() - 1 || alpha[idx_ + 1] < goal_alpha {
			end = true;
		}
		if end == true {
			println!("#Search = {:?}", time_complexity);
			break;
		}
	}

	// 5. count the number of matches for each matched pattern
	let mut match_count: Vec<u64> = Vec::new();
	for i in 0..return_pat.len() {
		let p = get_match_range(
			rough, rough_div, &return_pat[i], &file, num_tokens, idx_length
		);
		match_count.push(p.1 - p.0);
	}

	// 6. return
	let return_pat_real: Vec<Vec<u64>> = return_pat
		.into_iter().map(|inner| inner.into_iter().map(u64::from).collect()).collect();
	(return_pat_real, return_score, match_count, last_alpha)
}