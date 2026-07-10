use numpy::ndarray::{ArrayView1};
use std::collections::HashMap;
use std::fs::File;
use std::time::{Instant};
use crate::helper::compress;
use crate::helper::get_array4;
use crate::helper::read_from_file;
use crate::helper::retrieve_value;


// ====================================================================================================================
// 1. The function to return the sequence hash of the given sequence
//     - Input
//          * seq: pattern sequence whose length is 12 or less
// 
//     - Output
//          * random hash value between [0, 2^64)
// ====================================================================================================================
pub fn get_hash(seq: &[u32]) -> u64 {
	let mut hash_val: u64 = 99999999;
	for i in 0..seq.len() {
		hash_val = 8691201001 * hash_val + ((seq[i] as u64) + 1234567890123);
	}
	hash_val
}



// ====================================================================================================================
// 2. Get the "rough" place in th suffix array
//     - Input
//          * rough   : positions 0, 128, 256, ... of index list
//          * seq_hash: the hashed sequence of the pattern whose length is 4
// 
//     - Output
//          * the rough position of pattern sequence in index[] (error of +/- 128 is allowed)
// ====================================================================================================================
pub fn get_rough_lower_bound(rough: ArrayView1<u64>, seq_hash: &[u64; 4]) -> (i64, i64, i64) {
	let mut ng: i64 = -1;
	let mut ok: i64 = (rough.len() / 5) as i64;

	// binary search
	while ok - ng > 1 {
		let mid: i64 = (ok + ng) / 2;
		let mid_: usize = mid as usize;
		let mut flg: bool = true;
		let chunk: [u64; 4] = [
			rough[mid_ * 5 + 0],
			rough[mid_ * 5 + 1],
			rough[mid_ * 5 + 2],
			rough[mid_ * 5 + 3]
		];
		if seq_hash > &chunk {
			flg = false;
		}
		if flg == false {
			ng = mid;
		}
		else {
			ok = mid;
		}
	}

	// return
	if ok == 0 {
		(ok, -1, rough[ok as usize * 5 + 4] as i64)
	}
	else {
		(ok, rough[ok as usize * 5 - 1] as i64 - 1, rough[ok as usize * 5 + 4] as i64)
	}
}



// ====================================================================================================================
// 3. Get the "rigor" place in th suffix array
//     - Input
//          * rough     : positions 0, 128, 256, ... of index list
//          * rough_div : 128
//          * seq       : pattern sequence
//          * file      : path to index
//          * num_tokens: number of tokens
//          * idx_length: the length of the index file
// 
//     - Output
//          * the exact position of pattern sequence in index[]
// ====================================================================================================================
pub fn get_rigor_lower_bound(
	rough: ArrayView1<u64>,
	rough_div: usize,
	seq: &[u32],
	file: &File,
	num_tokens: usize,
	idx_length: usize
) -> (i64, i64) {

	// [a] get rough lower bound
	let va = compress(seq, 0);
	let (pos, ng__, ok__) = get_rough_lower_bound(rough, &va);
	let ng_ = ng__.max(-1);
	let ok_ = ok__.min(idx_length as i64);
	if ng_ >= ok_ {
		((pos * rough_div as i64).min(num_tokens as i64) + (1 << 59), ok_ + (1 << 59))
	}

	else {
		// [b] initialize
		let stt_bytes = ((ng_ + 1) as usize) * 32;
		let end_bytes = (idx_length.min(ok_ as usize + 1)) * 32;
		let buf = read_from_file(&file, stt_bytes, end_bytes);
		let seq_hash: [u64; 4] = compress(seq, 0);
		let mut ng = ng_.max(-1);
		let mut ok = ok_.min(idx_length as i64);

		// [c] binary search
		while ok - ng > 1 {
			let mid: i64 = (ok + ng) / 2;
			let mid_: usize = (mid - (ng_ + 1)) as usize;
			let mut flg: i32 = 1;
			let chunk = get_array4(&buf, mid_);
			if seq_hash > chunk {
				flg = 0;
			}
			else if seq_hash < chunk {
				flg = 2;
			}
			if flg == 0 {
				ng = mid;
			}
			else {
				ok = mid;
			}
		}

		// [d] last check
		if ok >= (idx_length as i64) {
			(num_tokens as i64 + (1 << 59), ok + (1 << 59))
		}
		else {
			let mut final_pos = (pos - 1) * (rough_div as i64) + retrieve_value(
				&get_array4(&buf, (ok - (ng_ + 1)) as usize), 12
			) as i64;
			if final_pos % (rough_div as i64) == 0 {
				final_pos += rough_div as i64;
			}
			let ok2 = (ok - (ng_ + 1)) as usize;
			let va = compress(seq, 0);
			let vb = compress(seq, 1);
			let chunk = get_array4(&buf, ok2);
			if va <= chunk && chunk < vb {
				(final_pos, ok)
			}
			else {
				(final_pos + (1 << 59), ok + (1 << 59))
			}
		}
	}
}



// ====================================================================================================================
// 4. Check whether match exists or not
//     - Input
//          * rough     : positions 0, 128, 256, ... of index list
//          * rough_div : 128
//          * seq       : pattern sequence
//          * file      : path to index
//          * num_tokens: number of tokens
//          * idx_length: the length of the index file
// 
//     - Output
//          * (exists or not, position/lower_bound in index)
// ====================================================================================================================
pub fn get_match_exists(
	rough: ArrayView1<u64>,
	rough_div: usize,
	seq: &[u32],
	file: &File,
	num_tokens: usize,
	idx_length: usize
) -> (bool, usize) {

	let pos = get_rigor_lower_bound(
		rough, rough_div,
		seq, file, num_tokens, idx_length
	).1;
	if (pos as usize) >= (1 << 60) {
		(false, 0)
	}
	else if (pos as usize) >= (1 << 59) {
		(false, (pos as usize) - (1 << 59))
	}
	else {
		(true, pos as usize)
	}
}



// ====================================================================================================================
// 5. Check whether match exists or not (public)
//     - Input
//          * rough     : positions 0, 128, 256, ... of index list
//          * rough_div : 128
//          * seq       : pattern sequence
//          * file      : path to index
//          * num_tokens: number of tokens
//          * idx_length: the length of the index file
// 
//     - First Output
//          * 0: already searched in history & match does not exist
//          * 1: already searched in history & match exists
//          * 2: searched first-time & match does not exist
//          * 3: searched first-time & match exists
// 
//     - Second Output
//          * The position (lower_bound) in index
// ====================================================================================================================
pub fn get_match_exists_main(
	rough     : ArrayView1<u64>,
	rough_div : usize,
	map       : &HashMap<u64, (u32, usize)>,
	seq       : &[u32],
	file      : &File,
	num_tokens: usize,
	idx_length: usize,
	timer     : &Instant,
	goal_time : f32
) -> (usize, usize) {
	let h = get_hash(&seq);
	let v = get_dict(map, h);
	if v.0 == 0 {
		if 1.0e-3 * (timer.elapsed().as_millis() as f32) >= goal_time {
			(0, 0)
		}
		else {
			let q = get_match_exists(rough, rough_div, seq, &file, num_tokens, idx_length);
			(2 + q.0 as usize, q.1)
		}
	}
	else if v.0 == 1 {
		(0, v.1)
	}
	else {
		(1, v.1)
	}
}



// ====================================================================================================================
// 6. Count the number of matches
//     - Input
//          * rough     : positions 0, 128, 256, ... of index list
//          * rough_div : 128
//          * seq       : pattern sequence
//          * file      : path to index
//          * num_tokens: number of tokens
//          * idx_length: the length of the index file
// 
//     - Output
//          * number of occurences in the corpus (return the lower_bound and upper_bound)
// ====================================================================================================================
pub fn get_match_range(
	rough: ArrayView1<u64>,
	rough_div: usize,
	seq: &[u32],
	file: &File,
	num_tokens: usize,
	idx_length: usize
) -> (u64, u64) {
	let seq1 = seq.to_vec();
	let mut seq2 = seq.to_vec();
	if seq1.len() == 0 {
		seq2 = vec![1 << 30];
	}
	else {
		seq2[seq.len() - 1] += 1;
	}

	// get rough range
	let pos1 = get_rigor_lower_bound(
		rough, rough_div,
		&seq1, file, num_tokens, idx_length
	).0;
	let pos2 = get_rigor_lower_bound(
		rough, rough_div,
		&seq2, file, num_tokens, idx_length
	).0;
	((pos1 % (1 << 59)) as u64, (pos2 % (1 << 59)) as u64)
}


// ====================================================================================================================
// 7. Check whether a sequence is already searched or not
//     - Input
//          * map     : a dictionary which records pairs of (pattern, whether match exists or not)
//          * hash_val: pattern sequence
// 
//     - Output
//          * 0: not searched yet
//          * 1: searched & such sequence does not exist in the corpus
//          * 2: searched & such sequence exists in the corpus
// ====================================================================================================================
pub fn get_dict(map: &HashMap<u64, (u32, usize)>, hash_val: u64) -> (u32, usize) {
	if let Some(value) = map.get(&hash_val) {
		*value
	}
	else {
		(0, 0)
	}
}


// ====================================================================================================================
// 8. Whether a 2-gram [u1, u2] exists in the corpus
//     - Input
//          * pair_    : an index that records the existance of 2-gram
//          * pair_cons: the size of index (top-"pair_cons" words should be considered)
//          * [u1, u2] : 2-gram which we want to check
// 
//     - Output
//          * 0: unknown
//          * 1: does not exist
//          * 2: exists
// ====================================================================================================================
pub fn search_2(pair_: ArrayView1<u64>, pair_cons: usize, u1: u32, u2: u32) -> u32 {
	if u1 < (pair_cons as u32) && u2 < (pair_cons as u32) {
		let hash: i64 = (u1 as i64) * (pair_cons as i64) + (u2 as i64);
		if ((pair_[(hash >> 6) as usize] >> (hash & 63)) & 1) == 1 {
			2
		}
		else {
			1
		}
	}
	else {
		0
	}
}


// ====================================================================================================================
// 9. Whether a 3-gram [u1, u2, u3] exists in the corpus
//     - Input
//          * trio_       : an index that records the existance of 3-gram
//          * trio_cons   : the size of index (top-"trio_cons" words should be considered)
//          * [u1, u2, u3]: 2-gram which we want to check
// 
//     - Output
//          * 0: unknown
//          * 1: does not exist
//          * 2: exists
// ====================================================================================================================
pub fn search_3(trio_: ArrayView1<u64>, trio_cons: usize, u1: u32, u2: u32, u3: u32) -> u32 {
	if u1 + u2 + u3 < (trio_cons as u32) {
		let t: i64 = trio_cons as i64;
		let v0: i64 = u1 as i64;
		let v1: i64 = u2 as i64;
		let v2: i64 = u3 as i64;
		let hash: i64 = ((t + 2) * (t + 1) * t) / 6 - ((t + 2 - v0) * (t + 1 - v0) * (t + 0 - v0)) / 6
				+ v1 * (t - v0) - v1 * (v1 - 1) / 2
				+ v2;
		if ((trio_[(hash >> 6) as usize] >> (hash & 63)) & 1) == 1 {
			2
		}
		else {
			1
		}
	}
	else {
		0
	}
}