use numpy::ndarray::{ArrayView1};
use std::fs::File;
use std::os::unix::fs::FileExt;


// =====================================================================================================================
// 1. Convert length-12 tokens to byte-hashed value [building mode]
//     - Input
//          * token    : the array of token.
//          * idx      : the index (we consider token[idx], ..., token[idx+11])
// 
//     - Output
//          * pair (byte-hashed value of tokens, idx)
// =====================================================================================================================
pub fn compress_build(token: &ArrayView1<u32>, idx: usize) -> [u64; 5] {
	[
		((token[idx + 1] as u64) << 24) | ((token[idx + 2] as u64) <<  4) | ((token[idx + 3] as u64) >> 16) | ((token[idx + 0] as u64) << 44),
		((token[idx + 4] as u64) << 28) | ((token[idx + 5] as u64) <<  8) | ((token[idx + 6] as u64) >> 12) | ((token[idx + 3] as u64 & 65535) << 48),
		((token[idx + 7] as u64) << 32) | ((token[idx + 8] as u64) << 12) | ((token[idx + 9] as u64) >>  8) | ((token[idx + 6] as u64 &  4095) << 52),
		((token[idx +10] as u64) << 36) | ((token[idx +11] as u64) << 16) | ((token[idx + 9] as u64 &  255) << 56),
		idx as u64
	]
}



// ====================================================================================================================
// 2. Convert length-12 tokens to byte-hashed values
//     - Input
//          * seq  : pattern sequence whose length is 12 or less
//          * modes: whether we take sequence hash of "lower_bound" (0) or "upper_bound" (1)
// 
//     - Output
//          * byte-hashed value of tokens
// ====================================================================================================================
pub fn compress(seq: &[u32], modes: u32) -> [u64; 4] {
	if seq.len() == 0 {
		if modes == 1 {
			[18446744073709551615u64, 0, 0, 0]
		}
		else {
			[0, 0, 0, 0]
		}
	}
	else {
		let mut token: [u32; 12] = [0; 12];
		for i in 0..seq.len() {
			if i + 1 == seq.len() && modes == 1 {
				token[i] = seq[i] + 1;
			}
			else {
				token[i] = seq[i];
			}
		}
		[
			((token[ 1] as u64) << 24) | ((token[ 2] as u64) <<  4) | ((token[ 3] as u64) >> 16) | ((token[ 0] as u64) << 44),
			((token[ 4] as u64) << 28) | ((token[ 5] as u64) <<  8) | ((token[ 6] as u64) >> 12) | ((token[ 3] as u64 & 65535) << 48),
			((token[ 7] as u64) << 32) | ((token[ 8] as u64) << 12) | ((token[ 9] as u64) >>  8) | ((token[ 6] as u64 &  4095) << 52),
			((token[10] as u64) << 36) | ((token[11] as u64) << 16) | ((token[ 9] as u64 &  255) << 56)
		]
	}
}



// ====================================================================================================================
// 3. Get an integer from byte-hashed values
//     - Input
//          * seq: byte-hashed sequence
//          * pos: position that we want to get (0 <= pos < 13)
// 
//     - Output
//          * get the pos-th value of the hashed sequence
// ====================================================================================================================
pub fn retrieve_value(seq: &[u64], pos: usize) -> u32 {
	if pos == 0 {
		((seq[0] >> 44) & 1048575) as u32
	}
	else if pos == 1 {
		((seq[0] >> 24) & 1048575) as u32
	}
	else if pos == 2 {
		((seq[0] >> 4) & 1048575) as u32
	}
	else if pos == 3 {
		(((seq[0] & 15) << 16) + (seq[1] >> 48)) as u32
	}
	else if pos == 4 {
		((seq[1] >> 28) & 1048575) as u32
	}
	else if pos == 5 {
		((seq[1] >> 8) & 1048575) as u32
	}
	else if pos == 6 {
		(((seq[1] & 255) << 12) + (seq[2] >> 52)) as u32
	}
	else if pos == 7 {
		((seq[2] >> 32) & 1048575) as u32
	}
	else if pos == 8 {
		((seq[2] >> 12) & 1048575) as u32
	}
	else if pos == 9 {
		(((seq[2] & 4095) << 8) + (seq[3] >> 56)) as u32
	}
	else if pos == 10 {
		((seq[3] >> 36) & 1048575) as u32
	}
	else if pos == 11 {
		((seq[3] >> 16) & 1048575) as u32
	}
	else {
		(seq[3] & 65535) as u32
	}
}



// =====================================================================================================================
// 4. Convert duration to string
//     - Input
//          * secs: the remaining seconds
// 
//     - Output
//          * the string of remaining time (e.g. 52s, 17m)
// =====================================================================================================================
pub fn format_eta(secs: u64) -> String {
	if secs < 60 {
		format!("{}s", secs)
	}
	else if secs < 300 {
		format!("{}m{}s", secs / 60, ((secs % 60) / 10) * 10)
	}
	else if secs < 3600 {
		format!("{}m", secs / 60)
	}
	else {
		format!("{}h{}m", secs / 3600, (secs / 60) % 60)
	}
}



// ====================================================================================================================
// 5. Output the merged similarity of "similarity a" & "similarity b"
//     - Input
//          * a: the first similarity
//          * b: the second similarity
//
//     - Output
//          * the merged similarity
// ====================================================================================================================
pub fn softmin(a: f32, b: f32) -> f32 {
	let alpha = 1.0e4_f32;
	let sum = (alpha).powf(1.0 - a) + (alpha).powf(1.0 - b) - 1.0;
	1.0 - sum.log2() / (alpha).log2()
}



// ====================================================================================================================
// 6. Return values from [start, end) bytes in a binary file
//     - Input
//          * file : the path to file
//          * start: the starting byte-offset in the file
//          * end  : the ending byte-offset in the file
// 
//     - Output
//          * the values in [stt, end) bytes in this file as u8
// ====================================================================================================================
pub fn read_from_file(file: &File, start: usize, end: usize) -> Vec<u8> {
	let mut buf = vec![0u8; end - start];
	if let Err(_) = file.read_at(&mut buf, start as u64) {
		eprintln!("read_at failed at mid");
	}
	return buf
}



// ====================================================================================================================
// 7. Convert u8 array to u32 array
//     - Input
//          * buf: a u8 array of length 32
// 
//     - Output
//          * a u64 array of length 8
// ====================================================================================================================
pub fn get_array4(buf: &Vec<u8>, pos: usize) -> [u64; 4] {
	return [
		u64::from_le_bytes(buf[pos * 32 +  0.. pos * 32 +  8].try_into().unwrap()),
		u64::from_le_bytes(buf[pos * 32 +  8.. pos * 32 + 16].try_into().unwrap()),
		u64::from_le_bytes(buf[pos * 32 + 16.. pos * 32 + 24].try_into().unwrap()),
		u64::from_le_bytes(buf[pos * 32 + 24.. pos * 32 + 32].try_into().unwrap()),
	];
}



// ====================================================================================================================
// 8. Get upper-right convex in two-dimensional points
//     - Input
//          * lis: list of pairs (x, y)
//
//     - Output
//          * deletes the elements where both x & y are smaller than others
// ====================================================================================================================
pub fn get_upper_convex(lis: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
	let mut ret: Vec<(f32, f32)> = Vec::new();
	for i in 0..lis.len() {
		let mut flg: bool = true;
		for j in 0..lis.len() {
			if i != j {
				if lis[j].0 == lis[i].0 && lis[j].1 == lis[i].1 {
					if j < i {
						flg = false;
					}
				}
				else {
					if lis[j].0 >= lis[i].0 && lis[j].1 >= lis[i].1 {
						flg = false;
					}
				}
			}
		}
		if flg == true {
			ret.push(lis[i]);
		}
	}
	ret
}



// ====================================================================================================================
// 9. Check b[] is a subsequence of a[]
//     - Input
//          * a: pattern sequence A
//          * b: pattern sequence B
//
//     - Output
//          * whether B is a subsequence of A
// ====================================================================================================================
pub fn check_subsequence(a: Vec<u32>, b: Vec<u32>) -> bool {
	if a.len() < b.len() {
		false
	}
	else {
		let mut ans: bool = false;
		for i in 0..(a.len() - b.len() + 1) {
			let mut flg: bool = true;
			for j in 0..b.len() {
				if a[i + j] != b[j] {
					flg = false;
				}
			}
			if flg == true {
				ans = true;
			}
		}
		ans
	}
}


// ====================================================================================================================
// 10. The function for binary searching array
//     - Input
//          * arr: sequence whose data is (token_id, position)
//          * tkn: the token id
// 
//     - Output
//          * if "token_id = tkn" exists in "arr", return the lowest "position"
//          * otherwise, return -1
// ====================================================================================================================
pub fn bsearch(arr: &Vec<(u32, i32)>, tkn: u32) -> i32 {
	let mut ng: i32 = -1;
	let mut ok: i32 = arr.len() as i32;
	while ok - ng > 1 {
		let mid: i32 = (ng + ok) / 2;
		if arr[mid as usize].0 >= tkn {
			ok = mid;
		}
		else {
			ng = mid;
		}
	}
	if (ok as usize) == arr.len() || arr[ok as usize].0 != tkn {
		-1
	}
	else {
		arr[ok as usize].1
	}
}