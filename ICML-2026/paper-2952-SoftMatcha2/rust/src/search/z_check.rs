use numpy::ndarray::{ArrayView1};
use crate::search::z_bsearch::search_2;
use crate::search::z_bsearch::search_3;


// ====================================================================================================================
// Check whether the pattern sequence is "obviously not in the corpus" or not
//     - Input
//          * seq      : the pattern sequence
//          * pair_    : the existance of 2-gram
//          * trio_    : the existance of 3-gram
//          * pair_cons: the size of table (top-"pair_cons" words should be considered)
//          * trio_cons: the size of table (top-"trio_cons" words should be considered)
//
//     - Output
//          * false: the pattern sequence is obviously not in the corpus
//          * true : otherwise
// ====================================================================================================================
pub fn check_valid(
	seq: &Vec<u32>,
	pair_: ArrayView1<u64>,
	trio_: ArrayView1<u64>,
	pair_cons: usize,
	trio_cons: usize
) -> bool {
	if seq.len() >= 2 && search_2(pair_, pair_cons, seq[seq.len() - 2], seq[seq.len() - 1]) == 1 {
		false
	}
	else if seq.len() >= 3 && search_3(trio_, trio_cons, seq[seq.len() - 3], seq[seq.len() - 2], seq[seq.len() - 1]) == 1 {
		false
	}
	else {
		true
	}
}