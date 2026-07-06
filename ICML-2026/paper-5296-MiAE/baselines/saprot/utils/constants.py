import itertools


aa_set = {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}
aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

struc_unit = "abcdefghijklmnopqrstuvwxyz"


def create_vocab(size: int) -> dict:
    """

    Args:
        size:   Size of the vocabulary

    Returns:
        vocab:  Vocabulary
    """

    token_len = 1
    while size > len(struc_unit) ** token_len:
        token_len += 1

    vocab = {}
    for i, token in enumerate(itertools.product(struc_unit, repeat=token_len)):
        vocab[i] = "".join(token)
        if len(vocab) == size:
            vocab[i+1] = "#"
            return vocab