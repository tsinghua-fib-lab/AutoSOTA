import argparse
import os
import pickle

import numpy as np
import torch


class Utils:
    _MAX_NUM_BLOCKS = 10000

    def __init__(self, args):
        self._seed = args.seed
        self._block_size = args.watermark_block_size
        self._embedding_dim = args.watermark_embedding_dim

        self._secret_bit_sequence_path = f'watermark/secret_key/secret_bit_sequence_{self._block_size}.pkl'
        self._secret_vectors_path = f'watermark/secret_key/secret_vectors_{self._block_size}_{self._embedding_dim}.pkl'

    # Secret Key Related
    @property
    def secret_bit_sequence(self):
        if not os.path.exists(self._secret_bit_sequence_path):
            raise FileNotFoundError("Secret bit sequence not found. Please generate it first.")
        
        with open(self._secret_bit_sequence_path, 'rb') as f:
            secret_bit_sequence = pickle.load(f)
        return secret_bit_sequence
    
    @property
    def secret_vectors(self):
        if not os.path.exists(self._secret_vectors_path):
            raise FileNotFoundError("Secret vectors not found. Please generate them first.")

        with open(self._secret_vectors_path, 'rb') as f:
            secret_vectors = pickle.load(f)
        return secret_vectors
    
    def _gen_secret_bit_sequence(self):
        low = 0
        high = 2 ** self._block_size - 1
        all_possible_values = [i for i in range(low, high + 1)]

        rng = np.random.default_rng(self._seed)
        rng.shuffle(all_possible_values)
        print(f"Shuffled values: {all_possible_values}")
        # make them bianry
        bit_sequence = []
        for value in all_possible_values:
            bits = [(value >> i) & 1 for i in range(self._block_size)]
            bits.reverse()
            bit_sequence.append(bits)

        secret_bit_sequence = []
        while len(secret_bit_sequence) < self._MAX_NUM_BLOCKS:
            secret_bit_sequence += bit_sequence
        
        return secret_bit_sequence

    def _gen_secret_vectors(self):
        rng = np.random.default_rng(self._seed) 
        A = rng.standard_normal((self._embedding_dim, self._block_size))
        Q, _ = np.linalg.qr(A)  
        return torch.tensor(Q.T)

    def _gen_and_save_secrets(self):
        secret_bit_sequence = self._gen_secret_bit_sequence()
        secret_vectors = self._gen_secret_vectors()

        with open(self._secret_bit_sequence_path, 'wb') as f:
            pickle.dump(secret_bit_sequence, f)
        
        with open(self._secret_vectors_path, 'wb') as f:
            pickle.dump(secret_vectors, f)

        print("Secret bit sequence and vectors generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--watermark_block_size', type=int, default=16, help='Block size for watermarking')
    parser.add_argument('--watermark_embedding_dim', type=int, default=768, help='Dimension of embeddings')

    args = parser.parse_args()
    utils = Utils(args)

    print(utils.secret_vectors)
    print(len(utils.secret_bit_sequence))

