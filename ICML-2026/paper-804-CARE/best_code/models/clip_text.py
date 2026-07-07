import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP_Text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


import numpy as np

class GloVe_Text(nn.Module):
    def __init__(self, glove_path='glove.6B.300d.txt'):
        super().__init__()
        self.embedding_dim = 300
        self.glove_dict = self.load_glove(glove_path)

    def load_glove(self, path):
        glove = {}
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove[word] = vector
        return glove

    def encode_text(self, text):
        words = text.lower().split()
        vecs = [self.glove_dict[w] for w in words if w in self.glove_dict]
        if vecs:
            return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)
        else:
            return torch.zeros(self.embedding_dim, dtype=torch.float32)

    def forward(self, text_list):
        embeddings = [self.encode_text(label) for label in text_list]
        return torch.stack(embeddings).to(torch.float32)  # (C, D)

