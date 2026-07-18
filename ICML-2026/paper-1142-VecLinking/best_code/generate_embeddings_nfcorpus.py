#!/usr/bin/env python3
"""
Generate NFCorpus embeddings for VecLink reproduction.
Uses open-source sentence-transformers models as substitutes for the
paper's API-based models (Mistral-embed and text-embedding-3-small).

Requirements: sentence-transformers, beir, numpy
Run: python3 generate_embeddings_nfcorpus.py
"""
import os
os.environ['HF_HOME'] = '/autosota_cache/hf'

import numpy as np
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

# Download NFCorpus
from beir import util
dataset = 'nfcorpus'
url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
out_dir = '/repo/datasets'
util.download_and_unzip(url, out_dir)

# Load corpus
corpus, queries, qrels = GenericDataLoader(data_folder=f'/repo/datasets/{dataset}').load(split='test')
corpus_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id]['title'] + ' ' + corpus[doc_id]['text'] for doc_id in corpus_ids]

os.makedirs('/repo/embeddings', exist_ok=True)

# Model 1: all-MiniLM-L6-v2 as 'mistral' substitute (384-dim)
m1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
emb1 = m1.encode(corpus_texts, show_progress_bar=True, normalize_embeddings=True)
np.save('/repo/embeddings/corpus_embeddings_mistral_nfcorpus.npy', emb1.astype(np.float32))

# Model 2: all-mpnet-base-v2 as 'openai' substitute (768-dim)
m2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
emb2 = m2.encode(corpus_texts, show_progress_bar=True, normalize_embeddings=True)
np.save('/repo/embeddings/corpus_embeddings_openai_nfcorpus.npy', emb2.astype(np.float32))

print('Embeddings generated successfully.')
print(f'  corpus_embeddings_mistral_nfcorpus.npy: {emb1.shape}')
print(f'  corpus_embeddings_openai_nfcorpus.npy: {emb2.shape}')
