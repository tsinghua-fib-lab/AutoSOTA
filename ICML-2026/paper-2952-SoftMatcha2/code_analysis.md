# Code Analysis for SoftMatcha 2 (Paper 2952)

## Evaluation Path
- `eval_ir_v2.py` — main evaluation script
- Uses BeIR/trec-covid dataset from HuggingFace (cached)
- Uses SoftMatcha2 Rust extension via Python bindings
- GloVe embeddings at `/root/gensim-data/glove-wiki-gigaword-300`
- Index at `/repo/index_trec_covid`

## Key Configuration (lines 14-17)
- `ALPHA = 0.45` — similarity threshold for SoftMatcha
- `K = 20` — top-K candidates from SoftMatcha search
- `K1 = 1.2` — BM25 term frequency saturation
- `B = 0.75` — BM25 length normalization

## Metric Parser
- Output lines: `P@20=X.X, R@1000=X.X` after method name
- `record_score.sh` parses these from stdout

## Safe Modification Targets
1. `eval_ir_v2.py` lines 14-17: ALPHA, K, K1, B constants
2. `eval_ir_v2.py` lines 63-68: stop_words set
3. `eval_ir_v2.py` lines 109-130: extract_key_phrases()
4. `eval_ir_v2.py` lines 134-184: evaluate() scoring loop
5. `eval_ir_v2.py` lines 91-101: bm25_score_doc()

## Risky Files (do not modify)
- SoftMatcha2 Rust source (would require rebuild)
- BeIR dataset loading (would change data)
- qrels_data processing (would change labels)
- Metric computation (P@20, R@1000 formulas)

## No Pre-downloaded Paper Data
- No `/paper_data` mount; all data from HuggingFace cache
