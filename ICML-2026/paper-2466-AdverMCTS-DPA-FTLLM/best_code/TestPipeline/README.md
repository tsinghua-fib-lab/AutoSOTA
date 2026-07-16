# Text Cleaning Pipeline Reproductions

This repository provides local, single-machine reproductions of several widely used
web-scale text cleaning pipelines. The goal is to make their core document-level
processing stages executable on plain-text JSONL inputs, without requiring
Common Crawl metadata, WARC/WET files, or distributed infrastructure.

The implemented pipelines are derived from:
- C4
- CCNet
- The Pile
- RedPajama
- FineWeb-2

Our implementation focuses exclusively on **natural-language text cleaning and
near-duplicate detection**, and omits all components that fundamentally depend on
web metadata or large-scale distributed execution.

For a comprehensive explanation of the data preparation pipeline and its experimental setup, please refer to:
Description and Technical Details of the Data Preparation Pipeline Experiments.pdf

**We recommend running these scripts on Google Colab.**
Some components require installing a specific NumPy version due to library dependency constraints, which may need a runtime restart after installation.

---

## Design Principles

All pipelines follow the same design principles:

1. **Plain-text input**  
   All pipelines operate directly on JSONL documents containing raw text fields.

2. **Single-machine execution**  
   Distributed frameworks (Apache Beam, JSONQL, Datatrove, Slurm) are replaced by
   local in-memory implementations.

3. **Faithful logic, simplified infrastructure**  
   Whenever possible, original implementations are reused. When this is not
   feasible, we reconstruct equivalent logic with local data structures.

4. **Non-web setting**  
   All components that require URLs, HTML, WARC/WET metadata, or domain lists
   are explicitly omitted.

---

## C4

The original C4 pipeline operates on Common Crawl WET files using Apache Beam.
We reproduce only the document-level cleaning components implemented in
`c4_utils.py`.

### Reused components
- `clean_page`
- `is_valid_length`
- `detect_english`
- English branch of `get_badwords_filter_fn`

These functions operate purely on text and are reused without modification.

### Reimplemented components
- Line-level deduplication using local MD5 hashing

### Omitted components
- WET/WARC parsing
- URL canonicalization and domain filtering
- Beam-based global deduplication
- TFDS serialization

---

## CCNet

The original CCNet pipeline assumes access to sharded Common Crawl data and
JSONQL-based distributed execution.

### Reused components
- `normalize_for_dedup`
- fastText language identification
- SentencePiece tokenization
- KenLM perplexity scoring
- `PerplexityBucket`

### Reimplemented components
- Global deduplication using in-memory SHA-1 sets
- Tokenized-field cleanup
- Corpus-specific cutoff generation

### Omitted components
- CCShardReader and WARC ingestion
- Shard-level regrouping
- Metadata-based routing
- Distributed execution

---

## The Pile

The public Pile repository provides only limited dataset-specific cleaning utilities.
Most preprocessing steps described in the original paper are not available in code.

### Reused components
- `strip_markdown_colons`
- `remove_advertisement`

### Reimplemented components
- jusText boilerplate removal
- Unicode normalization via `ftfy`
- English-language filtering using `pycld2`
- Global deduplication

All cleaning steps are applied uniformly across all inputs.

---

## RedPajama

RedPajama is built on CCNet-processed inputs and relies on distributed worker
execution and auxiliary artifacts.

### Reused components
- `Document` class and normalization logic
- Token n-gram extraction
- Quality signal modules
- Official MinHash implementation

### Reimplemented components
- LSH bucketing and clustering (local version)
- Connected-component deduplication

### Omitted components
- CCNet-specific readers
- Classifier and DSIR signals
- Sharded signal writers
- Distributed execution

---

## Dolma

Dolma is a non-destructive annotation framework.

### Behavior
- No text modification is performed
- All taggers operate via span-level annotations
- The original text is preserved exactly

Therefore, Dolma is included only as a reference and is not reimplemented.

---

## FineWeb-2

FineWeb-2 assumes WARC ingestion and Datatrove-based distributed processing.

### Reused components
- MinHash deduplication stages
- Gopher repetition and quality filters
- FineWeb quality filters
- Text normalization formatters

### Reimplemented components
- Language identification using fastText (FT-176)
- Local execution of all MinHash stages

### Omitted components
- WARC ingestion
- HTML extraction
- URL filtering
- GlotLID multilingual routing
- Slurm-based execution

---

## Code Organization

Each pipeline is implemented as an independent .ipynb file.


