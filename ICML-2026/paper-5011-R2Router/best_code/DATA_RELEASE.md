# Data Release Plan

This repository intentionally does not commit the full training and evaluation artifacts.

For public release, split data into three buckets:

1. Redistributable artifacts
   - category labels
   - train/test split files
   - query embeddings if licensing permits
   - per-query per-model per-budget accuracy labels
   - per-query per-model per-budget token statistics

2. Third-party source data that may require separate download
   - RouterArena source JSON files
   - benchmark prompts and answers inherited from external datasets
   - model pricing tables maintained by another repository

3. Reconstructed artifacts
   - `training_data.pkl`
   - cached sweep summaries
   - checkpoint-ready feature matrices

Recommended public release pattern:

- Code on GitHub
- Large data on Hugging Face Datasets or Zenodo
- This repository stores small metadata files and reconstruction scripts only

Minimum documentation each released artifact should include:

- source
- license
- creation command
- checksum
- schema
- whether it can be redistributed directly
