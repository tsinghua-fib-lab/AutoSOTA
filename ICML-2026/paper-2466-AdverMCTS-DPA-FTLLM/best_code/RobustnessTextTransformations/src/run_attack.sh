#!/bin/bash

# run text transformation for homoglyph baseline
python perturb_watermarked_homo.py

# analyze text transformation results for homoglyph baseline
python homo_perturbed_word_summary.py

# run text transformation for our invisible watermark
python perturb_watermarked_jsonl.py

# analyze text transformation results for our invisible watermark
python wm_perturbed_summary.py