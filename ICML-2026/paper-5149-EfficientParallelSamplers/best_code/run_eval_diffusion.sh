#!/bin/bash
# GSM8K evaluation with diffusion forcing sampler for Huginn-0125
set -e
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/autosota_cache/hf
export HF_ENDPOINT=https://hf-mirror.com
MODEL_PATH=${MODEL_PATH:-/models/huginn-0125}
LIMIT=${LIMIT:-200}
echo "=== GSM8K Diffusion Sampler Evaluation ==="
echo "Model: $MODEL_PATH, Limit: $LIMIT"
# Apply to_legacy_cache patch
python3 -c "
import sys
with open(\"${MODEL_PATH}/raven_modeling_minimal.py\", \"r\") as f:
    content = f.read()
if \"def to_legacy_cache\" not in content:
    old = \"        self.lookup_strategy = lookup_strategy\\n\"
    new = \"        self.lookup_strategy = lookup_strategy\\n\\n    def to_legacy_cache(self):\\n        return ()\\n\"
    if old in content:
        content = content.replace(old, new, 1)
        with open(\"${MODEL_PATH}/raven_modeling_minimal.py\", \"w\") as f:
            f.write(content)
        print(\"Patched HuginnDynamicCache\")
"
python3 /repo/eval_gsm8k_runner.py
echo "=== Done ==="
