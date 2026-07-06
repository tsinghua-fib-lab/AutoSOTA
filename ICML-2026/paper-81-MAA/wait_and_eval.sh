#!/bin/bash
# Wait for model download to complete, then run evaluation
MODEL_DIR="/autosota_cache/models/liuhaotian/llava-v1.6-mistral-7b"

echo "[$(date)] Waiting for model download to complete..."

# Wait for all safetensors files
while true; do
    s1=$(ls "$MODEL_DIR/model-00001-of-00004.safetensors" 2>/dev/null)
    s2=$(ls "$MODEL_DIR/model-00002-of-00004.safetensors" 2>/dev/null)
    s3=$(ls "$MODEL_DIR/model-00003-of-00004.safetensors" 2>/dev/null)
    s4=$(ls "$MODEL_DIR/model-00004-of-00004.safetensors" 2>/dev/null)
    
    if [ -n "$s1" ] && [ -n "$s2" ] && [ -n "$s3" ] && [ -n "$s4" ]; then
        echo "[$(date)] All model shards present!"
        break
    fi
    
    inc=$(find "$MODEL_DIR/.cache" -name "*.incomplete" 2>/dev/null | wc -l)
    size=$(du -sm "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "[$(date)] Waiting... ${size}MB, ${inc} incomplete files"
    sleep 60
done

echo "[$(date)] Model download complete. Starting evaluation..."
bash /repo/run_eval.sh
