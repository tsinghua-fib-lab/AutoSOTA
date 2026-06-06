# Audio-video merge script for SAVVY-Bench data preprocessing - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing"
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.

cd ../aea/aea_processed

for dir in */; do
    id="${dir%/}"
    echo "Processing ID: ${id}"
    mkdir -p "${id}/video_merged"

    ffmpeg \
        -i "${id}/video/${id}.mp4" \
        -i "${id}/audio/${id}.wav" \
        -c:v copy \
        -c:a aac \
        -ar 48000 \
        -ac 2 \
        -map 0:v:0 \
        -map 1:a:0 \
        "${id}/video_merged/${id}.mp4" \
        -y
done