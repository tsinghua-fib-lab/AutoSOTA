#!/usr/bin/env bash
set -euo pipefail

# Directory containing all your .mp4 source videos
SOURCE_DIR="/root/autodl-tmp/aaai_workspace/autodl-fs/benchmark_realm/source_videos"

# Where the cropped/downsized benchmarks will go (relative or absolute)
OUTPUT_BASE="data/benchmark_realm"

# Optional parameters
TARGET_FRAMES=64
TEST_SPLIT=4
list='a00033 a00034 a00035 a00067 a00068 a00069'
list='a00090 a00091 a00092 a00093 a00094 a00095 a00096 a00097 a00098 a00099 a00100'


for video_path in "${SOURCE_DIR}"/*.mp4; do
  video_name=$(basename "$video_path" .mp4)

  # extract the numeric suffix (e.g. "00012" → 12)
  num=${video_name#a}
  # force decimal interpretation, then skip if less than 13
  if ((10#$num < 77)); then
    echo "Skipping ${video_name} (scene < a00013)"
    continue
  fi

  # if ((10#$num > )); then
  #   echo "Skipping ${video_name} (scene < a00013)"
  #   continue
  # fi

  output_root="${OUTPUT_BASE}/${video_name}"
  echo "▶ Processing ${video_name}.mp4 → ${output_root}/"

  python prepare_benchmark.py \
    --video_path "$video_path" \
    --output_root "$output_root" \
    --target_frames "$TARGET_FRAMES" \
    --test_split "$TEST_SPLIT"

  echo "✔ Done ${video_name}"
  echo
done
