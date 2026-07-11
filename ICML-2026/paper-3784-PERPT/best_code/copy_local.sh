#!/bin/bash

# Check if the environment variable LOCAL_RANK is 0
if [ "$LOCAL_RANK" -eq 0 ]; then
  # Create the directory /data if it doesn't exist
  mkdir -p /data_processing/data/240-mammalian

  # Copy the file /mnt/blob/foo to /data/foo
  cp -r ./data_processing/data/240-mammalian /data_processing/data/240-mammalian
  touch /data_processing/data/240-mammalian/copy_complete.txt
fi

while [ ! -f /data_processing/data/240-mammalian/copy_complete.txt ]; do
  sleep 1
done