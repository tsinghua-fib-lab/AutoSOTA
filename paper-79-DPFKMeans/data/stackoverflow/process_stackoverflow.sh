#!/bin/bash

# Run the following commands in order to preprocess the stackoverflow dataset, ready to be
# used in run.py. To speed up download_preprocess.py and get_topic_to_user_dict.py can
# be parallelized by setting setting NJPS to and n and running with --job_id in [0, 3*n]
# in parallel.

NJPS=${1:-1}
#python download_preprocess.py --job_id 0 --num_jobs_per_split $NJPS
#python download_preprocess.py --job_id 1 --num_jobs_per_split $NJPS
#python download_preprocess.py --job_id 2 --num_jobs_per_split $NJPS

python -u download.py
wait

mkdir -p logs
for ((i=0; i<3*NJPS; i++)); do
    python -u preprocess.py --job_id $i --num_jobs_per_split $NJPS > logs/preprocess_$i.log 2>&1 &
done
wait

#python get_topic_to_user_dict.py --job_id 0 --num_jobs_per_split $NJPS
#python get_topic_to_user_dict.py --job_id 1 --num_jobs_per_split $NJPS
#python get_topic_to_user_dict.py --job_id 2 --num_jobs_per_split $NJPS

for ((i=0; i<3*NJPS; i++)); do
    python -u get_topic_to_user_dict.py --job_id $i --num_jobs_per_split $NJPS > logs/topics_$i.log 2>&1 &
done
wait

python -u extract_users.py --num_jobs_per_split $NJPS


