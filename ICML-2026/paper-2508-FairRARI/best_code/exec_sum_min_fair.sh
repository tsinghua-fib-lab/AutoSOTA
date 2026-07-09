dt_name_vec='blogs erdos twitchPTBR twitchRU twitchES twitchFR twitchENGB twitchDE dblp_gender dblp_pub twitter deezer github twitch_gamers'

for dt_name in $dt_name_vec
do
    for phi in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        python run_sum_min_fair_FairRARI.py --dataset-name "$dt_name" --phi $phi --max-iters 1000
        python run_sum_min_fair_post_processing.py --dataset-name "$dt_name" --phi $phi --max-iters 1000
    done
done
