data_dirs=(qasper_e passage_count_e triviaqa_e lcc_e repobench-p_e trec_e samsum_e 2wikimqa_e gov_report_e hotpotqa_e multi_news_e multifieldqa_en_e passage_retrieval_en_e)
presses=(no_press streaming_llm pyramidkv balance_kv uniform snapkv compress_kv_12)
for device in {0..3}; do
  for index in "${!data_dirs[@]}"
  do
    if [ $((index % 4)) -eq $device ]; then
    # compression_ratio = fraction of points NOT preserved
    for compression_ratio in 0.75 0.875 0.9375; do
      for press in "${presses[@]}"
      do
        CMD="export CUDA_VISIBLE_DEVICES=$device; python evaluate.py --config_file evaluate_config.yaml --data_dir ${data_dirs[$index]} --press_name $press --compression_ratio $compression_ratio"
        echo $CMD
        eval $CMD
      done
    done
    fi
  done &
done