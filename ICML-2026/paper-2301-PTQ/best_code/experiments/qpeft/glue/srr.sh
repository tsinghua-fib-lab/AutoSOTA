function check_return_value() {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
        exit 1
    fi
}

task_list=("rte" "cola" "stsb" "mrpc" "mnli" "qnli" "sst2" "qqp")
rank=8
quant_type=mxint 
mxint_block_size=32 
quant_bits=4
seed=0
init_method="srr" 
qera_num_iter=1

for task in ${task_list[@]}; do
    bash ./adapt_and_glue_train.sh Anonymous-Pineapple/roberta-base $task qera $rank $quant_type \
    $quant_bits $seed $mxint_block_size $init_method $global_dir $qera_num_iter
    check_return_value "qera $task"
done


