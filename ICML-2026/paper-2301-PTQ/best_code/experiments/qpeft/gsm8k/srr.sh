model_name=meta-llama/Llama-2-7b-hf
adapt_init=qera
lora_rank=64
quant_type=mxint
quant_bits=4
seed=42
mxint_block_size=32
init_method="srr" 
qera_num_iter=1

bash adapt_and_gsm8k_train.sh \
    $model_name \
    $adapt_init \
    $lora_rank \
    $quant_type \
    $quant_bits \
    $seed \
    $mxint_block_size \
    $init_method \
    $qera_num_iter

