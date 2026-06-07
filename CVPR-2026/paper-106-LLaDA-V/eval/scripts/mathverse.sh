export OPENAI_API_KEY=YOUR_OPENAI_API_KEY


# no accelerate
accelerate launch --num_processes=8 --main_process_port 23456 -m lmms_eval \
--model llava_onevision_llada \
--gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think"}' \
--model_args pretrained=GSAI-ML/LLaDA-V,conv_template=llava_llada,model_name=llava_llada \
--tasks mathverse_testmini_vision \
--batch_size 1 \
--log_samples \
--log_samples_suffix mathverse_testmini_vision \
--output_path exp/llava_v_eval/LLaDA-V

# fast-dLLM
accelerate launch --num_processes=8 --main_process_port 23456 -m lmms_eval \
--model llava_onevision_llada \
--gen_kwargs='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think","threshold": 1, "prefix_refresh_interval": 32}' \
--model_args pretrained=GSAI-ML/LLaDA-V,conv_template=llava_llada,model_name=llava_llada,use_fast_dllm=true \
--tasks mathverse_testmini_vision \
--batch_size 1 \
--log_samples \
--log_samples_suffix mathverse_testmini_vision \
--output_path exp/llava_v_eval/LLaDA-V