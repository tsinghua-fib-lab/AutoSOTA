import os
import time
from pathlib import Path
from utils import get_ptq_arguments, get_aespa_weight_quant_infos, save_ppl_results, set_qmodel_dir
from model_utils import get_model
from data_utils import get_weight_quant_data
from quantize import aespa_fwrd
from eval_utils import evaluate
import pandas as pd

if __name__ == '__main__':
    args = get_ptq_arguments()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # load the model to be quantized
    model = get_model(args.model_path)
    model.seqlen = args.seqlen
    model.eval()
    args.model_name = args.model_path.split('/')[-1]
    args.model_type = args.model_name.split('-')[0]

    # quantization
    if args.w_bits < 16:
        # load calibration dataset
        calib_data = get_weight_quant_data(args)

        # quantize
        qconfigs, aespa_opts, hyperparams = get_aespa_weight_quant_infos(args)
        print("Start quantization")
        tick = time.time()
        quantizers = aespa_fwrd(model, calib_data, qconfigs, aespa_opts, hyperparams)
        process_time = round(time.time() - tick, 3)
        print(f"Quantization processing time: {process_time}")

        # evaluate
        print(args)
        print('ppl of the quantized model')
        ppl_results = evaluate(model, args)
        save_ppl_results(ppl_results, process_time, args)

        # save the quantized model
        if args.save_model:
            parent_dir = "qmodels"
            qmodel_dir = set_qmodel_dir(args)
            model.save_pretrained(os.path.join(parent_dir, qmodel_dir))
        
        # save the quantization results using excel
        result_dict = {**ppl_results, **vars(args), 'process_time': process_time}
        df = pd.DataFrame([result_dict])
        df.to_csv('quantization_results.csv', mode='a', header=not os.path.exists('quantization_results.csv'), index=False)
        
            