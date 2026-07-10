import argparse
import torch
from experiments.exp_forecasting import Exp_Forecast
import random
import numpy as np
import time

if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    """Change the model name"""
    parser = argparse.ArgumentParser(description='TemplateProject')
    
    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ExampleModel', help='model name, options: [ExampleModel]')
    
    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    """Change the root path"""
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    
    # forecasting config
    parser.add_argument('--num_patches', type=int, default=6, help='number of patches')
    parser.add_argument('--seq_len', type=int, default=96, help='input (lookback) window length')
    parser.add_argument('--label_len', type=int, default=0, help='overlap length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction (horizon) length')
    
    # model config
    parser.add_argument('--d_input', type=int, default=1, help='input dimension')
    parser.add_argument('--d_model', type=int, default=128, help='model hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--frequency_bins', type=float, default=1.0, help='percentage of frequency bins to keep in FFT')
    parser.add_argument('--learnable_diagonal', type=int, default=0, help='use learnable diagonal in LeaLa filters')
    
    # optimization
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping max_norm (0=disabled)')
    
    # model task: default is forecast; add more as appropriate for projects
    parser.add_argument('--task', type=str, default='forecast', help='task name, options: [forecast]')
    
    # custom config
    parser.add_argument('--use_prime', type=int, default=0, help='use Prime')
    parser.add_argument('--filter_type', type=int, default=1, help='filter type, options: [1: Full Prime, 2: Lead-Lag Only]')
    parser.add_argument('--idrop', type=float, default=0.0, help='identity dropout rate')
    parser.add_argument('--fredf_loss', type=int, default=0, help='use FreDFLoss')
    parser.add_argument('--horizon_weight_beta', type=float, default=0.0, help='exponential horizon loss weighting beta')
    
    # ablation config
    parser.add_argument('--save_pred', type=int, default=0, help='save predictions')
    parser.add_argument('--filter_ablation', type=int, default=0, help='[0: none, 1: identity, 2: leala, 3: instantaneous, 4: full ReDi]')
    parser.add_argument('--adj_p', type=float, default=0.4, help='adjacency matrix population percentage')
    
    # misc.
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='sampling rate')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args.device = device
    
    print("Args in experiment:")
    print(args)
    
    if args.task == 'forecast':
        exp = Exp_Forecast(args)
        
        if args.is_training:
            model = exp.train()
            
            test_loss_mse, test_loss_mae = exp.test()
            print(f"\nTest Loss (MSE): {test_loss_mse}, Test Loss (MAE): {test_loss_mae}")
        else:
            test_loss_mse, test_loss_mae = exp.test()
            print(f"Test Loss (MSE): {test_loss_mse}, Test Loss (MAE): {test_loss_mae}")
    
    else:
        raise ValueError(f"Invalid task: {args.task}, options: [forecast, imputation]")