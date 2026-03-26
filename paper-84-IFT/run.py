from main import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time Series Forecasting")

    # Seed CFG
    parser.add_argument('--seed', type=int, default=0)

    # Init CFG
    parser.add_argument('--phase', type=int, default=-1)
    parser.add_argument('--model', type=str, default='IFT')

    # File CFG
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='datasets/ETT/ETTh1.csv')
    parser.add_argument('--file_path', type=str, default='checkpoints/')
    parser.add_argument('--delete_checkpoints', action='store_false')

    # Task CFG
    parser.add_argument('--mode', type=str, default='M')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')

    # Forecasting CFG
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--inverse_transform', action='store_true')

    # Model CFG
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)

    parser.add_argument('--d_ff', type=int, default=4096)
    parser.add_argument('--d_model', type=int, default=1024)

    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--cycle', type=int, default=24)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_core', type=int, default=512)
    parser.add_argument('--seg_len', type=int, default=48)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--spectrum_size', type=int, default=720)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)

    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=1)
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--use_revin', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--fourier_norm', type=str, default='ortho')
    parser.add_argument('--network_norm', type=str, default='instance')
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--distil', action='store_false')
    parser.add_argument('--output_attention', action='store_true')

    # Loss CFG
    parser.add_argument('--loss', type=str, default='MAE')
    parser.add_argument('--reduction', type=str, default='mean')
    parser.add_argument('--decay', action='store_true')

    # Optimization CFG
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_scheduler', type=str, default='type1')

    # Other CFG
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--num_plots', type=int, default=10)
    parser.add_argument('--metric1', type=str, default='MSE')
    parser.add_argument('--metric2', type=str, default='MAE')
    parser.add_argument('--models', type=str, default='TimeXer SOFTS CycleNet IFT iTransformer TimeMixer FreTS SegRNN PatchTST Crossformer TimesNet DLinear FEDformer Autoformer Informer')
    parser.add_argument('--attn_visualization', action='store_false')
    parser.add_argument('--test_visualization', action='store_false')

    # Parse CFG
    CFG = parser.parse_args()
    if CFG.seed:
        random.seed(CFG.seed)
        np.random.seed(CFG.seed)
        torch.manual_seed(CFG.seed)
    if CFG.mode == 'S':
        CFG.c_out = 1
        CFG.enc_in = 1
        CFG.dec_in = 1

    # Main
    main = Main(CFG)

    if CFG.phase == 0:
        print(CFG)
        print("[PHASE 0] Running...")
        main.train()
        print("=======================================================")
        main.vali()
        main.test()
        update_result(CFG)
        update_table(CFG)
    elif CFG.phase == 1:
        print(CFG)
        print("[PHASE 1] Running...")
        main.test()
        update_result(CFG)
        update_table(CFG)
    else:
        update_result(CFG)
        update_table(CFG)
        print("[TABLE UPDATED]")

    torch.cuda.empty_cache()
    checkpoint = f"{os.path.join(CFG.root_path, CFG.file_path, CFG.model)}/{mark(CFG)}.pt"
    if CFG.delete_checkpoints and os.path.exists(checkpoint):
        os.remove(checkpoint)
