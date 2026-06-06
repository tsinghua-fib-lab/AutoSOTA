import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import os
import psutil
def use_cpus(gpus: list, cpus_per_gpu: int):
        cpus = []
        for gpu in gpus:
            cpus.extend(list(range(gpu* cpus_per_gpu, (gpu+1)* cpus_per_gpu)))
        p = psutil.Process()
        p.cpu_affinity(cpus)
        print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TeCh")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="TeCh",
        help="model name, options: [Autoformer, Medformer, TeCh]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    # model define for baselines
    parser.add_argument("--patch_len", type=int, default=1, help="for cross_channel pacthing")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--t_layer", type=int, default=6, help="num of encoder layers")
    parser.add_argument("--v_layer", type=int, default=6, help="num of encoder layers")
    parser.add_argument("--dropout", type=float, default=0., help="dropout")
    
    # Augmentation
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip0.8,frequency0.,jitter0.,mask0.0,channel0.4,drop0.0",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )
    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=16, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="cosine", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0,1,2,3,4,5,6,7], help="List of GPU indices to use")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )
    # parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # For server 1, set cpus_per_gpu to 12
    #For server 2, set cpus_per_gpu to 24
    use_cpus(gpus=args.gpu_idx, cpus_per_gpu=12)
    
    # print("Args in experiment:")
    # print(args)

    Exp = Exp_Classification
    avg_metrics=[]
    means=[]
    stds=[]
    # 42
    for ii in range(args.itr):
            seed = 42 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # setting record of experiments
            args.seed = seed
            setting = "{}_{}_seed_{}_dm_{}_dp_{}_tl_{}_vl_{}_bs_{}_lr{}_aug_{}_pl_{}".format(
                args.model,
                args.data,
                args.seed,
                args.d_model,
                args.dropout,
                args.t_layer,
                args.v_layer,
                args.batch_size,
                args.learning_rate,
                args.augmentations,
                args.patch_len,
            )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            avg_metrics.append(exp.test(setting))
            torch.cuda.empty_cache()
            
    means=[np.mean([avg_metrics[i][j] for i in range(args.itr)]) for j in ('Accuracy', 'Precision', 'Recall', 'F1', 'AUROC','AUPRC')]
    stds=[np.std([avg_metrics[i][j] for i in range(args.itr)]) for j in ('Accuracy', 'Precision', 'Recall', 'F1', 'AUROC','AUPRC')]
    print(f'Mean accuracy: {means[0]:.4f}, precision: {means[1]:.4f},recall: {means[2]:.4f}, f1: {means[3]:.4f}, AUROC: {means[4]:.4f}, AUPRC: {means[5]:.4f}')
    print(f'Std accuracy: {stds[0]:.4f}, precision: {stds[1]:.4f},recall: {stds[2]:.4f}, f1: {stds[3]:.4f}, AUROC: {stds[4]:.4f}, AUPRC: {stds[5]:.4f}')

