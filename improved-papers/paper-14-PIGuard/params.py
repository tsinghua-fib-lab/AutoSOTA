import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default='0',
        help="Experiments Name",
    )
    parser.add_argument(
        "--train_set",
        default="datasets/train.json",
        help="The path to the train set file.",
    )
    parser.add_argument(
        "--valid_set",
        default="datasets/valid.json",
        help="The path to the valid set file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default='datasets',
        help="For evaluation, the folder to place test sets.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="the number of training iterations for each sample."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Batch size for eval per GPU."
    )
    parser.add_argument(
        "--display", type=int, default=10, help="The step interval to display."
    )
    parser.add_argument(
        "--save_step", type=int, default=200, help="The step interval to save models."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='logs',
        help="Path to save the model",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="logs",
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="The maximum length of input tokens."
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-08, help="Adam epsilon.")
    parser.add_argument(
        "--warmup", type=int, default=100, help="Number of steps to warmup."
    )
    parser.add_argument("--save_thres", type=float, default=0.8, help="The performance threshold to save models.")
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="The model you want to load. Use None to avoid resume models.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed."
    )
    args = parser.parse_args()

    return args
