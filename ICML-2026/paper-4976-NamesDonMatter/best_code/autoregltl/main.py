import argparse
import sys


def add_dataset_arguments(parser):
    group = parser.add_argument_group("Dataset Arguments")
    group.add_argument('--data-dir', default='data-prop', help='directory of datasets')
    group.add_argument('--ds-name', default='ltl-35', help='Name of the dataset to use')
    group.add_argument('--max-trace-length', type=int, default=-1, help='Maximum length of a trace. Samples exceeding this will be filtered out')
    group.add_argument('--min-aps', type=int, help='Minimum number of APs in a formula')
    group.add_argument('--max-aps', type=int, help='Maximum number of APs in a formula')
    group.add_argument('--exact-aps', type=int, help='Exact number of APs in a formula, overrides min-aps and max-aps')
    group.add_argument('--vocab-aps', type=int, help='Override the number of APs in the vocabulary')


def add_training_arguments(parser):
    add_dataset_arguments(parser)

    group = parser.add_argument_group("Training Arguments")
    group.add_argument("--val-split", type=str, default="val")
    # Train config
    group.add_argument("--learning-rate", type=float, default=1e-3)
    group.add_argument("--lr-scheduler-type", type=str, default="cosine")
    group.add_argument("--warmup-steps", type=int, default=1000)
    # Optional AdamW command line arguments
    group.add_argument('--weight-decay', type=float, default=0.1)
    group.add_argument('--adam-beta1', type=float, default=0.9)  
    group.add_argument('--adam-beta2', type=float, default=0.95)
    group.add_argument('--max-grad-norm', type=float, default=1.0)
    # Batch size and steps
    group.add_argument("--epochs", type=int, default=60)
    group.add_argument("--batch-size", type=int, default=512)
    group.add_argument("--eval-batch-size", type=int, default=512)
    group.add_argument("--grad-acc-steps", type=int, default=1)
    group.add_argument("--logging-steps", type=int, default=500)
    group.add_argument("--eval-steps", type=int, default=3000, help="Eval and save every X steps (default: 3000)")
    group.add_argument("--train-max-samples", type=int, help="Maximum samples for the training set")
    group.add_argument("--val-max-samples", type=int, help="Maximum samples for the validation set")
    group.add_argument("--trace-max-samples", type=int, default=100, help="Maximum samples for trace evaluation")
    # Misc
    group.add_argument('--dry', action='store_true', default=False, help='print parameter count and exit')
    group.add_argument('--eval', action='store_true', default=False, help='evaluate on the validation set and exit')
    group.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    group.add_argument("--resume-from", type=str, help="Path to a checkpoint to resume training from")

    group.add_argument("--loss-fct", type=str, help="Loss function, cross entropy by default")


def add_eval_arguments(parser):
    add_dataset_arguments(parser)

    group = parser.add_argument_group("Evaluation Arguments")
    group.add_argument('--split', default='val', help='which dataset split to use for evaluation')
    group.add_argument("--max-samples", type=int)

    group.add_argument("--max-length", type=int, default=100, help="Maximum length of the generated LTL formula")
    # group.add_argument("--batch", type=int, default=1)
    group.add_argument("--result-dir-name", type=str)

    group.add_argument('--eval-threads', type=int, help="Number of threads used while trace checking")
    group.add_argument('--eval-timeout', type=int, default=30, help="Timeout before the trace checking for a single trace is terminated in seconds")

    group.add_argument('--syntax-enforcing', '--se', action='store_true', default=False, help='Enable syntax enforcing')
    group.add_argument('--load-non-se', action='store_true', default=False, help='Load the results without syntax enforcing instead of starting from scratch')

    group.add_argument('--max-perm', type=int, help="Maximum number of permutations in resymbolize evaluation")


def add_embed_arguments(parser):
    group = parser.add_argument_group("Embedder Arguments", "Only applicable if using a decoder-only model or --merged-vocab.")
    group.add_argument("--d_ap", type=int, default=0)
    group.add_argument("--embed-base-normalization", type=str, default="l2")
    group.add_argument("--embed-ap-normalization", type=str, default="l2")
    group.add_argument("--embed-final-normalization", type=str, default="l2")
    group.add_argument("--feature-normalization", type=str, default="disabled", help="Normalization before the projection matrix")
    group.add_argument("--embed-scaling", type=str)


def add_ted_arguments(parser):
    group = parser.add_argument_group("Transformer Encoder-Decoder Model Arguments")
    group.add_argument('--num-heads', type=int, default=4)
    group.add_argument('--d-embed-enc', type=int, default=128, help="Embedding dimension of the encoder")
    group.add_argument('--d-embed-dec', type=int, default=None, help="Embedding dimension of the decoder (equal to d-embed-enc by default)")
    group.add_argument('--d-ff', type=int, default=512)
    group.add_argument('--ff-activation', default='relu')
    group.add_argument('--num-layers', type=int, default=4)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--layer-norm-eps', type=float, default=1e-6, help='Epsilon value used in layer norm')
    group.add_argument("--enc-pe", type=str, default='sinusoid', help="Encoder's positional embedding type")
    group.add_argument("--dec-pe", type=str, default='sinusoid', help="Decoder's positional embedding type")
    group.add_argument('--no-pe-cross-keys', action='store_true', default=False, help="When RoPE is enabled, don't use RoPE for cross-attention keys")
    group.add_argument('--tree-pos-enc', action='store_true', default=False, help='use tree positional encoding')
    group.add_argument('--cross-attn', type=str, default="", help="Cross attention configuration: supports 'per', 'agg', 'per-agg', 'agg-per', or ''")
    group.add_argument('--no-enc-agg', action='store_true', default=False, help='Disable aggregation attention in encoder')
    group.add_argument('--no-dec-agg', action='store_true', default=False, help='Disable aggregation attention in decoder')
    group.add_argument('--no-enc-per', action='store_true', default=False, help='Disable per-stream self-attention in encoder')
    group.add_argument('--no-dec-per', action='store_true', default=False, help='Disable per-stream self-attention in decoder')
    add_embed_arguments(parser)

def add_ted_gen_arguments(parser):
    group = parser.add_argument_group("Transformer Encoder-Decoder Generation Arguments")
    # Beam search
    group.add_argument('--alpha', type=float, default=1.0)
    group.add_argument('--beam-size', type=int, default=1)
    group.add_argument("--gen-batch-size", type=int, default=64)


def apply_seed(seed):
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Not used but just to make sure
    random.seed(seed)
    np.random.seed(seed)
    print("Manual Seed:", seed)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/default")
    parser.add_argument("--device", help="Device to use for training (default: cuda if available)")
    parser.add_argument('--seed', type=int, help='Seed for the random number generator')

    subparsers = parser.add_subparsers(dest="subparser", required=True)

    for model_type in ["ted"]:
        train_parser = subparsers.add_parser(f"train-{model_type}")
        add_training_arguments(train_parser)
        globals()[f"add_{model_type}_arguments"](train_parser)
        eval_parser = subparsers.add_parser(f"eval-{model_type}")
        add_eval_arguments(eval_parser)
        globals()[f"add_{model_type}_gen_arguments"](eval_parser)
        eval_parser = subparsers.add_parser(f"resym-eval-{model_type}")
        add_eval_arguments(eval_parser)
        globals()[f"add_{model_type}_gen_arguments"](eval_parser)

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    if args.seed is not None:
        apply_seed(args.seed)

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print(f"Using device: {args.device}")

    action, model_type = args.subparser.rsplit('-', 1)
    args.action = action
    args.model_type = model_type

    if model_type == "ted":
        from autoregltl import ted
        create_model = ted.create_model
        trainer_cls = ted.TedTrainer
        load_model = ted.load_model
        get_gen_args = ted.get_gen_args
        args.decoder_only = False
    else:
        print("Unknown model type in command:", model_type)
        sys.exit(1)

    if action == "train":
        from autoregltl.train import train
        train(load_model=load_model, create_model=create_model, trainer_cls=trainer_cls, args=args)
    elif action == "eval":
        from autoregltl.eval import evaluate
        evaluate(load_model=load_model, get_gen_args=get_gen_args, args=args)
    elif action == "resym-eval":
        from autoregltl.resymbolize_eval import resymbolize_eval
        resymbolize_eval(load_model=load_model, get_gen_args=get_gen_args, args=args)
    else:
        print("Unknown action:", action)
        sys.exit(1)
