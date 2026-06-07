import argparse


def add_common_runtime_args(parser: argparse.ArgumentParser):
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def teacher_parser():
    parser = argparse.ArgumentParser(description="Train the CoFIDA + MONET teacher.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--target-val-dir", default="")
    parser.add_argument("--monet-csv", required=True)
    parser.add_argument("--save-dir", default="outputs/teacher")
    add_common_runtime_args(parser)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--w-kl-max", type=float, default=0.6)
    parser.add_argument("--w-feat-max", type=float, default=0.10)
    parser.add_argument("--w-edit-max", type=float, default=0.10)
    parser.add_argument("--pos-boost", type=float, default=1.5)
    parser.add_argument("--pseudo-warmup-epochs", type=int, default=5)
    parser.add_argument("--pseudo-t-start", type=float, default=0.95)
    parser.add_argument("--pseudo-t-end", type=float, default=0.70)
    parser.add_argument("--pseudo-t-end-epoch", type=int, default=15)
    parser.add_argument("--lambda-ortho", type=float, default=0.01)
    parser.add_argument("--lambda-norm", type=float, default=0.01)
    parser.add_argument("--r-max", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--conf-gate-pow", type=float, default=1.0)
    parser.set_defaults(use_recall_floor=True)
    parser.add_argument("--no-recall-floor", action="store_false", dest="use_recall_floor")
    parser.add_argument("--mel-recall-floor", type=float, default=0.75)
    return parser


def eval_teacher_parser():
    parser = argparse.ArgumentParser(description="Evaluate the CoFIDA + MONET teacher.")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--monet-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def student_parser():
    parser = argparse.ArgumentParser(description="Train the image-only student.")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--monet-csv", required=True)
    parser.add_argument("--save-dir", default="outputs/student")
    add_common_runtime_args(parser)
    parser.set_defaults(batch_size=32)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--feat-align-w", type=float, default=0.1)
    parser.add_argument("--print-freq", type=int, default=100)
    return parser


def eval_student_parser():
    parser = argparse.ArgumentParser(description="Evaluate the image-only student.")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-map-melanoma", action="store_true")
    return parser


def export_split_parser():
    parser = argparse.ArgumentParser(description="Export the deterministic student train/val split.")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--monet-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser
