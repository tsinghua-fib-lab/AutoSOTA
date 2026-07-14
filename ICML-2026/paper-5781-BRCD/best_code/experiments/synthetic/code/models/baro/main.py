from argparse import ArgumentParser, RawTextHelpFormatter
from baro.reproducibility import reproduce_baro, reproduce_bocpd, reproduce_rq4

# Define dataset mappings
DATASET_MAPS = {
    "OnlineBoutique": "fse-ob",
    "SockShop": "fse-ss",
    "TrainTicket": "fse-tt"
}

# Define command-line arguments
parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("--anomaly-detection", action="store_true",
    help="Reproduce anomaly detection results."
)
parser.add_argument("--saved", action="store_true",
    help="Use saved anomaly detection results to reproduce the\n"
         "presented results without rerunning anomaly detection."
)
parser.add_argument("--dataset", type=str, default=None,
    help="Choose a dataset to analyze. Options:\n"
         "['OnlineBoutique', 'SockShop', and 'TrainTicket']."
)
parser.add_argument("--fault-type", type=str, default=None,
    help="Specify the fault type for root cause analysis.\n"
         "Options: ['cpu', 'mem', 'delay', 'loss', and 'all'].\n"
         "If 'all' is selected, the program will run the root\n"
         "cause analysis for all fault types."
)
parser.add_argument("--rq4", action="store_true",
    help="Reproduce RQ4 results.")
parser.add_argument("--eval-metric", type=str, default="avg5",
    help="Evaluation metric for RQ4. Options: ['top1', 'top3',\n"
        "'avg5']. Default: 'avg5'.")

args = parser.parse_args()
if args.dataset not in DATASET_MAPS:
    print(f"Dataset {args.dataset} is not supported! Valid datasets are {list(DATASET_MAPS.keys())}")
    exit(0)
if args.fault_type not in [None, "all", "cpu", "mem", "delay", "loss"]:
    print(f"Fault type {args.fault_type} is not supported! Valid fault types are [None, 'all', 'cpu', 'mem', 'delay', 'loss']")
    exit(0)

if args.anomaly_detection:
    reproduce_bocpd(dataset=DATASET_MAPS[args.dataset], saved=args.saved)
elif args.rq4:
    reproduce_rq4(dataset=DATASET_MAPS[args.dataset], eval_metric=args.eval_metric)
else:
    reproduce_baro(dataset=DATASET_MAPS[args.dataset], fault=args.fault_type)