import argparse

supported_subgraph_fl_datasets = [
    "PubMed", "CS", "Physics", "Actor",
    "Roman-empire"]

parser = argparse.ArgumentParser()

parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)

parser.add_argument("--root", type=str, default="change_to_your_root_path")

parser.add_argument("--dataset", type=str, default=[], action='append')

parser.add_argument("--dp_eps", type=float, default=1e10)

parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--client_frac", type=float, default=1.0)

parser.add_argument("--louvain_resolution", type=float, default=1)
parser.add_argument("--louvain_delta", type=float, default=20,
                    help="Maximum allowable difference in node counts between any two clients in the graph_fl_louvain simulation.")

parser.add_argument("--train_val_test", type=str,
                    default="default_split")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--weight_decay", type=float, default=5e-4)

parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--hid_dim", type=int, default=64)

parser.add_argument("--dp_mech", type=str, default='no_dp')
parser.add_argument("--grad_clip", type=float, default=1e10)
parser.add_argument("--dp_delta", type=float, default=1e-10)
parser.add_argument("--lambda_reg", type=float, default=0.0)
parser.add_argument("--epsilon", type=float, default=0.0)

parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--log_root", type=str, default=None)
parser.add_argument("--log_name", type=str, default=None)
parser.add_argument("--comm_cost", type=bool, default=False)

parser.add_argument("--features", type=int, default=128)
parser.add_argument("--classes", type=int, default=40)
parser.add_argument("--graph_repair", type=bool, default=True)
parser.add_argument("--S_noi_list", type=list, default=[])
parser.add_argument("--corruption_ratio", type=float, default=0.5)
parser.add_argument("--noise_extent", type=float, default=1)
parser.add_argument("--alpha", type=float, default=0.3)

args, unknown = parser.parse_known_args()
