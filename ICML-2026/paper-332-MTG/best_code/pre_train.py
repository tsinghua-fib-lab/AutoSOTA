import argparse
from prompt_graph.defines import GRAPH_TASKS, NODE_TASKS
from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE, NodePrePrompt, GraphPrePrompt, \
    DGI, GraphMAE
from prompt_graph.utils import seed_everything
from prompt_graph.utils import get_args
from prompt_graph.data import load4graph


def get_pretrain_task_by_dataset_name(dataset_name: str) -> str:
    if dataset_name in GRAPH_TASKS:
        return "GraphTask"
    elif dataset_name in NODE_TASKS:
        return "NodeTask"
    else:
        raise ValueError(f"Does not support this kind of dataset {dataset_name}.")


def get_pretrain_task_delegate(args: argparse.Namespace):
    seed_everything(args.seed)
    if args.pretrain_task == 'SimGRACE':
        pt = SimGRACE(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer,
                      num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'GraphCL':
        pt = GraphCL(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer,
                     num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'Edgepred_GPPT':
        pt = Edgepred_GPPT(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim,
                           gln=args.num_layer, num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task == 'Edgepred_Gprompt':
        pt = Edgepred_Gprompt(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim,
                              gln=args.num_layer, num_epoch=args.epochs, device=args.device,
                              num_workers=args.num_workers)
    elif args.pretrain_task == 'DGI':
        pt = DGI(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer,
                 num_epoch=args.epochs, device=args.device, num_workers=args.num_workers)
    elif args.pretrain_task in ('NodeMultiGprompt', 'MultiGprompt', 'GraphMultiGprompt'):
        if args.pretrain_task == "NodeMultiGprompt" or args.dataset_name in NODE_TASKS:
            nonlinearity = 'prelu'
            pt = NodePrePrompt(args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3,
                               device=args.device)
        elif args.pretrain_task == 'GraphMultiGprompt' or args.dataset_name in GRAPH_TASKS:
            # TODO: Bugged unknown parameters: graph_list, input_dim, out_dim
            nonlinearity = 'prelu'
            graph_list, input_dim, out_dim = load4graph(args.dataset_name, pretrained=True)
            pt = GraphPrePrompt(graph_list, input_dim, out_dim, args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9,
                                0.1, 1, 0.3, device=args.device)
        else:
            raise ValueError(f"Unsupported args.pretrain_task type for MultiGprompt {args.pretrain_task}")
    elif args.pretrain_task == 'GraphMAE':
        pt = GraphMAE(dataset_name=args.dataset_name, gnn_type=args.gnn_type, hid_dim=args.hid_dim, gln=args.num_layer,
                      num_epoch=args.epochs, device=args.device,
                      mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2,
                      num_workers=args.num_workers)
    else:
        raise ValueError(f"Unexpected args.pretrain_task type: {args.pretrain_task}.")
    return pt


if __name__ == "__main__":
    args = get_args()

    pt = get_pretrain_task_delegate(args=args)
    pt.pretrain()