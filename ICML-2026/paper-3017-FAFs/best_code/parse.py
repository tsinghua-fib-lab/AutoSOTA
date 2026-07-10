from model import MPNNs

def parse_method(args, n, c, d, device):
    model_name = str(getattr(args, 'model', 'mpnn')).lower()
    if model_name in ('faf', 'fafmlp', 'faf-mlp'):
        from model_faf import FAFMLP
        model = FAFMLP(
            d, args.hidden_channels, c,
            mlp_layers=args.mlp_layers,
            dropout=args.dropout,
            ln=args.ln, bn=args.bn
        ).to(device)
        print("FAFMLP")
    else:
        model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout, 
        heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, ln=args.ln, bn=args.bn, jk=args.jk, gnn = args.gnn).to(device)
    return model
        

def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')

    # own
    parser.add_argument('--project', type=str, default='default-project', help='Wandb project entity')
    parser.add_argument('--project_name', type=str, default='default-project-name', help='Wandb project name')
    parser.add_argument('--info_dir', type=str, default=None, help='Directory to save additional information about the dataset')

    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of layers for MLP in FAF-MLP')
    parser.add_argument('--clip_grad', action='store_true', default=False, help='whether to clip gradients during training')
    parser.add_argument('--swa', action='store_true', default=False, help='use Stochastic Weight Averaging in final training phase')
    parser.add_argument('--swa_start_epoch', type=int, default=2000, help='epoch to start SWA weight averaging')
    parser.add_argument('--use_scheduler', action='store_true', default=False, help='use cosine annealing LR schedule with linear warmup')
    parser.add_argument('--warmup_epochs', type=int, default=125, help='number of linear warmup epochs for LR scheduler')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='which optimizer to use')

    # pca
    parser.add_argument('--pca', action='store_true', default=False, help='whether to use PCA to reduce features (not used currently)')
    parser.add_argument('--pca_before', action='store_true', default=False, help='whether to use PCA before aggregation (not used currently)')
    # scaler
    parser.add_argument('--scaler', type=str, default='none', choices=['standard', 'minmax', 'none'], help='which scaler to use for node features')
    parser.add_argument('--scaler_before', type=str, default=False, help='whether to apply scaler before aggregation (not used currently)')

    # FAF-specific
    parser.add_argument('--multi_agg', action='store_true', default=False, help='whether to use multi-aggregation')
    parser.add_argument('--sum_agg', action='store_true', default=False, help='whether to use sum aggregation')
    parser.add_argument('--mean_agg', action='store_true', default=False, help='whether to use mean aggregation')
    parser.add_argument('--max_agg', action='store_true', default=False, help='whether to use max aggregation')
    parser.add_argument('--std_agg', action='store_true', default=False, help='whether to use std aggregation')
    parser.add_argument('--last_agg', action='store_true', default=False, help='whether to use last neighbor feature as aggregation')
    parser.add_argument('--last_agg_only', action='store_true', default=False, help='whether to use only last neighbor feature as aggregation (overrides other aggregation methods)')
    parser.add_argument('--all_agg', action='store_true', default=False, help='whether to use all neighbor features as aggregation')
    parser.add_argument('--exp_agg', action='store_true', default=False, help='whether to use exponential neighbor aggregation')
    parser.add_argument('--meansumall_agg', action='store_true', default=False, help='whether to use meansum neighbor aggregation')
    parser.add_argument('--mmask_agg', action='store_true', default=False, help='whether to use mmask aggregation')
    # KA
    parser.add_argument('--ka_agg', action='store_true', default=False, help='whether to use KA aggregation')
    parser.add_argument('--ka_order', type=str, default='as_is', choices=['by_src_asc', 'by_src_desc', 'as_is', 'by_edge_id'], help='Order of edges for KA aggregation')
    parser.add_argument('--ka_D_max', type=int, default=None, help='Maximum distance for KA aggregation (None for no limit)')
    parser.add_argument('--ka_truncate', action='store_true', default=False, help='whether to truncate distances larger than ka_D_max to ka_D_max')
    parser.add_argument('--ka_pad_value', type=float, default=0.0, help='Padding value for KA aggregation')
    parser.add_argument('--ka_transform', type=str, default='sigmoid', choices=['identity', 'sigmoid', 'softsign', 'log1p'], help='Transformation for distances in KA aggregation')
    parser.add_argument('--ka_temperature', type=float, default=1.0, help='Temperature for KA aggregation')
    parser.add_argument('--ka_n_bits', type=int, default=16, help='Number of bits for KA aggregation')
    # BIN
    parser.add_argument('--bin_agg', action='store_true', default=False, help='whether to use BIN aggregation')
    parser.add_argument('--bin_num', type=int, default=4, help='Number of bins for BIN aggregation')
    parser.add_argument('--bin_edges', type=float, nargs='+', default=None, help='Edges of bins for BIN aggregation')
    parser.add_argument('--bin_cdf', action='store_true', default=False, help='whether to use CDF for BIN aggregation')
    # SIM
    parser.add_argument('--sim_agg', action='store_true', default=False, help='whether to use SIM aggregation')
    parser.add_argument('--sim_type', type=str, default='mean', choices=['mean', 'max', 'min', 'sum'], help='Type of similarity aggregation for SIM aggregation')
    parser.add_argument('--sim_mode', type=str, default='cosine', choices=['cosine', 'dot', 'rbf'], help='Similarity mode for SIM aggregation')
    parser.add_argument('--sim_slice', type=int, nargs='+', default=None, help='Slice of features to use for SIM aggregation')
    parser.add_argument('--sim_clamp_negatives', action='store_true', default=False, help='whether to clamp negative similarities to zero in SIM aggregation')
    parser.add_argument('--sim_clamp_positives', action='store_true', default=False, help='whether to clamp positive similarities to zero in SIM aggregation')
    parser.add_argument('--sim-normalize', type=str, default='l1', choices=['softmax','l1','none'])
    parser.add_argument('--sim_temperature', type=float, default=1.0, help='Temperature for SIM aggregation')
    parser.add_argument('--sim_eps', type=float, default=1e-8, help='Epsilon for SIM aggregation')
    parser.add_argument('--rewire', action='store_true', default=False, help='whether to rewire the graph based on feature similarities before aggregation')
    parser.add_argument('--split_comp', action='store_true', default=False, help='whether to split the computational graph')
    # Quantiles
    parser.add_argument('--q_agg', action='store_true', help='whether to use Q aggregation')
    parser.add_argument('--q_include', type=str, default='quantile_25,quantile_50,quantile_75', help='Quantiles to include for Q aggregation, comma-separated')
    parser.add_argument('--q_interpolation', type=str, default='linear', choices=['linear', 'lower', 'higher', 'midpoint', 'nearest'], help='Interpolation method for Q aggregation')
    # Network Science
    parser.add_argument('--ns_agg', action='store_true', default=False, help='whether to use NS aggregation')
    parser.add_argument('--ns_include', type=str, default='degree,log_degree,closeness', help='Network science features to include for NS aggregation, comma-separated')
    parser.add_argument('--ns_cc_k', type=int, default=64, help='Number of source nodes to sample for closeness in NS aggregation')
    parser.add_argument('--ns_ev_max_iter', type=int, default=100, help='Maximum iterations for eigenvector centrality in NS aggregation')
    parser.add_argument('--ns_ev_tol', type=float, default=1e-6, help='Tolerance for eigenvector centrality in NS aggregation')
    parser.add_argument('--ns_betweenness_cpu', action='store_true', default=False, help='whether to use CPU for betweenness in NS aggregation')
    parser.add_argument('--ns_bc_k', type=int, default=64, help='Number of source nodes to sample for betweenness in NS aggregation')   

    # SHAP (FAF)
    parser.add_argument('--shap', action='store_true', default=False,
                        help='whether to compute SHAP values (for FAF)')
    parser.add_argument('--shap_background', type=int, default=100,
                        help='number of background samples for SHAP (for FAF)')