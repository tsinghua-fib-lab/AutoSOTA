from argparse import ArgumentParser
import logging

def setup_logger():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    logger = logging.getLogger()
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setFormatter(log_formatter)
    logger.addHandler(log_stream_handler)
    logger.setLevel(logging.INFO)
    return logger

def setup_parser(parser:ArgumentParser):
    ##### General settings #####
    # seed
    parser.add_argument('--seed',
                        type=int,
                        default=2022,
                        help='Reproducibility')

    ##### Dataset settings #####
    parser.add_argument('--n',
                        type=int,
                        required=True,
                        default=1000,
                        help='Number of observations on each client.')
                        
    parser.add_argument('--num_client',
                        type=int,
                        required=True,
                        default=10,
                        help='Number of the clients.')

    ## simulation data generation settings
    parser.add_argument('--node',
                        type=int,
                        default=10,
                        help='Number of nodes.')

    parser.add_argument('--edge',
                        type=int,
                        default=20,
                        help='Number of edges.')

    parser.add_argument('--graph_type',
                        type=str,
                        default='er',
                        choices=['er', 'sf'],
                        help='Type of the graph.')

    parser.add_argument('--gen_method',
                        type=str,
                        default='multiiid',
                        choices=['noniid', 'multiiid'],
                        help='Type of the data distributed on different clients.')
    
    parser.add_argument('--linearity',
                        type=str,
                        default='linear',
                        help='The linearity of data model.')

    parser.add_argument('--sem_type',
                        type=str,
                        default='gp',
                        choices=['gp', 'mim', 'gp-add', 'mlp'],
                        help='Types of the SEM function.')
    
    ##### Training settings #####
    parser.add_argument('--lr',
                        type=float,
                        default=3e-2,
                        help='Learning rate')

    parser.add_argument('--init_rho',
                        type=float,
                        default=0.001,
                        help='Initial value of rho.')

    parser.add_argument('--rho_multiply',
                        type=float,
                        default=10,
                        help='Multiplication to amplify rho each time.')

    parser.add_argument('--l1_graph_penalty',
                        type=float,
                        required=False,
                        default=0.01,
                        help='L1 sparsity coefficient.')
                        
    parser.add_argument('--it_fl',
                        type=int,
                        required=False,
                        default=100,
                        help='The iterations for gradient exchange.')

    parser.add_argument('--init_alpha',
                        type=float,
                        required=False,
                        default=0.0,
                        help='The initial number of alpha.')

    ##### Other settings #####
    parser.add_argument('--use_gpu',
                        action='store_true')

    parser.add_argument('--fed_type',
                        type=str,
                        choices=['GS', 'AS', 'AS_linear'],
                        required=True,
                        help='Different sharing methods.')
    
    parser.add_argument('--num_shared_client',
                        default=None,
                        type=int,
                        help='Number of clients involed in the fed process.')
    
    parser.add_argument('--max_iter',
                        default=20,
                        type=int,
                        help='Number of max iterations.')
    
    parser.add_argument('--iter_step',
                        default=1000,
                        type=int,
                        help='Iterations for each sub-problem.')