import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from models import AS_FedDAG, GS_FedDAG, AS_FedDAG_linear
from datasets.simulation import property_generation
from helpers.evaluation import MetricsDAG
from helpers.tf_utils import set_seed

from datasets.simulation import property_generation
from helpers.config_utils import setup_parser, setup_logger
from datasets.data_gen import data_gen
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def main():

    # Parser the arguments
    parser = ArgumentParser()
    setup_parser(parser)
    args = parser.parse_args()
    
    # Set up logger and seed
    logger = setup_logger()
    set_seed(args.seed)

    # Generate the properties for heterogeneous data
    dataset_property = property_generation(args.num_client)

    # Generate the data
    print('Generating data, maybe slow! You can pre-genrate the data.')
    B_true, _, dataset, _ = data_gen(args.graph_type,
                                     args.node,
                                     args.edge,
                                     args.seed,
                                     args.num_client,
                                     args.gen_method,
                                     args.n,
                                     args.sem_type,
                                     dataset_property=dataset_property,
                                     method=args.linearity)

    # Run the FedDAG method
    print('Begin running the mothod.....')
    if args.fed_type == 'GS':
        model = GS_FedDAG(d=args.node,
                          num_client=args.num_client,
                          use_gpu=args.use_gpu,
                          seed=args.seed,
                          init_rho=args.init_rho,
                          l1_graph_penalty=args.l1_graph_penalty,
                          rho_multiply=args.rho_multiply,
                          lr=args.lr,
                          max_iter=args.max_iter,
                          iter_step=args.iter_step,
                          it_fl=args.it_fl,
                          init_alpha=args.init_alpha,
                          num_shared_client=args.num_shared_client,
                          logger=logger)

        model.learn(dataset)

    elif args.fed_type == 'AS':
        model = AS_FedDAG(n=args.n,
                          d=args.node,
                          use_gpu=args.use_gpu,
                          num_client=args.num_client,
                          seed=args.seed,
                          init_rho=args.init_rho,
                          l1_graph_penalty=args.l1_graph_penalty,
                          rho_multiply=args.rho_multiply,
                          lr=args.lr,
                          max_iter=args.max_iter,
                          iter_step=args.iter_step,
                          it_fl=args.it_fl,
                          init_alpha=args.init_alpha,
                          num_shared_client=args.num_shared_client,
                          logger=logger)

        model.learn(dataset)

    elif args.fed_type == 'AS_linear':
        model = AS_FedDAG_linear(n=args.n,
                                 d=args.node,
                                 use_gpu=args.use_gpu,
                                 num_client=args.num_client,
                                 max_iter=args.max_iter, 
                                 iter_step=args.iter_step,
                                 seed=args.seed,
                                 logger=logger)

        model.learn(dataset)

    raw_result = MetricsDAG(model.causal_matrix, B_true).metrics
    logger.info("run result:{0}".format(raw_result))

if __name__ == '__main__':
    main()