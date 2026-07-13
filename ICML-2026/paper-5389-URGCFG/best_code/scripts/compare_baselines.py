"""Simple baseline runner comparing OT solvers on DGPS-generated Gaussian pairs."""

from optimal_transport.ot_icnn_map import ICNNOT
from optimal_transport.ot_gnot_map import GNOTOT
from optimal_transport.ot_fc_map import FCOT
from config import WRITING_ROOT
from scripts import gauss_params
import argparse
import torch
from tools.dgps import generate_gaussian_pairs
from tools.visualize import visualize_transport
from tools.utils import L22, inverse_grad_L22
from tools.feedback import set_log_level

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=None, help='torch save file containing x and y arrays or the shapes')    
    p.add_argument('--generate', '-g', action='store_true', help='Generate gaussian paired data via tools.dgps (expects params in the file).')
    p.add_argument('--out', type=str, default=WRITING_ROOT, help='output directory')
    p.add_argument('--iters', type=int, default=gauss_params.niters, help='training steps for optimal_transport (small default)')
    p.add_argument('--nparams', type=int, default=gauss_params.model_size, help='number of parameters')
    p.add_argument('--inner_steps', type=int, default=gauss_params.inner_steps, help='inner loop steps')
    p.add_argument('--force_retrain', '-f', action='store_true', help='Force re-computation even if output file exists')
    p.add_argument('--lr', type=float, default=gauss_params.lr, help='learning rate')
    p.add_argument('--batch_size', type=int, default=gauss_params.batch_size, help='batch size')
    p.add_argument('--inner_optimizer', type=str, default=gauss_params.inner_optimizer, help='inner loop optimizer')
    p.add_argument('--log_level', type=str, default='INFO', 
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='Set logging level (default: INFO)')
    args = p.parse_args()
    
    # Set logging level from command line
    set_log_level(args.log_level)
    
    iters = args.iters
    fpath = args.data_path
    force_retrain = args.force_retrain
    nparams = args.nparams
    inner_steps = args.inner_steps
    lr = args.lr
    batch_size = args.batch_size
    inner_optimizer = args.inner_optimizer
    if fpath:
        specification = torch.load(fpath)
        if args.generate:
            x, y, _ = generate_gaussian_pairs(**specification["params"])
        else:
            x, y = specification["x"], specification["y"]
    else:
        # Use default DGPS params when no data path provided
        x, y, _ = generate_gaussian_pairs(**gauss_params.params)
    d = x.shape[1]
    
    # Initialize solvers
    # icnnot = ICNNOT.initialize_right_architecture(d, nparams, batch_size=batch_size)
    # losses_icnn = icnnot.fit(x, y, iters=iters, inner_steps=inner_steps, force_retrain=force_retrain)
    
    # gnotot = GNOTOT.initialize_right_architecture(d, nparams, T_lr=lr, D_lr=lr, cost_fn=euclidean_squared_cost, batch_size=batch_size)
    # losses_gnot = gnotot.fit(x, y, iters=iters, inner_steps=inner_steps, force_retrain=force_retrain)
    # FCOT is used as the active baseline in this script
    fcot = FCOT.initialize_right_architecture(d,
                                              nparams,
                                              cost=L22,
                                              inverse_kx=inverse_grad_L22,
                                              lr=lr,
                                              inner_optimizer=inner_optimizer,
                                              is_cost_metric=False)
    losses_fc = fcot.fit(x,
                         y,
                         iters=iters,
                         inner_steps=inner_steps,
                         force_retrain=force_retrain)
    
    visualize_transport(x, y, fcot)

# Expected --data_path contents:
#   - if --generate: a dict with "params" for generate_gaussian_pairs
#   - else: a dict with "x" and "y" tensors to load directly
# Artifacts: transport visualization written under --out (WRITING_ROOT by default).
