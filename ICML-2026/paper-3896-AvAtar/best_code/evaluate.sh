#!/bin/bash
cd /repo/source
export PYTHONPATH=/autosota_cache/PlanetAlign:${PYTHONPATH}
python3 active_na.py --alg PARROT --dataset phone-email --device cuda --query_round 10 --query_portion 0.2 --init_train_ratio 0.2 --outIter 10 --modes sq_l2_adjoint_grad --anchor_selection_seed 0 2>&1 | grep "MRR:" | tail -2
