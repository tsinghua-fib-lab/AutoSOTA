CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle_big python train.py -s ./datasets/mipnerf360/bicycle -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --grad_abs_thresh 0.0008
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=flowers_big python train.py -s ./datasets/mipnerf360/flowers -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000 --dense 0.005 --grad_abs_thresh 0.001
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=garden_big python train.py -s ./datasets/mipnerf360/garden -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0003
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=stump_big python train.py -s ./datasets/mipnerf360/stump -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --dense 0.004 --grad_abs_thresh 0.001
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=treehill_big python train.py -s ./datasets/mipnerf360/treehill -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --dense 0.01 --grad_abs_thresh 0.0018
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=room_big python train.py -s ./datasets/mipnerf360/room -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0004 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=counter_big python train.py -s ./datasets/mipnerf360/counter -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0004
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen_big python train.py -s ./datasets/mipnerf360/kitchen -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bonsai_big python train.py -s ./datasets/mipnerf360/bonsai -i images --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck_big python train.py -s ./datasets/tanksandtemples/truck --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=train_big python train.py -s ./datasets/tanksandtemples/train --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0004 --dense 0.015 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom_big python train.py -s ./datasets/db/playroom --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=drjohnson_big python train.py -s ./datasets/db/drjohnson --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0005 --dense 0.005 --mult 0.7


CUDA_VISIBLE_DEVICES=0 python render.py -m output/bicycle_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/flowers_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/garden_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/stump_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/treehill_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/room_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/counter_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/kitchen_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/bonsai_big --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_big --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/train_big --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/playroom_big --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/drjohnson_big --skip_train --mult 0.7

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/bicycle_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/flowers_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/garden_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/stump_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/treehill_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/room_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/counter_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/kitchen_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/bonsai_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/train_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/playroom_big
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/drjohnson_big
