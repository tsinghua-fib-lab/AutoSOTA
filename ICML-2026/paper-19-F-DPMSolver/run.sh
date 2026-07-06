# edm + cifar10
torchrun --standalone --nproc_per_node=4 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "CIFAR10-uncond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond

torchrun --standalone --nproc_per_node=4 main.py --subdirs cifar10_uncond --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "CIFAR10-uncond" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_uncond

torchrun --standalone --nproc_per_node=4 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "CIFAR10-cond" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond

torchrun --standalone --nproc_per_node=4 main.py --subdirs cifar10_cond --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "CIFAR10-cond" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz" --subdirs cifar10_cond

# edm2 + imagenet
torchrun --standalone --nproc_per_node=4 main.py --subdirs img64_l --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet64-L" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64_l

torchrun --standalone --nproc_per_node=4 main.py --subdirs img64_l --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet64-L" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64_l

torchrun --standalone --nproc_per_node=4 main.py --subdirs img64_s --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet64-S" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64_s

torchrun --standalone --nproc_per_node=4 main.py --subdirs img64_s --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet64-S" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img64.pkl --subdirs img64_s

torchrun --standalone --nproc_per_node=4 main.py --subdirs img512_xxl --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet512-XXL" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img512_xxl

torchrun --standalone --nproc_per_node=4 main.py --subdirs img512_xxl --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet512-XXL" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img512_xxl

torchrun --standalone --nproc_per_node=4 main.py --subdirs img512_xs --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet512-XS" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img512_xs

torchrun --standalone --nproc_per_node=4 main.py --subdirs img512_xs --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "ImageNet512-XS" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path refs/img512.pkl --subdirs img512_xs

# ldm + lsun/ffhq
torchrun --standalone --nproc_per_node=4 main.py --subdirs bedroom --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "LSUN" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz

torchrun --standalone --nproc_per_node=4 main.py --subdirs bedroom --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "LSUN" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs bedroom --ref_path refs/lsun-bedroom.npz

torchrun --standalone --nproc_per_node=4 main.py --subdirs ffhq --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "FFHQ" --order=1
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz

torchrun --standalone --nproc_per_node=4 main.py --subdirs ffhq --seeds=0-49999 --NFE=10 --batch=64 --algorithm_name="F-DPMSolver" --model_name "FFHQ" --order=2
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs ffhq --ref_path refs/ffhq-256.npz
