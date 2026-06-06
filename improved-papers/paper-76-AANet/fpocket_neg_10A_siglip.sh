source ~/miniconda3/etc/profile.d/conda.sh

run_name=$(basename $0)
run_name="${run_name%.*}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

n_gpu=4
MASTER_PORT=10058

conda activate unimol
data_path="dataset/cache_r10_buf4_conf10.lmdb"
finetune_mol_model="pretrain/mol_pre_no_h_220816.pt"
finetune_pocket_model="pretrain/pocket_pre_220816.pt"
run_dir="./"
if [ -f /.dockerenv ]; then
       echo "Running in Docker"
       pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple zstandard rdkit-pypi==2022.9.3
       data_path="/data/dataset/cache_r10_buf4_conf10.lmdb"
       finetune_mol_model="/data/pretrained/mol_pre_no_h_220816.pt"
       finetune_pocket_model="/data/pretrained/pocket_pre_220816.pt"
       run_dir="/data/run_dir/"
fi

save_dir="${run_dir}/savedir/${run_name}"
tmp_save_dir="${run_dir}/tmp_save_dir/${run_name}"
tsb_dir="${run_dir}/tsb_dir/${run_name}"

save_dir=$save_dir/$TIMESTAMP
tmp_save_dir=$tmp_save_dir/$TIMESTAMP
tsb_dir=$tsb_dir/$TIMESTAMP

batch_size=32
batch_size_valid=128
epoch=200
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-3

mkdir -p $save_dir
cp $0 $save_dir
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task drugclip --loss siglip --arch drugclip  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 10 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-last-epochs 5 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --subset exdudepcba --fpocket_aug --fpocket_neg
