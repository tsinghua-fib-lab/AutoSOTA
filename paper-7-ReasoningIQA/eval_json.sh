CUDA_VISIBLE_DEVICES=0 python eval_json.py \
    --ckpt ./checkpoints/ckpt.pt \
    --clip_model ./checkpoints/best \
    --pca_path ./checkpoints/pca.pkl \
    --basis_path ./checkpoints/basis.npz \
    --test_json ./Data-DeQA-Score/KADID10K/metas/test_kadid_2k.json \
    --image_root ./Data-DeQA-Score \
    --out_json logs/pred_kadid.json