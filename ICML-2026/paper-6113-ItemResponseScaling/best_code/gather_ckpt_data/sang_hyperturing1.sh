# download from hf
# 410m, 1b, 1.4b, 2.8b, smol135m

for repo_id in EleutherAI/pythia-1.4b EleutherAI/pythia-1b EleutherAI/pythia-410m EleutherAI/pythia-2.8b LLM360/Amber HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints
do
    python pretrain_helm_json2csv_sang_hyper1.py --repo_id "$repo_id"
    python pretrain_helm_csv2matrix_sang_hyper1.py --repo_id "$repo_id"
done
