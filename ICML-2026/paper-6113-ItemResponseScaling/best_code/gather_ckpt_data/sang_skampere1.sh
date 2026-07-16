# 410m, 1b, 1.4b, 2.8b, amber, smol1.7b, smol360m

for repo_id in EleutherAI/pythia-1.4b EleutherAI/pythia-1b EleutherAI/pythia-410m EleutherAI/pythia-2.8b LLM360/Amber HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints
do
    python pretrain_helm_json2csv.py --repo_id "$repo_id"
    python pretrain_helm_csv2matrix.py --repo_id "$repo_id"
done
