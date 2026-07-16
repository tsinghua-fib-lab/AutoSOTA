# 6.9b, 12b, amber

for repo_id in EleutherAI/pythia-12b EleutherAI/pythia-6.9b LLM360/Amber
do
    python pretrain_helm_json2csv.py --repo_id "$repo_id"
    python pretrain_helm_csv2matrix.py --repo_id "$repo_id"
done
