# deal with path
# 6.9b, 12b

for repo_id in EleutherAI/pythia-6.9b EleutherAI/pythia-12b
do
    python pretrain_helm_json2csv.py --repo_id "$repo_id" --benchmark_dir /lfs/skampere1/0/yuhengtu/deval/helm/src/benchmark_output/runs
    python pretrain_helm_csv2matrix.py --repo_id "$repo_id" --benchmark_dir /lfs/skampere1/0/yuhengtu/deval/helm/src/benchmark_output/runs
done
