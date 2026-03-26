for dir in scripts/scam_multi_patchtst scripts/scam_multi_itransformer scripts/scam_multi_cyclenet scripts/scam_multi_mlp
do
    for file in $dir/5datasets.sh $dir/2datasets.sh
    do
        bash $file &
    done
done