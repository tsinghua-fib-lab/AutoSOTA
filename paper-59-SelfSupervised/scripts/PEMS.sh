for dir in scripts/patchtst scripts/cyclenet scripts/mlp scripts/itransformer scripts/scam_multi_patchtst scripts/scam_multi_itransformer scripts/scam_multi_cyclenet scripts/scam_multi_mlp
do
    for file in $dir/PEMS.sh
    do
        bash $file &
    done
done