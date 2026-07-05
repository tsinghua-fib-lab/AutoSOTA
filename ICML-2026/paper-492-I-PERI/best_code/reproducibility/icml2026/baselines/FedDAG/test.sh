n_clientss="2 4 8 10"
n_variabless="3 5 8 10 20"
n_sampless="500 1000 2000"
data_types="struct"

for n_clients in $n_clientss; do 
    for n_variables in $n_variabless; do 
            for data_type in $data_types; do
                python reproducibility/icml2026/baselines/FedDAG/main.py --seed $1 --n_clients $n_clients --n_samples_client $n_samples_client --n_variables $n_variables --data_type $data_type --horizontal_split 1 --save 1 --linear 1
                for n_samples_client in $n_sampless; do
                    python reproducibility/icml2026/baselines/FedDAG/main.py --seed $1 --n_clients $n_clients --n_samples_client $n_samples_client --n_variables $n_variables --data_type $data_type --save 1 --linear 1
            done
        done            
    done
done
