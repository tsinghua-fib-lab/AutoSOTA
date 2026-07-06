n_clientss="2 4 8 10"
n_variabless="3 5 8 10 20"
n_sampless="500 1000 2000"
data_types="struct"
cd_functions="pc"


for n_clients in $n_clientss; do 
  for n_variables in $n_variabless; do 
    for cd_function in $cd_functions; do 
    for data_type in $data_types; do
          if [ "$cd_function" = "lingam" ]; then
              noise_distribution="uniform"
          else
              noise_distribution="normal"
          fi
          python main.py --n_clients $n_clients --n_variables $n_variables --cd_function $cd_function --data_type $data_type --save 1 --linear 1 --masked 1 --noise_distribution $noise_distribution --horizontal_split 1 --n_samples_client 100 --seed $1
          python main.py --n_clients $n_clients --n_variables $n_variables --cd_function $cd_function --data_type $data_type --save 1 --linear 1 --noise_distribution $noise_distribution --horizontal_split 1 --seed $1 --n_samples_client 100
          for n_samples in $n_sampless; do
              python main.py --n_clients $n_clients --n_variables $n_variables --cd_function $cd_function --data_type $data_type --save 1 --linear 1 --masked 1 --noise_distribution $noise_distribution --n_samples_client $n_samples --seed $1
              python main.py --n_clients $n_clients --n_variables $n_variables --cd_function $cd_function --data_type $data_type --save 1 --linear 1 --noise_distribution $noise_distribution --seed $1 --n_samples_client $n_samples
          done
        # done
      done
    done
  done
done