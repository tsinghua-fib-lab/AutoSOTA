from vos_benchmark.benchmark import benchmark

# both arguments are passed as a list -- multiple datasets can be specified
benchmark(["current_work_dir/baseline/DAVIS/Annotations/480p"], ["current_work_dir/experiments/last_res/initial_preds"])