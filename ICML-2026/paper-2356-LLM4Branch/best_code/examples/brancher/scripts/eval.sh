DATASET="cauctions"
export PYTHONPATH=$(pwd)
python ./examples/brancher/tester.py \
    --program  ./examples/brancher/program/${DATASET}/program.py \
    --dataset "${DATASET}" \
    --num_instances 80 \
    --method evolve \
    --easy