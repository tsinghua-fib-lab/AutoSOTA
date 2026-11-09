config=${1:-default_run}

timestamp=$(date +%Y%m%d-%H%M%S)
for i in $(seq 5)
do
    echo "Starting" $timestamp "fold $i"
    python train.py --config $config --timestamp $timestamp --fold $i
done
python infer.py --config $config --timestamp $timestamp
# python infer.py --config $config --timestamp $timestamp --submit
