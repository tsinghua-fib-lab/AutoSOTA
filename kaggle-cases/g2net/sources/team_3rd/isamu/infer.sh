#export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
python infer.py --config effb3
python infer.py --config effb4
python infer.py --config effb5
python infer.py --config effb7
python infer.py --config inceptionV3