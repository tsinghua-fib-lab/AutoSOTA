curl -L -o ammonia_train.xyz https://archive.materialscloud.org/record/file\?record_id\=1979\&filename\=ammonia_train.xyz
curl -L -o ammonia_test.xyz https://archive.materialscloud.org/record/file\?record_id=\1979\&filename\=ammonia_test.xyz
mkdir -p data
mkdir -p data/ammonia
mv *.xyz data/ammonia