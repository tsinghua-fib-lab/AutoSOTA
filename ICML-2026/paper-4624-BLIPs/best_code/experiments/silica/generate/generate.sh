curl -L -o silica_train.xyz https://archive.materialscloud.org/records/tz6a0-hsb25/files/silica_train.xyz?download=1
curl -L -o silica_test.xyz https://archive.materialscloud.org/records/tz6a0-hsb25/files/silica_test.xyz?download=1
mkdir -p data
mkdir -p data/silica
mv *.xyz data/silica