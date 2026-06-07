#!/bin/bash

save_dir="data/BEHAVE/data"

mkdir -p "$save_dir"

# Download files
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date01.zip -O "$save_dir/Date01.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date02.zip -O "$save_dir/Date02.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date03.zip -O "$save_dir/Date03.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date04.zip -O "$save_dir/Date04.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date05.zip -O "$save_dir/Date05.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date06.zip -O "$save_dir/Date06.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date07.zip -O "$save_dir/Date07.zip"

wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip -O "$save_dir/objects.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/calibs.zip -O "$save_dir/calibs.zip"
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/split.json -O "$save_dir/split.json"

# Unzip Date archives into their own folders
for i in {01..07}; do
    mkdir -p "$save_dir/Date$i"
    unzip "$save_dir/Date$i.zip" -d "$save_dir/Date$i"
done

# Unzip other archives normally
unzip "$save_dir/objects.zip" -d "$save_dir"
unzip "$save_dir/calibs.zip" -d "$save_dir"

# Remove zip files after extraction
rm "$save_dir"/*.zip