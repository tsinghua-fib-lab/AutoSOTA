for DATASET in eurosat dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
do
    bash scripts/alignednorm/base2new_train.sh $DATASET
    bash scripts/alignednorm/base2new_test.sh $DATASET
done
