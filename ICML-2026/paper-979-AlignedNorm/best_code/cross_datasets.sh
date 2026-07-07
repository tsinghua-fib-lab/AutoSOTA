bash scripts/alignednorm/cross_datasets_train.sh
for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet_a imagenet_r imagenet_sketch imagenetv2
do
    bash scripts/alignednorm/cross_datasets_test.sh $DATASET
done
