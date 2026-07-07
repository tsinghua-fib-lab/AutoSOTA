for DATASET in dtd eurosat ucf101 oxford_flowers oxford_pets fgvc_aircraft caltech101 food101 stanford_cars sun397 imagenet
do
    bash scripts/alignednorm/few_shot.sh $DATASET
done
