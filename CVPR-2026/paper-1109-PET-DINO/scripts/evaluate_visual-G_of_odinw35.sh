#!/usr/bin/env bash

CONFIG=$1
ODinW35_CONFIG=$2
CHECKPOINT_FILE=$3
HOST_GPU_NUM=$4

# Init envs
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
PORT=${PORT:-29500}
export PYTHONPATH=`pwd`:$PYTHONPATH

set -e

# # ---------------------1 AerialMaritimeDrone---------------------#
# ODINW_IDX=1
# COCO_JSON_PATH=./data/odinw/AerialMaritimeDrone/large/train/annotations_without_background.json
# IMAGE_PREFIX=./data/odinw/AerialMaritimeDrone/large/train/
# ODVG_SAVE_PATH=./images/odinw/AerialMaritimeDrone_odvg_16.json
# OUTPUT_PATH=./outputs/odinw/AerialMaritimeDrone/

datasets=(
    # Format: ODINW_IDX:"dataset_name:COCO_JSON_PATH:IMAGE_PREFIX:ODVG_SAVE_PATH:OUTPUT_PATH"
    # odinw13
    "1:AerialMaritimeDrone_large:./data/odinw/AerialMaritimeDrone/large/train/annotations_without_background.json:./data/odinw/AerialMaritimeDrone/large/train/:./images/odinw/AerialMaritimeDrone_train_odvg_16.json:./outputs/odinw/AerialMaritimeDrone/"
    "4:Aquarium:./data/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/annotations_without_background.json:./data/odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/:./images/odinw/Aquarium_train_odvg_16.json:./outputs/odinw/Aquarium/"
    "9:CottontailRabbits:./data/odinw/CottontailRabbits/train/annotations_without_background.json:./data/odinw/CottontailRabbits/train/:./images/odinw/CottontailRabbits_train_odvg_16.json:./outputs/odinw/CottontailRabbits/"
    "12:EgoHands_generic:./data/odinw/EgoHands/generic/train/annotations_without_background.json:./data/odinw/EgoHands/generic/train/:./images/odinw/EgoHands_train_odvg_16.json:./outputs/odinw/EgoHands/"
    "17:NorthAmericaMushrooms:./data/odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/annotations_without_background.json:./data/odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/:./images/odinw/NorthAmericaMushrooms_train_odvg_16.json:./outputs/odinw/NorthAmericaMushrooms/"
    "22:Packages:./data/odinw/Packages/Raw/train/annotations_without_background.json:./data/odinw/Packages/Raw/train/:./images/odinw/Packages_train_odvg_16.json:./outputs/odinw/Packages/"
    "23:PascalVOC:./data/odinw/PascalVOC/train/annotations_without_background.json:./data/odinw/PascalVOC/train/:./images/odinw/PascalVOC_train_odvg_16.json:./outputs/odinw/PascalVOC/"
    "24:pistols:./data/odinw/pistols/export/train_annotations_without_background.json:./data/odinw/pistols/export/:./images/odinw/pistols_train_odvg_16.json:./outputs/odinw/pistols/"
    "26:pothole:./data/odinw/pothole/train/annotations_without_background.json:./data/odinw/pothole/train/:./images/odinw/pothole_train_odvg_16.json:./outputs/odinw/pothole/"
    "27:Raccoon:./data/odinw/Raccoon/Raccoon.v2-raw.coco/train/annotations_without_background.json:./data/odinw/Raccoon/Raccoon.v2-raw.coco/train/:./images/odinw/Raccoon_train_odvg_16.json:./outputs/odinw/Raccoon/"
    "29:ShellfishOpenImages:./data/odinw/ShellfishOpenImages/raw/train/annotations_without_background.json:./data/odinw/ShellfishOpenImages/raw/train/:./images/odinw/ShellfishOpenImages_train_odvg_16.json:./outputs/odinw/ShellfishOpenImages/"
    "31:thermalDogsAndPeople:./data/odinw/thermalDogsAndPeople/train/annotations_without_background.json:./data/odinw/thermalDogsAndPeople/train/:./images/odinw/thermalDogsAndPeople_train_odvg_16.json:./outputs/odinw/thermalDogsAndPeople/"
    "33:VehiclesOpenImages:./data/odinw/VehiclesOpenImages/416x416/train/annotations_without_background.json:./data/odinw/VehiclesOpenImages/416x416/train/:./images/odinw/VehiclesOpenImages_train_odvg_16.json:./outputs/odinw/VehiclesOpenImages/"
    # odinw35 - odinw13
    "2:AerialMaritimeDrone_tiled:./data/odinw/AerialMaritimeDrone/tiled/train/annotations_without_background.json:./data/odinw/AerialMaritimeDrone/tiled/train/:./images/odinw/AerialMaritimeDrone_tiled_train_odvg_16.json:./outputs/odinw/AerialMaritimeDrone_tiled/"
    "3:AmericanSignLanguageLetters:./data/odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/annotations_without_background.json:./data/odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/:./images/odinw/AmericanSignLanguageLetters_train_odvg_16.json:./outputs/odinw/AmericanSignLanguageLetters/"
    "5:BCCD:./data/odinw/BCCD/BCCD.v3-raw.coco/train/annotations_without_background.json:./data/odinw/BCCD/BCCD.v3-raw.coco/train/:./images/odinw/BCCD_train_odvg_16.json:./outputs/odinw/BCCD/"
    "6:boggleBoards:./data/odinw/boggleBoards/416x416AutoOrient/export/train_annotations_without_background.json:./data/odinw/boggleBoards/416x416AutoOrient/export/:./images/odinw/boggleBoards_train_odvg_16.json:./outputs/odinw/boggleBoards/"
    "7:brackishUnderwater:./data/odinw/brackishUnderwater/960x540/train/annotations_without_background.json:./data/odinw/brackishUnderwater/960x540/train/:./images/odinw/brackishUnderwater_train_odvg_16.json:./outputs/odinw/brackishUnderwater/"
    "8:ChessPieces:./data/odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/annotations_without_background.json:./data/odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/:./images/odinw/ChessPieces_train_odvg_16.json:./outputs/odinw/ChessPieces/"
    "10:dice:./data/odinw/dice/mediumColor/export/train_annotations_without_background.json:./data/odinw/dice/mediumColor/export/:./images/odinw/dice_train_odvg_16.json:./outputs/odinw/dice/"
    "11:DroneControl:./data/odinw/DroneControl/Drone Control.v3-raw.coco/train/annotations_without_background.json:./data/odinw/DroneControl/Drone Control.v3-raw.coco/train/:./images/odinw/DroneControl_train_odvg_16.json:./outputs/odinw/DroneControl/"
    "13:EgoHands_specific:./data/odinw/EgoHands/specific/train/annotations_without_background.json:./data/odinw/EgoHands/specific/train/:./images/odinw/EgoHands_specific_train_odvg_16.json:./outputs/odinw/EgoHands_specific/"
    "14:HardHatWorkers:./data/odinw/HardHatWorkers/raw/train/annotations_without_background.json:./data/odinw/HardHatWorkers/raw/train/:./images/odinw/HardHatWorkers_train_odvg_16.json:./outputs/odinw/HardHatWorkers/"
    "15:MaskWearing:./data/odinw/MaskWearing/raw/train/annotations_without_background.json:./data/odinw/MaskWearing/raw/train/:./images/odinw/MaskWearing_train_odvg_16.json:./outputs/odinw/MaskWearing/"
    "16:MountainDewCommercial:./data/odinw/MountainDewCommercial/train/annotations_without_background.json:./data/odinw/MountainDewCommercial/train/:./images/odinw/MountainDewCommercial_train_odvg_16.json:./outputs/odinw/MountainDewCommercial/"
    "18:openPoetryVision:./data/odinw/openPoetryVision/512x512/train/annotations_without_background.json:./data/odinw/openPoetryVision/512x512/train/:./images/odinw/openPoetryVision_train_odvg_16.json:./outputs/odinw/openPoetryVision/"
    "19:OxfordPets_by_breed:./data/odinw/OxfordPets/by-breed/train/annotations_without_background.json:./data/odinw/OxfordPets/by-breed/train/:./images/odinw/OxfordPets_by_breed_train_odvg_16.json:./outputs/odinw/OxfordPets_by_breed/"
    "20:OxfordPets_by_species:./data/odinw/OxfordPets/by-species/train/annotations_without_background.json:./data/odinw/OxfordPets/by-species/train/:./images/odinw/OxfordPets_by_species_train_odvg_16.json:./outputs/odinw/OxfordPets_by_species/"
    "21:PKLot:./data/odinw/PKLot/640/train/annotations_without_background.json:./data/odinw/PKLot/640/train/:./images/odinw/PKLot_train_odvg_16.json:./outputs/odinw/PKLot/"
    "25:plantdoc:./data/odinw/plantdoc/416x416/train/annotations_without_background.json:./data/odinw/plantdoc/416x416/train/:./images/odinw/plantdoc_train_odvg_16.json:./outputs/odinw/plantdoc/"
    "28:selfdrivingCar:./data/odinw/selfdrivingCar/fixedLarge/export/train_annotations_without_background.json:./data/odinw/selfdrivingCar/fixedLarge/export/:./images/odinw/selfdrivingCar_train_odvg_16.json:./outputs/odinw/selfdrivingCar/"
    "30:ThermalCheetah:./data/odinw/ThermalCheetah/train/annotations_without_background.json:./data/odinw/ThermalCheetah/train/:./images/odinw/ThermalCheetah_train_odvg_16.json:./outputs/odinw/ThermalCheetah/"
    "32:UnoCards:./data/odinw/UnoCards/raw/train/annotations_without_background.json:./data/odinw/UnoCards/raw/train/:./images/odinw/UnoCards_train_odvg_16.json:./outputs/odinw/UnoCards/"
    "34:WildfireSmoke:./data/odinw/WildfireSmoke/train/annotations_without_background.json:./data/odinw/WildfireSmoke/train/:./images/odinw/WildfireSmoke_train_odvg_16.json:./outputs/odinw/WildfireSmoke/"
    "35:websiteScreenshots:./data/odinw/websiteScreenshots/train/annotations_without_background.json:./data/odinw/websiteScreenshots/train/:./images/odinw/websiteScreenshots_train_odvg_16.json:./outputs/odinw/websiteScreenshots/"
)

for dataset_cfg in "${datasets[@]}"; do
    IFS=':' read -r ODINW_IDX \
        name \
        COCO_JSON_PATH \
        IMAGE_PREFIX \
        ODVG_SAVE_PATH \
        OUTPUT_PATH <<< "$dataset_cfg"

    echo -e "\n\n# ---------------------Processing dataset $ODINW_IDX $name---------------------#"
    
    ###########################
    # 1. Extract n images to odvg file
    echo '1. Extract n images to odvg file'
    python3 ./scripts/put_n_images_per_class_to_odvg.py --json_path "$COCO_JSON_PATH" \
        --image_prefix "$IMAGE_PREFIX" --output_path "$ODVG_SAVE_PATH" --number 16
    echo -e '\n'
    
    ###########################
    # 2. Extract visual embedding
    echo '2. Extract visual embedding'
    mkdir -p "${OUTPUT_PATH}/visual_embedding/"
    python3 scripts/image_demo.py "$IMAGE_PREFIX" \
        $CONFIG --weights $CHECKPOINT_FILE \
        --prompt_type 'Visual' \
        --input_od_json "$ODVG_SAVE_PATH" --extract-visual-embedding --out-dir "$OUTPUT_PATH/visual_embedding/"
    echo -e '\n'
    
    ###########################
    # 3. Run evaluation
    echo '3. Start evaluation'
    bash ./tools/dist_test.sh $ODinW35_CONFIG $CHECKPOINT_FILE $HOST_GPU_NUM --cfg-options \
        model.test_cfg.prompt_type='Visual' \
        test_cfg.type='CustomTestLoop' \
        test_cfg.prompt_visual_embedding_path="${OUTPUT_PATH}/visual_embedding/" \
        val_dataloader.dataset.odinw_idx=$ODINW_IDX \
        test_dataloader.dataset.odinw_idx=$ODINW_IDX \
        val_evaluator.odinw_idx=$ODINW_IDX \
        test_evaluator.odinw_idx=$ODINW_IDX \
        val_evaluator.record_file="${OUTPUT_PATH}/record_file.json" \
        test_evaluator.record_file="${OUTPUT_PATH}/record_file.json"
    echo -e '\n'
done
