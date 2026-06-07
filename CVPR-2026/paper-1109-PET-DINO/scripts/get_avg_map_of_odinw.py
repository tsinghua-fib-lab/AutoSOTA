import os
import json
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser('coco to odvg format.', add_help=True)
    parser.add_argument('--dataset', type=str, choices=['odinw13', 'odinw35'], default='odinw35', help='dataset name')
    args = parser.parse_args()

    output_path = "./outputs/odinw/"
    record_filename = "record_file.json"

    # odinw13
    if args.dataset == 'odinw13':
        odinw_folder_list = [
            'AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands',
            'NorthAmericaMushrooms', 'Packages', 'PascalVOC', 'pistols', 'pothole',
            'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople',
            'VehiclesOpenImages'
        ]
    else:

        # odinw35
        odinw_folder_list  = [
            'AerialMaritimeDrone',      # AerialMaritimeDrone_large
            'AerialMaritimeDrone_tiled',
            'AmericanSignLanguageLetters',
            'Aquarium',
            'BCCD',
            'boggleBoards',
            'brackishUnderwater',
            'ChessPieces',
            'CottontailRabbits',
            'dice',
            'DroneControl',
            'EgoHands',     # EgoHands_generic
            'EgoHands_specific',
            'HardHatWorkers',
            'MaskWearing',
            'MountainDewCommercial',
            'NorthAmericaMushrooms',
            'openPoetryVision',
            'OxfordPets_by_breed',
            'OxfordPets_by_species',
            'PKLot',
            'Packages',
            'PascalVOC',
            'pistols',
            'plantdoc',
            'pothole',
            'Raccoon',
            'selfdrivingCar',
            'ShellfishOpenImages',
            'ThermalCheetah',
            'thermalDogsAndPeople',
            'UnoCards',
            'VehiclesOpenImages',
            'WildfireSmoke',
            'websiteScreenshots',
        ]

    # Collect mAP values from all datasets
    map_values = []
    
    for dataset_prefix in odinw_folder_list:
        record_file_path = os.path.join(output_path, dataset_prefix, record_filename)
        
        # Read and parse JSON file
        with open(record_file_path, 'r') as f:
            record_data = json.load(f)
        
        # Extract bbox_mAP value and round to 4 decimal places
        map_value = round(record_data["avg/coco/bbox_mAP"], 4)
        map_values.append(map_value)
        print(f"{dataset_prefix}: {map_value}")
    
    # Calculate and print average
    avg_map = round(sum(map_values) / len(map_values), 4)
    print(f"\n{args.dataset} Average mAP: {avg_map}")




