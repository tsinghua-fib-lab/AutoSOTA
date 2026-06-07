## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data  
|   |-- base_data
|   |-- MMVP
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- BEHAVE
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- RICH
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- MOYO
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- Hi4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
```
* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/feco-data/blob/main/train/data/base_data.tar.gz) by running:
```
bash scripts/download_train_base_data.sh
```
#### preprocessed_data
* Download `preprocessed_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/feco-data) by running:
```
bash scripts/download_train_preprocessed_data.sh
```
#### MMVP dataset
```
${ROOT} 
|-- data
|   |-- MMVP
|   |   |-- data
|   |   |   |-- annotations
|   |   |   |   |-- 20230422
|   |   |   |   |   |-- floor_info
|   |   |   |   |   |-- smpl_pose
|   |   |   |-- images
|   |   |   |   |-- 20230422
|   |   |   |   |   |-- S01
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- S12
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `images.7z` and `annotations.7z` to `"data/MMVP/data` and extract them.
#### BEHAVE dataset
```
${ROOT} 
|-- data
|   |-- BEHAVE
|   |   |-- data
|   |   |   |-- calibs
|   |   |   |-- Data01
|   |   |   |-- ...
|   |   |   |-- Data07
|   |   |   |-- objects
|   |   |   |-- split.json
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_behave.sh
```
#### RICH dataset
```
${ROOT} 
|-- data
|   |-- RICH
|   |   |-- data
|   |   |   |-- hsc
|   |   |   |-- images_jpg_subset
|   |   |   |-- multicam2world
|   |   |   |-- scan_calibration
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_rich.sh
```
#### MOYO dataset
```
${ROOT} 
|-- data
|   |-- MOYO
|   |   |-- data
|   |   |   |-- 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a
|   |   |   |-- 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-b
|   |   |   |-- ...
|   |   |   |-- 220926_yogi_body_hands_03596_Yogic_sleep_pose-a
|   |   |   |-- 220926_yogi_body_hands_03596_Yogic_sleep_pose-b
|   |   |   |-- cameras
|   |   |   |-- essentials
|   |   |   |-- mosh
|   |   |   |-- mosh_smpl
|   |   |   |-- pressure
|   |   |   |-- v_templates
|   |   |   |-- vicon
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Please clone MOYO toolkit [GitHub repository](https://github.com/sha2nkt/moyo_toolkit) in your home directory.
* Run the script below from MOYO toolkit [GitHub repository](https://github.com/sha2nkt/moyo_toolkit) to download MOYO dataset after environment setup:
```
bash ./moyo/bash/download_moyo.sh -o ./data/ -u -i -d
```
* Move the downloaded `data/MOYO` from MOYO toolkit to our `data/MOYO/data`.
#### Hi4D dataset
```
${ROOT} 
|-- data
|   |-- Hi4D
|   |   |-- data
|   |   |   |-- pair00
|   |   |   |-- pair00_1
|   |   |   |-- pair01
|   |   |   |-- pair02
|   |   |   |-- pair09
|   |   |   |-- pair10
|   |   |   |-- pair12
|   |   |   |-- pair13
|   |   |   |-- pair14
|   |   |   |-- pair15
|   |   |   |-- pair16
|   |   |   |-- pair17
|   |   |   |-- pair18
|   |   |   |-- pair19_1
|   |   |   |-- pair19_2
|   |   |   |-- pair21
|   |   |   |-- pair22
|   |   |   |-- pair23
|   |   |   |-- pair27
|   |   |   |-- pair28
|   |   |   |-- pair32_1
|   |   |   |-- pair32_2
|   |   |   |-- pair37
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `data` by running after download request from [official website](https://yifeiyin04.github.io/Hi4D):
```
bash scripts/download_official_hi4d.sh
bash scripts/extract_official_hi4d.sh
```