## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data
|   |-- base_data
|   |-- BEHAVE
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- EgoBody
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- Hi4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- InstaVariety
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- InterCap
|   |   |-- calibration
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- MMVP
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- MotionPro
|   |   |-- camera_pose
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- MOYO
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- MPII
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- OpenPose
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- PennAction
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- PROX
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- RICH
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- UTZap50K
|   |   |-- data
|   |   |-- dataset.py
```
#### base_data
* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/feco-data/blob/main/train/data/base_data.tar.gz) by running:
```
bash scripts/download_train_base_data.sh
```
#### preprocessed_data
* Download `preprocessed_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/feco-data) by running:
```
bash scripts/download_train_preprocessed_data.sh
```
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
#### EgoBody dataset
```
${ROOT} 
|-- data
|   |-- Decaf
|   |   |-- data
|   |   |   |-- annotation_egocentric_smpl_npz
|   |   |   |-- calibrations
|   |   |   |-- egocentric_color
|   |   |   |-- human3d_egobody_test_set_release
|   |   |   |-- init_egobody_rgb
|   |   |   |-- kinect_cam_params
|   |   |   |-- kinect_color
|   |   |   |-- scene_mesh
|   |   |   |-- scene_mesh_4render_dart
|   |   |   |-- smpl_camera_wearer_test
|   |   |   |-- smpl_camera_wearer_train
|   |   |   |-- smpl_camera_wearer_val
|   |   |   |-- smpl_interactee_test
|   |   |   |-- smpl_interactee_train
|   |   |   |-- smpl_interactee_val
|   |   |   |-- smplx_camera_wearer_test
|   |   |   |-- smplx_camera_wearer_train
|   |   |   |-- smplx_camera_wearer_val
|   |   |   |-- smplx_interactee_test
|   |   |   |-- smplx_interactee_train
|   |   |   |-- smplx_interactee_val
|   |   |   |-- data_info_release.csv
|   |   |   |-- data_splits.csv
|   |   |   |-- transf_matrices_all_seqs.pkl
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `data` by running after download request from [official website](https://egobody.ethz.ch/):
```
bash scripts/download_official_egobody.sh
```
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
#### InstaVariety dataset
```
${ROOT} 
|-- data
|   |-- InstaVariety
|   |   |-- data
|   |   |   |-- contacts
|   |   |   |-- images
|   |   |   |-- keypoints
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Everything already downloaded.
#### InterCap dataset
```
${ROOT} 
|-- data
|   |-- InterCap
|   |   |-- calibration
|   |   |   |-- Color_2.json
|   |   |   |-- Color_3.json
|   |   |   |-- Color_4.json
|   |   |   |-- Color_5.json
|   |   |   |-- Color_6.json
|   |   |   |-- Color.json
|   |   |-- data
|   |   |   |-- Res
|   |   |   |-- RGBD_Images
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- splits
|   |   |   |-- intercap_test.json
|   |   |   |-- intercap_train.json
|   |   |   |-- test.txt
|   |   |   |-- train.txt
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_intercap.sh
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
#### MotionPro dataset
```
${ROOT} 
|-- data
|   |-- MotionPro
|   |   |-- camera_pose
|   |   |   |-- 0729.npy
|   |   |   |-- ...
|   |   |   |-- 1009.npy
|   |   |-- data
|   |   |   |-- 0729
|   |   |   |   |-- csy
|   |   |   |   |-- qnx
|   |   |   |-- 0730
|   |   |   |   |-- xty
|   |   |   |   |-- zxy
|   |   |   |-- 0731
|   |   |   |   |-- yxy
|   |   |   |-- 0801
|   |   |   |   |-- hcc
|   |   |   |   |-- lh
|   |   |   |-- 0802
|   |   |   |   |-- hw
|   |   |   |   |-- zyy
|   |   |   |-- 0804
|   |   |   |   |-- zyh
|   |   |   |-- 0807
|   |   |   |   |-- wsy
|   |   |   |   |-- zyt
|   |   |   |-- 0809
|   |   |   |   |-- wcm
|   |   |   |   |-- zkx
|   |   |   |-- 0812
|   |   |   |   |-- amh
|   |   |   |   |-- wym
|   |   |   |-- 0814
|   |   |   |   |-- wjl
|   |   |   |   |-- zzb
|   |   |   |-- 0815
|   |   |   |   |-- jky
|   |   |   |-- 0905
|   |   |   |   |-- xzy
|   |   |   |-- 0906
|   |   |   |   |-- xfq
|   |   |   |-- 0909
|   |   |   |   |-- djl
|   |   |   |   |-- dx
|   |   |   |   |-- td
|   |   |   |-- 0910
|   |   |   |   |-- klf
|   |   |   |   |-- zyc
|   |   |   |-- 0911
|   |   |   |   |-- crf
|   |   |   |   |-- fjh
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `RGB data`, `Pressure data`, `SMPL parameters` for `MotionPRO_1.zip`, `MotionPRO_2.zip`, `MotionPRO_3.zip`, `MotionPRO_4.zip`, `MotionPRO_5.zip`, `MotionPRO_6.zip`, `MotionPRO_7.zip` from [official website](https://nju-cite-mocaphumanoid.github.io/MotionPRO/).
* Please move `0729`, `0730`, ..., `0911` to `data/MotionPro/data`.
* Extract frames from videos
```
python scripts/extract_official_motionpro_video.py
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
#### MPII dataset
```
${ROOT} 
|-- data
|   |-- MPII
|   |   |-- data
|   |   |   |-- contacts
|   |   |   |-- images
|   |   |   |-- keypoints
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Everything already downloaded.
#### OpenPose dataset
```
${ROOT} 
|-- data
|   |-- OpenPose
|   |   |-- data
|   |   |   |-- train2017
|   |   |   |-- train2017_contact
|   |   |   |-- val2017
|   |   |   |-- val2017_contact
|   |   |   |-- person_keypoints_train2017_foot_v1.json
|   |   |   |-- person_keypoints_val2017_foot_v1.json
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- split_sample_ids.txt
|   |   |   |-- test
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Everything already downloaded.
#### PennAction dataset
```
${ROOT} 
|-- data
|   |-- PennAction
|   |   |-- data
|   |   |   |-- contacts
|   |   |   |-- frames
|   |   |   |-- keypoints
|   |   |   |-- labels
|   |   |   |-- segmentation
|   |   |   |-- tools
|   |   |   |-- videos
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Everything already downloaded.
#### PROX dataset
```
${ROOT} 
|-- data
|   |-- PROX
|   |   |-- data
|   |   |   |-- quantitative
|   |   |   |   |-- body_segments
|   |   |   |   |-- ...
|   |   |   |   |-- sdf
|   |   |   |   |-- vicon2scene.json
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |   |-- ground_data
|   |   |   |   |-- pixel_height_data
|   |   |   |   |-- split_sample_ids.txt
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_prox.sh
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
#### UTZap50K dataset
```
${ROOT} 
|-- data
|   |-- UTZap50K
|   |   |-- data
|   |   |   |-- ut-zap50k-data
|   |   |   |-- ut-zap50k-feats
|   |   |   |-- ut-zap50k-images
|   |   |   |-- ut-zap50k-images-square
|   |   |   |-- ut-zap50k-lexi
|   |   |   |-- readme.txt
|   |   |-- dataset.py
```
* Everything already downloaded.