#!/bin/bash

# Please sign the license from EgoBody and get username and password first.

# Prompt user for credentials
read -p "Enter your EgoBody username: " USERNAME
read -s -p "Enter your EgoBody password: " PASSWORD
echo

# Set target directory (fixed)
TARGET_DIR="data/EgoBody/data"

# Base URL
BASE_URL="https://egobody.ethz.ch/data/dataset"

# List of files to download
FILES=(
  # "Egohmr_scene_preprocess_cube_s2_from_gt_release.zip"
  # "Egohmr_scene_preprocess_cube_s2_from_pred_release.zip"
  # "Egohmr_scene_preprocess_s1_release.zip"
  # "annotation_egocentric_smpl_npz.zip"
  # "calibrations.zip"
  # "data_info_release.csv"
  # "data_splits.csv"
  # "egocentric_color.zip"
  # "egocentric_depth.zip"
  # "egocentric_gaze.zip"
  # "human3d_egobody_pcd_data.zip"
  # "human3d_egobody_test_set_release.zip"
  # "kinect_cam_params.zip"
  "kinect_color.zip"
  # "kinect_depth.zip"
  # "rohm_init_egobody_rgb.zip"
  # "scene_mesh.zip"
  # "scene_mesh_4render_dart.zip"
  # "smpl_camera_wearer_test.zip"
  # "smpl_camera_wearer_train.zip"
  # "smpl_camera_wearer_val.zip"
  # "smpl_interactee_test.zip"
  # "smpl_interactee_train.zip"
  # "smpl_interactee_val.zip"
  # "smplx_camera_wearer_test.zip"
  # "smplx_camera_wearer_train.zip"
  # "smplx_camera_wearer_val.zip"
  # "smplx_interactee_test.zip"
  # "smplx_interactee_train.zip"
  # "smplx_interactee_val.zip"
  # "transf_matrices_all_seqs.pkl"
)

# Make target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# Start downloading
for FILE in "${FILES[@]}"; do
  echo "Downloading $FILE..."
  wget --user="$USERNAME" --password="$PASSWORD" "$BASE_URL/$FILE" -O "$FILE"

  # Unzip downloaded zip files and remove the zip file afterwards
  if [[ "$FILE" == *.zip ]]; then
    echo "Unzipping $FILE..."
    unzip -o "$FILE"
    rm -f "$FILE"
  fi
done

echo "All downloads completed."