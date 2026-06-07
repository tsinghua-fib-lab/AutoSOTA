#!/bin/bash

# URL encode function
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Prompt for username and password
echo -e "\nYou need to register at https://rich.is.tue.mpg.de/"
read -p "Username: " username
read -p "Password: " password

# Encode credentials
username=$(urle "$username")
password=$(urle "$password")

# Set save directory
save_dir="data/RICH/data"
mkdir -p "$save_dir"

# ----------- Download Human-Scene Contact -----------
echo "Downloading train_hsc.zip..."
wget --post-data "username=$username&password=$password" \
  'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=train_hsc.zip' \
  -O "$save_dir/train_hsc.zip" \
  --no-check-certificate --continue

echo "Downloading test_hsc.zip..."
wget --post-data "username=$username&password=$password" \
  'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=test_hsc.zip' \
  -O "$save_dir/test_hsc.zip" \
  --no-check-certificate --continue

# ----------- Download JPG Image Archives -----------
echo "Downloading JPG_images/train.tar.gz..."
wget --post-data "username=$username&password=$password" \
  'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=JPG_images/train.tar.gz' \
  -O "$save_dir/train.tar.gz" \
  --no-check-certificate --continue

echo "Downloading JPG_images/test.tar.gz..."
wget --post-data "username=$username&password=$password" \
  'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=JPG_images/test.tar.gz' \
  -O "$save_dir/test.tar.gz" \
  --no-check-certificate --continue

# ----------- Download Scan Calibration -----------
echo "Downloading scan_calibration.zip..."
wget --post-data "username=$username&password=$password" \
  'https://download.is.tue.mpg.de/download.php?domain=rich&resume=1&sfile=scan_calibration.zip' \
  -O "$save_dir/scan_calibration.zip" \
  --no-check-certificate --continue

# ----------- Download Multicam2World Info (no auth needed) -----------
echo "Downloading multicam2world.zip..."
wget 'https://rich.is.tue.mpg.de/media/upload/multicam2world.zip' \
  -O "$save_dir/multicam2world.zip" \
  --continue

# ----------- Unzip / Untar -----------
echo "Extracting scan_calibration.zip..."
unzip "$save_dir/scan_calibration.zip" -d "$save_dir"

echo "Extracting multicam2world.zip..."
unzip "$save_dir/multicam2world.zip" -d "$save_dir"

echo "Extracting train_hsc.zip..."
mkdir -p "$save_dir/hsc"
unzip "$save_dir/train_hsc.zip" -d "$save_dir"
mv "$save_dir/train_hsc" "$save_dir/hsc/train"

echo "Extracting test_hsc.zip..."
unzip "$save_dir/test_hsc.zip" -d "$save_dir/hsc"

echo "Extracting train.tar.gz..."
mkdir -p "$save_dir/images_jpg_subset"
tar -xzf "$save_dir/train.tar.gz" -C "$save_dir/images_jpg_subset"

echo "Extracting test.tar.gz..."
mkdir -p "$save_dir/images_jpg_subset"
tar -xzf "$save_dir/test.tar.gz" -C "$save_dir/images_jpg_subset"

echo "All RICH files downloaded and extracted to $save_dir"