#!/bin/bash

set -e

# URL encode function
urle () {
    [[ "$1" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        if [[ "$x" =~ [a-zA-Z0-9.~_-] ]]; then
            printf "%s" "$x"
        else
            printf '%%%02X' "'$x"
        fi
    done
    echo
}

# Prompt for credentials
echo -e "\nYou need to register at https://intercap.is.tue.mpg.de/"
read -p "Username: " username
read -s -p "Password: " password
echo

save_dir="data/InterCap/data"
mkdir -p "$save_dir"
mkdir -p "$save_dir/Res"
mkdir -p "$save_dir/RGBD_Images"

username=$(urle "$username")
password=$(urle "$password")

# Download and unzip RGBD_Individuals
for i in {01..10}; do
    wget --continue --no-check-certificate --save-cookies cookies.txt --keep-session-cookies \
        --post-data "username=$username&password=$password" \
        "https://download.is.tue.mpg.de/download.php?domain=intercap&resume=1&sfile=RGBD_Individuals/$i.zip" \
        -O "$save_dir/RGBD_Images/$i.zip"

    unzip -o "$save_dir/RGBD_Images/$i.zip" -d "$save_dir/RGBD_Images"
    rm -f "$save_dir/RGBD_Images/$i.zip"
done

# Download and unzip Res_Individuals
for i in {01..10}; do
    wget --continue --no-check-certificate --load-cookies cookies.txt \
        --post-data "username=$username&password=$password" \
        "https://download.is.tue.mpg.de/download.php?domain=intercap&resume=1&sfile=Res_Individuals/$i.zip" \
        -O "$save_dir/Res/$i.zip"

    unzip -o "$save_dir/Res/$i.zip" -d "$save_dir/Res"
    rm -f "$save_dir/Res/$i.zip"
done

# Download code to read dataset
wget https://intercap.is.tue.mpg.de/media/upload/Code_to_Read_Data.zip \
    -O "$save_dir/Code_to_Read_Data.zip"