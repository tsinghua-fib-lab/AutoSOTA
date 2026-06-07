#!/bin/bash

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nYou need to register at https://prox.is.tue.mpg.de/"
read -p "Username: " username
read -p "Password: " password

# Set save directory (hardcoded)
save_dir="data/PROX/data"

username=$(urle $username)
password=$(urle $password)

mkdir -p "$save_dir"

# Download
wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=prox&resume=1&sfile=quantitative.zip' \
    -O "$save_dir/quantitative.zip" \
    --no-check-certificate --continue

unzip data/PROX/data/quantitative.zip -d data/PROX/data
rm -f data/PROX/data/quantitative.zip