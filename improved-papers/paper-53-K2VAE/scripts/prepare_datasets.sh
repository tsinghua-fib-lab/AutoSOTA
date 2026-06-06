# Check if gdown is installed
if pip show gdown > /dev/null 2>&1; then
    echo "gdown is already installed, skipping installation."
else
    echo "gdown is not installed, installing..."
    pip install gdown
fi

python probts/utils/download_datasets.py --data_path $1