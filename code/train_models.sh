#!/bin/bash

if [[ -z "${KAGGLE_USERNAME}" || -z "${KAGGLE_KEY}" ]]; then
  echo "Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set."
  exit 1
fi

mkdir -p ~/.kaggle
cat <<EOF > ~/.kaggle/kaggle.json
{
  "username": "${KAGGLE_USERNAME}",
  "key": "${KAGGLE_KEY}"
}
EOF

chmod 600 ~/.kaggle/kaggle.json

if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

DATASET_NAME="manjilkarki/deepfake-and-real-images"
DATASET_FILE="deepfake-and-real-images.zip"
DATA_DIR="data_test"

mkdir -p $DATA_DIR

echo "Downloading dataset..."
kaggle datasets download -d ${DATASET_NAME} -p $DATA_DIR

echo "Unzipping dataset..."
unzip -q $DATA_DIR/${DATASET_FILE} -d $DATA_DIR

for model in "RESNET50" "EFFICIENTNET"; do
    for i in 1 3 5 10; do
        python3 train.py --train_path ${DATA_DIR}/Dataset/Train --test_path ${DATA_DIR}/Dataset/Test --model_out model_${model}_${i}_epochs.pth --model_name ${model}
    done
done