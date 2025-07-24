#!/bin/bash

# Parse command line arguments
while getopts o:l: flag
do
    case "${flag}" in
        o) output_path=${OPTARG};;
        l) language=${OPTARG};;
    esac
done

# Download the pre-trained model
python download_checkpoint.py --output_path $output_path

# Extend the vocabulary and adjust the configuration
python extend_vocab_config.py --output_path=$output_path --metadata_path data/manifest_train.csv --language $language --extended_vocab_size 2000

# Train the model
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py --output_path=$output_path \
                        --train_csv_path=data/manifest_train.csv \
                        --eval_csv_path=data/manifest_dev.csv \
                        --language=$language \
                        --num_epochs=10 \
                        --batch_size=128 \
                        --lr=5e-6
