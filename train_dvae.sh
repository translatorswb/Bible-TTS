# Download the pre-trained model
python download_checkpoint.py --output_path xtts_hausa/

# Extend the vocabulary and adjust the configuration
python extend_vocab_config.py --output_path=xtts_hausa/ --metadata_path data/manifest_train.csv --language ha --extended_vocab_size 2000

# Train the model
CUDA_VISIBLE_DEVICES=0,1 python train_dvae_xtts.py --output_path=xtts_hausa/ \
                        --train_csv_path=data/manifest_train.csv \
                        --eval_csv_path=data/manifest_dev.csv \
                        --language="ha" \
                        --num_epochs=5 \
                        --batch_size=128 \
                        --lr=5e-6
