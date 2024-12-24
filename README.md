# Bible-TTS

## Download the data
```
./download_data.sh
```

## Preprocess the data
```
python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -r requirements.txt
./prepare_data.sh
```

## Forced Alignment

```
docker build -t ctc-aligner -f align.dockerfile .
docker run --gpus all -v $PWD/text_by_chapter/processed/:/app/text_dir \
                      -v $PWD/chapters/:/app/audio_dir \
                      -v $PWD/data/:/app/output_dir \
                      ctc-aligner /app/text_dir /app/audio_dir /app/output_dir
sudo chown -R $USER:$USER data/
```
