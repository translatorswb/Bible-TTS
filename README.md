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
docker run --rm --runtime=nvidia --gpus all -v $PWD/text_by_chapter/processed:/app/text_dir \
                      -v $PWD/chapters:/app/audio_dir \
                      -v $PWD/data:/app/data \
                      ctc-aligner /app/text_dir /app/audio_dir /app/data
sudo chown -R $USER:$USER data/
```

## Isolate speakers and generate train, dev, and test splits

```
python process_manifest.py data/manifest.jsonl -55
```
Replace `-55` with the threshold you want to use to filter out unaligned segments.

## Train VITS model

```
docker build -t bible-tts -f train_vits.dockerfile .
docker run --ipc=host --runtime=nvidia --gpus all -v $PWD/data:/app/data \
                      -v $PWD/vits_hausa/:/app/vits_hausa \
                      bible-tts \
                      --coqpit.batch_size 26 \
                      --coqpit.eval_batch_size 26 \
                      --coqpit.batch_group_size 5 \
                      --coqpit.max_audio_len 264600
```

## Train xTTS model

### Prepare data
```
python convert_to_coqui.py data/manifest_train.jsonl
python convert_to_coqui.py data/manifest_dev.jsonl
python convert_to_coqui.py data/manifest_test.jsonl
```

### DVAE fine-tuning
```
docker build -t train_dvae -f train_dvae.dockerfile .
docker run --ipc=host --runtime=nvidia --gpus all -v $PWD/data:/app/data \
                      -v $PWD/xtts_hausa/:/app/xtts_hausa \
                        train_dvae
```

### GPT2 fine-tuning
```
docker build -t train_gpt -f train_gpt.dockerfile .
docker run --ipc=host --runtime=nvidia --gpus all -v $PWD/data:/app/data \
                      -v $PWD/xtts_hausa/:/app/xtts_hausa \
                      train_gpt
```

## Evaluate an XTTS checkpoint on the test set
```
docker build -t evaluate -f evaluate.dockerfile .
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' \
                      -v $PWD/xtts_hausa/:/app/xtts_hausa \
                      -v $PWD/data:/app/data \
                      -v $PWD/evaluation:/app/evaluation \
                      evaluate synthesize.py \
                      --model_path /app/xtts_hausa/path/to/checkpoint.pth \
                      --manifest_path /app/data/manifest_test.jsonl \
                      --output_dir /app/evaluation/xtts_hausa_test \
                      --speaker_audio_file /app/data/clips/path/to/audio/file.wav \
                      --calculate_mcd
```

## Evaluate an XTTS training run on the dev set
```
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' \
                      -v $PWD/xtts_hausa/:/app/xtts_hausa \
                      -v $PWD/data:/app/data \
                      evaluate evaluate_run.py \
                      --run_dir /app/xtts_hausa/path/to/run/dir/ \
                      --manifest_path /app/data/manifest_dev.jsonl \
                      --speaker_audio_file /app/data/clips/path/to/audio/file.wav
```

## Evaluate a VITS training run on the dev set
```
docker build -t evaluate_vits -f evaluate_vits.dockerfile .
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' \
                      -v $PWD/vits_hausa/:/app/vits_hausa \
                      -v $PWD/data:/app/data \
                      evaluate_vits evaluate_run.py \
                      --run_dir /app/vits_hausa/path/to/run/dir/ \
                      --manifest_path /app/data/manifest_dev.jsonl
```
