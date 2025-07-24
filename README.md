# Bible-TTS

## Create a virtual environment and install the dependencies
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Download the data and unzip archives (Hausa)
```bash
./download_data.sh data/hausa/
./unzip_data.sh data/hausa
```

## Preprocess the data (Hausa)
```bash
./prepare_data.sh data/hausa/text_by_book/release/USX_2 hausa
```

## Preprocess the data (Luo)
```bash
./prepare_data.sh data/luo/text_by_book/release/USX_1 luo
```

## Forced Alignment (Hausa)

```bash
docker build -t ctc-aligner -f align.dockerfile .
docker run --rm --runtime=nvidia --gpus '"device=0"' -v $PWD/data/hausa/text_by_chapter/processed:/app/text_dir \
                      -v $PWD/data/hausa/chapters:/app/audio_dir \
                      -v $PWD/data/hausa/tts_data:/app/data \
                      ctc-aligner /app/text_dir /app/audio_dir /app/data hausa
sudo chown -R $USER:$USER data/hausa/tts_data
```

## Forced Alignment (Luo)

```bash
docker run --rm --runtime=nvidia --gpus '"device=0"' -v $PWD/data/luo/text_by_chapter/processed:/app/text_dir \
                      -v $PWD/data/luo/chapters:/app/audio_dir \
                      -v $PWD/data/luo/tts_data:/app/data \
                      ctc-aligner /app/text_dir /app/audio_dir /app/data luo --sample_rate 24000
sudo chown -R $USER:$USER data/luo/tts_data
```

## Forced Alignment (Chichewa)

```bash
docker run --rm --runtime=nvidia --gpus '"device=0"' -v $PWD/data/chichewa/text_by_chapter/processed:/app/text_dir \
                      -v $PWD/data/chichewa/chapters:/app/audio_dir \
                      -v $PWD/data/chichewa/tts_data:/app/data \
                      ctc-aligner /app/text_dir /app/audio_dir /app/data chichewa --sample_rate 24000
sudo chown -R $USER:$USER data/chichewa/tts_data
```
You can use the `resample.py` script to resample the audio files to 22 kHz needed by the XTTS model. The YourTTS checkpoint we use was trained on 24 kHz audio files.

## Isolate speakers and generate train, dev, and test splits (Hausa)

```bash
python process_manifest.py data/hausa/tts_data/manifest.jsonl hausa -55
```
Replace `-55` with the threshold you want to use to filter out unaligned segments. We use a threshold of `-140` for Luo and `-45` for Chichewa.

For Chichewa, we also use the `--drop_numbers` flag to remove samples containing numbers from the dataset since they've not been converted to words.

## Train VITS model (Hausa)

```bash
docker build -t bible-tts -f train_vits.dockerfile .
docker run --ipc=host --runtime=nvidia --gpus all -v $PWD/data:/app/data \
                      -v $PWD/vits_hausa/:/app/vits_hausa \
                      bible-tts \
                      --coqpit.batch_size 26 \
                      --coqpit.eval_batch_size 26 \
                      --coqpit.batch_group_size 5 \
                      --coqpit.max_audio_len 264600
```

## Train YourTTS model (Luo)

```bash
docker build -t train_yourtts -f train_yourtts.dockerfile .
docker run --ipc=host --runtime=nvidia --gpus all -v $PWD/data/luo/tts_data:/app/data \
                      -v $PWD/yourtts_luo/:/app/yourtts_luo \
                      train_yourtts --restore_path /app/yourtts_luo/checkpoints_yourtts_cml_tts_dataset/best_model.pth \
                      --coqpit.batch_size 20 \
                      --coqpit.eval_batch_size 20
```

## Train XTTS model (Luo)

### Prepare data
```bash
python convert_to_coqui.py data/luo/tts_data_22khz/manifest_train.jsonl
python convert_to_coqui.py data/luo/tts_data_22khz/manifest_dev.jsonl
python convert_to_coqui.py data/luo/tts_data_22khz/manifest_test.jsonl
```

### DVAE fine-tuning
```bash
docker build -t train_dvae -f train_dvae.dockerfile .
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' -v $PWD/data/luo/tts_data_22khz:/app/data \
                      -v $PWD/xtts_luo/:/app/xtts_luo \
                        train_dvae -o /app/xtts_luo -l luo
```

### GPT2 fine-tuning
```bash
docker build -t train_gpt -f train_gpt.dockerfile .
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' -v $PWD/data/luo/tts_data_22khz:/app/data \
                      -v $PWD/xtts_luo:/app/xtts_luo \
                      train_gpt
```

## Evaluate an XTTS checkpoint on the test set
```bash
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
```bash
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' \
                      -v $PWD/xtts_hausa/:/app/xtts_hausa \
                      -v $PWD/data:/app/data \
                      evaluate evaluate_run.py \
                      --run_dir /app/xtts_hausa/path/to/run/dir/ \
                      --manifest_path /app/data/manifest_dev.jsonl \
                      --speaker_audio_file /app/data/clips/path/to/audio/file.wav
```

## Evaluate a VITS training run on the dev set
```bash
docker build -t evaluate_vits -f evaluate_vits.dockerfile .
docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' \
                      -v $PWD/vits_hausa/:/app/vits_hausa \
                      -v $PWD/data:/app/data \
                      evaluate_vits evaluate_run.py \
                      --run_dir /app/vits_hausa/path/to/run/dir/ \
                      --manifest_path /app/data/manifest_dev.jsonl
```
