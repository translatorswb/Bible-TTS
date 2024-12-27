from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

import os
import json

output_path = "vits_hausa"

dataset_config = BaseDatasetConfig(
    meta_file_train="manifest_train.jsonl", meta_file_val="manifest_dev.jsonl", language="ha", path="output_dir"
)

audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

vitsArgs = VitsArgs(
    use_speaker_embedding=True,
)

CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'W', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'w', 'y', 'z', 'ā', 'ă', 'ū', 'Ɓ', 'Ɗ', 'Ƙ', 'ƙ', 'ɓ', 'ɗ', '’']
PUNCT = [' ', '!', "'", ',', '.', ':', ';', '?']

character_config = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    characters="".join(CHARS),
    punctuations="".join(PUNCT),
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_openbible_hausa",
    run_description="vits_openbible_hausa",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=48,
    num_loader_workers=12,
    num_eval_loader_workers=12,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="no_cleaners",
    use_phonemes=False,
    characters=character_config,
    compute_input_seq_cache=True,
    print_step=25,
    target_loss="loss_1",
    print_eval=True,
    save_all_best=True,
    save_n_checkpoints=10,
    save_step=5000,
    mixed_precision=True,
    max_audio_len=23 * 22050,
    start_by_longest=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences=[
        ["Umarnai don zaman tsarki", "two", None, "ha"],
        ["wanda kuma ya faɗa mana ƙaunar da kuke yi cikin Ruhu.", "one", None, "ha"],
        ["Gama mun ji labarin bangaskiyarku a cikin Yesu Kiristi da kuma ƙaunar da kuke yi saboda dukan tsarkaka.", "two", None, "ha"],
    ]
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.

def nemo(root_path, meta_file, **kwargs):
    """
    Normalizes NeMo-style json manifest files to TTS format
    """
    meta_path = os.path.join(root_path, meta_file)
    items = []
    with open(meta_path, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = json.loads(line)
            wav_file = cols["audio_filepath"]
            text = cols["text"]
            speaker_name = cols["speaker_name"] if "speaker_name" in cols else "nemo"
            language = cols["language"] if "language" in cols else ""
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "language": language, "root_path": root_path})
    return items

train_samples, eval_samples = load_tts_samples(dataset_config, formatter=nemo)
print(f"Loaded {len(train_samples)} train samples")
print(f"Loaded {len(eval_samples)} eval samples")

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

# init model
model = Vits(config, ap, tokenizer, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
