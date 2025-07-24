import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.managers import save_file
from tqdm import tqdm
import json
import gdown
import tarfile

torch.set_num_threads(24)


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
            speaker_name = cols["speaker_name"] if "speaker_name" in cols else "one"
            language = cols["language"] if "language" in cols else ""
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "language": language, "root_path": root_path})
    return items


def compute_embeddings(
    model_path,
    config_path,
    output_path,
    old_speakers_file=None,
    old_append=False,
    config_dataset_path=None,
    formatter=None,
    dataset_name=None,
    dataset_path=None,
    meta_file_train=None,
    meta_file_val=None,
    disable_cuda=False,
    no_eval=False,
):
    use_cuda = torch.cuda.is_available() and not disable_cuda

    if config_dataset_path is not None:
        c_dataset = load_config(config_dataset_path)
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not no_eval)
    else:
        c_dataset = BaseDatasetConfig()
        c_dataset.dataset_name = dataset_name
        c_dataset.path = dataset_path
        if meta_file_train is not None:
            c_dataset.meta_file_train = meta_file_train
        if meta_file_val is not None:
            c_dataset.meta_file_val = meta_file_val
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=not no_eval, formatter=formatter)

    if meta_data_eval is None:
        samples = meta_data_train
    else:
        samples = meta_data_train + meta_data_eval

    encoder_manager = SpeakerManager(
        encoder_model_path=model_path,
        encoder_config_path=config_path,
        d_vectors_file_path=old_speakers_file,
        use_cuda=use_cuda,
    )

    class_name_key = encoder_manager.encoder_config.class_name_key

    # compute speaker embeddings
    if old_speakers_file is not None and old_append:
        speaker_mapping = encoder_manager.embeddings
    else:
        speaker_mapping = {}

    for fields in tqdm(samples):
        class_name = fields[class_name_key]
        audio_file = fields["audio_file"]
        embedding_key = fields["audio_unique_name"]

        # Only update the speaker name when the embedding is already in the old file.
        if embedding_key in speaker_mapping:
            speaker_mapping[embedding_key]["name"] = class_name
            continue

        if old_speakers_file is not None and embedding_key in encoder_manager.clip_ids:
            # get the embedding from the old file
            embedd = encoder_manager.get_embedding_by_clip(embedding_key)
        else:
            # extract the embedding
            embedd = encoder_manager.compute_embedding_from_clip(audio_file)

        # create speaker_mapping if target dataset is defined
        speaker_mapping[embedding_key] = {}
        speaker_mapping[embedding_key]["name"] = class_name
        speaker_mapping[embedding_key]["embedding"] = embedd

    if speaker_mapping:
        # save speaker_mapping if target dataset is defined
        if os.path.isdir(output_path):
            mapping_file_path = os.path.join(output_path, "speakers.pth")
        else:
            mapping_file_path = output_path

        if os.path.dirname(mapping_file_path) != "":
            os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

        save_file(speaker_mapping, mapping_file_path)
        print("Speaker embeddings saved at:", mapping_file_path)


LANG_TO_ISO = {
    "luo": "luo",
    "chichewa": "nya"
}

subdirs = [d for d in os.listdir() if os.path.isdir(d) and d.startswith('yourtts')]
OUT_PATH = subdirs[0]
LANG_NAME = OUT_PATH.split('_')[1]
ISO = LANG_TO_ISO[LANG_NAME]

# Name of the run for the Trainer
RUN_NAME = f"YourTTS-{LANG_NAME.capitalize()}"

# If you want to do transfer learning and speedup your training you can set here the path to the CML-TTS available checkpoint that can be downloaded here:  https://drive.google.com/u/2/uc?id=1yDCSJ1pFZQTHhL09GMbOrdjcPULApa0p
RESTORE_PATH = os.path.join(OUT_PATH, "checkpoints_yourtts_cml_tts_dataset/best_model.pth")

URL = "https://drive.google.com/u/2/uc?id=1yDCSJ1pFZQTHhL09GMbOrdjcPULApa0p"
OUTPUT_CHECKPOINTS_FILEPATH = os.path.join(OUT_PATH, "checkpoints_yourtts_cml_tts_dataset.tar.bz")

# Download the CML-TTS checkpoint if it does not exist
if not os.path.exists(RESTORE_PATH):
    print(f"Downloading the CML-TTS checkpoint from {URL}")
    gdown.download(url=URL, output=OUTPUT_CHECKPOINTS_FILEPATH, quiet=False, fuzzy=True)
    with tarfile.open(OUTPUT_CHECKPOINTS_FILEPATH, "r:bz2") as tar:
        tar.extractall(OUT_PATH)
else:
    print(f"Checkpoint already exists at {RESTORE_PATH}")

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 4

# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 24000

# Max audio length in seconds to be used in training
MAX_AUDIO_LEN_IN_SECONDS = 12
# Min audio length in seconds to be used in training
MIN_AUDIO_LEN_IN_SECONDS = 0.8

dataset_conf = BaseDatasetConfig(
    dataset_name=f"{ISO}_openbible",
    meta_file_train="manifest_train.jsonl",
    meta_file_val="manifest_dev.jsonl",
    language=ISO,
    path="data"
)

### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Checks if the speakers embeddings are already computated, if not compute it
embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
if not os.path.isfile(embeddings_file):
    print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
    compute_embeddings(
        SPEAKER_ENCODER_CHECKPOINT_PATH,
        SPEAKER_ENCODER_CONFIG_PATH,
        embeddings_file,
        formatter=nemo,
        dataset_name=dataset_conf.dataset_name,
        dataset_path=dataset_conf.path,
        meta_file_train=dataset_conf.meta_file_train,
        meta_file_val=dataset_conf.meta_file_val,
    )
D_VECTOR_FILES.append(embeddings_file)

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    spec_segment_size=62,
    hidden_channels=192,
    hidden_channels_ffn_text_encoder=768,
    num_heads_text_encoder=2,
    num_layers_text_encoder=10,
    kernel_size_text_encoder=3,
    dropout_p_text_encoder=0.1,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=False,
    # Useful parameters to enable multilingual training
    use_language_embedding=True,
    embedded_language_dim=4,
)

LUO_CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'ʼ']
LUO_PUNCT = [' ', '!', "'", ',', '.', ':', ';', '?', '’']

CHICHEWA_CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'ʼ']
CHICHEWA_PUNCT = [' ', '!', ',', '.', ':', ';', '?']

CHARS = {
    "luo": LUO_CHARS,
    "chichewa": CHICHEWA_CHARS
}

PUNCT = {
    "luo": LUO_PUNCT,
    "chichewa": CHICHEWA_PUNCT
}

LUO_TEST_SENTENCES = [
    "jo kolosai achiel.",
    "magoyo erokamano ni wuoro ka un gi mor.",
    "epafra bende nonyisowa kuom hera ma roho maler osemiyou."
    ]

CHICHEWA_TEST_SENTENCES = [
    "umene unafika kwa inu.",
    "tukiko adzakuwuzani zonse za ine.",
    "iye anachita mtendere kudzera mʼmagazi ake, wokhetsedwa pa mtanda."
    ]

TEST_SENTENCES = {
    "luo": [[text, "one", None, "luo"] for text in LUO_TEST_SENTENCES],
    "chichewa": [[text, "one", None, "nya"] for text in CHICHEWA_TEST_SENTENCES]
    }

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description=f"""
            - YourTTS trained using the {LANG_NAME.capitalize()} OpenBible dataset.
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=4,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    # eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    # log_model_step=1000,
    save_step=1000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=True,
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="no_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="".join(CHARS[LANG_NAME]),
        punctuations="".join(PUNCT[LANG_NAME]),
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=[dataset_conf],
    cudnn_benchmark=False,
    min_audio_len=int(SAMPLE_RATE * MIN_AUDIO_LEN_IN_SECONDS),
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=True,
    test_sentences=TEST_SENTENCES[LANG_NAME],
    # Enable the weighted sampler
    # use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    # weighted_sampler_attrs={"language": 1.0, "speaker_name": 1.0},
    # weighted_sampler_attrs={"language": 1.0},
    # weighted_sampler_multipliers={
    #     # "speaker_name": {
    #     # you can force the batching scheme to give a higher weight to a certain speaker and then this speaker will appears more frequently on the batch.
    #     # It will speedup the speaker adaptation process. Considering the CML train dataset and "new_speaker" as the speaker name of the speaker that you want to adapt.
    #     # The line above will make the balancer consider the "new_speaker" as 106 speakers so 1/4 of the number of speakers present on CML dataset.
    #     # 'new_speaker': 106, # (CML tot. train speaker)/4 = (424/4) = 106
    #     # }
    # },
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the YourTTS paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    formatter=nemo,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
