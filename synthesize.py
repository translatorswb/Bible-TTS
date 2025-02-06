import argparse
import os
import json

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import torch
import torchaudio

from pymcd.mcd import Calculate_MCD
from joblib import Parallel, delayed
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def nemo(meta_path):
    """
    Normalizes NeMo-style json manifest files to TTS format
    """
    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = json.loads(line)
            wav_file = cols["audio_filepath"]
            text = cols["text"]
            speaker_name = cols["speaker_name"] if "speaker_name" in cols else None
            language = cols["language"] if "language" in cols else None
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "language": language})
    return items


def load_xtts(xtts_checkpoint, xtts_config, xtts_vocab, speaker_audio_file):
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    model.to(DEVICE)

    print("Model loaded successfully!")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    print("Conditioning latents computed successfully!")

    return model, gpt_cond_latent, speaker_embedding


def xtts(model, gpt_cond_latent, speaker_embedding, text, file_path):
    out = model.inference(
        text=text,
        language="ha",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        # temperature=0.1,
        # length_penalty=1.0,
        # repetition_penalty=10.0,
        # top_k=10,
        # top_p=0.3,
    )
    torchaudio.save(file_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)


def synthesize(model_path, config_path, vocab_path, model_name, items, output_dir, speaker_audio_file, mono_speaker=False, manifest_name="synthesized_manifest.jsonl"):
    manifest_path = os.path.join(output_dir, manifest_name)

    if os.path.exists(manifest_path):
        print(f"Manifest file already exists at {manifest_path}")
        return manifest_path

    if model_path and vocab_path and speaker_audio_file:
        model, gpt_cond_latent, speaker_embedding = load_xtts(model_path, config_path, vocab_path, speaker_audio_file)
    elif model_path:        
        model = TTS(model_path=model_path, config_path=config_path).to(DEVICE)
    elif model_name:
        model = TTS(model_name=model_name, config_path=config_path).to(DEVICE)
    else:
        raise ValueError("Either model path or model name must be provided")

    os.makedirs(os.path.join(output_dir, "synthesized"), exist_ok=True)
    for i, item in enumerate(tqdm(items)):
        file_path = os.path.join(output_dir, "synthesized", os.path.basename(item["audio_file"]))

        # Check if synthesized file already exists
        if os.path.exists(file_path):
            items[i]["synthesized_file"] = os.path.abspath(file_path)
            continue

        speaker = item["speaker_name"] if not mono_speaker else None
        if model_path and vocab_path and speaker_audio_file:
            xtts(model, gpt_cond_latent, speaker_embedding, item["text"], file_path)
        else:
            model.tts_to_file(item["text"], file_path=file_path, speaker=speaker, language=item["language"])

        items[i]["synthesized_file"] = os.path.abspath(file_path)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Synthesized audio files and manifest saved to {output_dir}")

    return manifest_path


def calculate_mcd(manifest):
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
    with open(manifest, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    refs = [item["audio_file"] for item in items]
    synths = [item["synthesized_file"] for item in items]

    # Check if MCDs are already calculated
    if all("mcd" in item for item in items):
        mcds = [item["mcd"] for item in items]
        print("MCD values already calculated")
    else:
        mcds = Parallel(n_jobs=-1)(
            delayed(mcd_toolbox.calculate_mcd)(ref, synth) for ref, synth in zip(refs, synths)
            )
        # Save manifest with MCD values
        for i, item in enumerate(items):
            items[i]["mcd"] = mcds[i]

        with open(manifest, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    mean_mcd = sum(mcds) / len(mcds)

    return mean_mcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--config_path", type=str, help="Path to the config")
    parser.add_argument("--vocab_path", type=str, help="Path to the vocab file")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the synthesized audio")
    parser.add_argument("--speaker_audio_file", type=str, help="Path to the reference speaker audio file")
    parser.add_argument("--mono_speaker", action="store_true", help="Override speaker names in case the model is not multi-speaker")
    parser.add_argument("--calculate_mcd", action="store_true", help="Calculate Mel Cepstral Distortion")
    args = parser.parse_args()

    if args.model_path and not args.config_path:
        args.config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
        assert os.path.exists(args.config_path), "Config file not found, please provide the path to the config file"

    if args.model_path and not args.vocab_path:
        args.vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.json")
        assert os.path.exists(args.vocab_path), "Vocab file not found, please provide the path to the vocab file"

    items = nemo(args.manifest_path)
    print(f"Fetched {len(items)} items from the manifest file")

    out_manifest_path = synthesize(
               args.model_path,
               args.config_path,
               args.vocab_path,
               args.model_name,
               items,
               args.output_dir,
               args.speaker_audio_file,
               args.mono_speaker
    )

    if args.calculate_mcd:
        mean_mcd = calculate_mcd(out_manifest_path)
        print(f"Mean MCD: {mean_mcd}")
