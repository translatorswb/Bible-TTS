import argparse
import os
import pandas as pd
from synthesize import synthesize
from joblib import Parallel, delayed
import numpy as np


def process_chunk(idx, chunk, model_path, config_path, vocab_path, model_name, output_dir, speaker_audio_file, mono_speaker):
    items = []
    for _, row in chunk.iterrows():
        items.append({
            "text": row["Text"],
            "audio_file": row["ID"] + ".wav",
            "speaker_name": None,
            "language": "ha"
        })

    manifest_name = f"synthesized_manifest_{idx}.jsonl"

    synthesize(
        model_path,
        config_path,
        vocab_path,
        model_name,
        items,
        output_dir,
        speaker_audio_file,
        mono_speaker,
        manifest_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--config_path", type=str, help="Path to the config")
    parser.add_argument("--vocab_path", type=str, help="Path to the vocab file")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the csv file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the synthesized audio")
    parser.add_argument("--speaker_audio_file", type=str, help="Path to the reference speaker audio file")
    parser.add_argument("--mono_speaker", action="store_true", help="Override speaker names in case the model is not multi-speaker")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()

    if args.model_path and not args.config_path:
        args.config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
        assert os.path.exists(args.config_path), "Config file not found, please provide the path to the config file"

    if args.model_path and not args.vocab_path:
        args.vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.json")
        assert os.path.exists(args.vocab_path), "Vocab file not found, please provide the path to the vocab file"

    df = pd.read_csv(args.csv_path)

    # shuffle the dataframe
    df = df.sample(frac=1)

    print(f"Fetched {len(df)} items from the csv file")

    chunks = np.array_split(df, args.n_jobs)
    Parallel(n_jobs=args.n_jobs)(delayed(process_chunk)(
        i,
        chunk, 
        args.model_path, 
        args.config_path, 
        args.vocab_path, 
        args.model_name, 
        args.output_dir, 
        args.speaker_audio_file, 
        args.mono_speaker
    ) for i, chunk in enumerate(chunks))
