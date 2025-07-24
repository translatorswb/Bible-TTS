import argparse
import os
import pandas as pd
from synthesize import synthesize
from joblib import Parallel, delayed
import numpy as np


def process_chunk(idx, chunk, model_path, config_path, vocab_path, model_name, language, speaker, output_dir, speaker_audio_file, device, suffix):
    suffix = f"_{suffix}" if suffix else ""
    items = []
    for _, row in chunk.iterrows():
        items.append({
            "text": row["Text"],
            "audio_file": row["ID"] + suffix + ".wav",
            "speaker_name": speaker,
            "language": language
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
        device,
        manifest_name=manifest_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--config_path", type=str, help="Path to the config")
    parser.add_argument("--vocab_path", type=str, help="Path to the vocab file")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--language", type=str, help="Language code")
    parser.add_argument("--speaker", type=str, help="Speaker name")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the csv file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the synthesized audio")
    parser.add_argument("--speaker_audio_file", type=str, help="Path to the reference speaker audio file")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the synthesized audio file name")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cpu', 'cuda:0')")
    args = parser.parse_args()

    if args.model_path and not args.config_path:
        args.config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
        assert os.path.exists(args.config_path), "Config file not found, please provide the path to the config file"

    df = pd.read_csv(args.csv_path)
    print(f"Fetched {len(df)} items from the csv file")

    # shuffle the dataframe
    df = df.sample(frac=1, random_state=42)

    chunks = np.array_split(df, args.n_jobs)
    Parallel(n_jobs=args.n_jobs)(delayed(process_chunk)(
        i,
        chunk,
        args.model_path,
        args.config_path,
        args.vocab_path,
        args.model_name,
        args.language,
        args.speaker,
        args.output_dir,
        args.speaker_audio_file,
        args.device,
        args.suffix
    ) for i, chunk in enumerate(chunks))
