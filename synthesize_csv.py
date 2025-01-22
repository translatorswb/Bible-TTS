import argparse
import os
import pandas as pd
from synthesize import synthesize


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
    args = parser.parse_args()

    if args.model_path and not args.config_path:
        args.config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
        assert os.path.exists(args.config_path), "Config file not found, please provide the path to the config file"

    if args.model_path and not args.vocab_path:
        args.vocab_path = os.path.join(os.path.dirname(args.model_path), "vocab.json")
        assert os.path.exists(args.vocab_path), "Vocab file not found, please provide the path to the vocab file"

    df = pd.read_csv(args.csv_path)
    print(f"Fetched {len(df)} items from the csv file")

    eval_1 = df[df["Evaluator"] == 'evaluator1']
    eval_2 = df[df["Evaluator"] == 'evaluator2']

    items_1 = []
    for _, row in eval_1.iterrows():
        items_1.append({
            "text": row["Text"],
            "audio_file": row["ID"] + ".wav",
            "speaker_name": None,
            "language": "ha"
        })

    items_2 = []
    for _, row in eval_2.iterrows():
        items_2.append({
            "text": row["Text"],
            "audio_file": row["ID"] + ".wav",
            "speaker_name": None,
            "language": "ha"
        })

    synthesize(
        args.model_path,
        args.config_path,
        args.vocab_path,
        args.model_name,
        items_1,
        os.path.join(args.output_dir, "evaluator1"),
        args.speaker_audio_file,
        args.mono_speaker
    )

    synthesize(
        args.model_path,
        args.config_path,
        args.vocab_path,
        args.model_name,
        items_2,
        os.path.join(args.output_dir, "evaluator2"),
        args.speaker_audio_file,
        args.mono_speaker
    )
