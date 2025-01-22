import argparse
from synthesize import nemo, synthesize, calculate_mcd
import torch
import os
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, help="Path to the training run directory")
    parser.add_argument("--manifest_path", type=str, help="Path to the manifest file")
    parser.add_argument("--speaker_audio_file", type=str, help="Path to the speaker audio file")

    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    vocab_path = os.path.join(args.run_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = None

    items = nemo(args.manifest_path)
    print(f"Fetched {len(items)} items from the manifest file")

    checkpoints = [f for f in os.listdir(args.run_dir) if f.endswith(".pth") and f.split("_")[-1].split(".")[0].isdigit()]
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    mcds = []
    for checkpoint in tqdm(checkpoints):
        model_path = os.path.join(args.run_dir, checkpoint)
        output_dir = os.path.join(args.run_dir, "eval", checkpoint.split(".")[0])

        out_manifest_path = synthesize(
            model_path,
            config_path,
            vocab_path,
            None,
            items,
            output_dir,
            args.speaker_audio_file
        )

        mcd = calculate_mcd(out_manifest_path)
        mcds.append(mcd)

        print(f"Checkpoint: {checkpoint}, MCD: {mcd}")

    # Save a csv file with the MCD values
    with open(os.path.join(args.run_dir, "eval", "mcd.csv"), "w", encoding="utf-8") as f:
        f.write("checkpoint,mcd\n")
        for checkpoint, mcd in zip(checkpoints, mcds):
            f.write(f"{checkpoint},{mcd}\n")
