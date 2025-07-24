import argparse
import glob
import os
from argparse import RawTextHelpFormatter
from multiprocessing import Pool

import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def resample_file(func_args):
    input_file, output_file, output_sr = func_args
    y, sr = librosa.load(input_file, sr=output_sr)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sf.write(output_file, y, sr)


def resample_files(input_dir=None, output_sr=22050, output_dir=None, file_ext="wav", n_jobs=10, csv_file=None):
    if csv_file:
        # Read audio files from CSV
        print("Reading audio file paths from CSV...")
        df = pd.read_csv(csv_file)
        if 'audio_filepath' not in df.columns:
            raise ValueError("CSV file must have a column named 'audio_filepath'")
        input_files = df['audio_filepath'].tolist()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_files = [os.path.join(output_dir, os.path.basename(f)) for f in input_files]
        else:
            output_files = input_files  # In-place resampling
    else:
        print("Finding audio files to resample...")
        input_files = glob.glob(os.path.join(input_dir, f"**/*.{file_ext}"), recursive=True)
        print(f"Found {len(input_files)} files...")
        
        if output_dir:
            # Create relative paths for output
            rel_paths = [os.path.relpath(f, input_dir) for f in input_files]
            output_files = [os.path.join(output_dir, rel_path) for rel_path in rel_paths]
        else:
            output_files = input_files  # In-place resampling
    
    print("Resampling audio files...")
    resample_args = list(zip(input_files, output_files, [output_sr] * len(input_files)))
    with Pool(processes=n_jobs) as p:
        with tqdm(total=len(resample_args)) as pbar:
            for _, _ in enumerate(p.imap_unordered(resample_file, resample_args)):
                pbar.update()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Resample a folder recursively with librosa or from a CSV file
                       Can be used in place or create a copy of the folder as an output.\n\n
                       Example runs:
                            python resample.py
                                --input_dir /root/LJSpeech-1.1/
                                --output_sr 22050
                                --output_dir /root/resampled_LJSpeech-1.1/
                                --file_ext wav
                                --n_jobs 24
                                
                            python resample.py
                                --csv_file /path/to/metadata.csv
                                --output_sr 22050
                                --output_dir /root/resampled_files/
                                --n_jobs 24
                    """,
        formatter_class=RawTextHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path of the folder containing the audio files to resample",
    )
    input_group.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to a CSV file with an 'audio_filepath' column containing paths to audio files",
    )

    parser.add_argument(
        "--output_sr",
        type=int,
        default=22050,
        required=False,
        help="Samlple rate to which the audio files should be resampled",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=False,
        help="Path of the destination folder. If not defined, the operation is done in place",
    )

    parser.add_argument(
        "--file_ext",
        type=str,
        default="wav",
        required=False,
        help="Extension of the audio files to resample (only used with --input_dir)",
    )

    parser.add_argument(
        "--n_jobs", type=int, default=None, help="Number of threads to use, by default it uses all cores"
    )

    args = parser.parse_args()

    resample_files(
        input_dir=args.input_dir,
        output_sr=args.output_sr,
        output_dir=args.output_dir,
        file_ext=args.file_ext,
        n_jobs=args.n_jobs,
        csv_file=args.csv_file
    )
