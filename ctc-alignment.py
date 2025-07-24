import argparse
import os
import torch
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
from tqdm import tqdm
import json
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

LANG_TO_ISO = {
    "hausa": "ha",
    "luo": "luo",
    "chichewa": "nya"
}

def process_timestamps(timestamps, base_name_no_ext, audio_file, output_dir, sample_rate):
    # Load audio file
    audio = AudioSegment.from_file(audio_file)

    # Audio should be in mono, sample_rate Hz
    if audio.channels != 1 or audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate).set_channels(1)

    # Prepare manifest data
    manifest_data = []

    # Process each segment
    for i, segment in enumerate(timestamps):
        # Extract the audio segment
        text = segment['text']
        score = segment['score']
        start_time = segment['start']
        end_time = segment['end']
        duration = end_time - start_time
        segment = audio[start_time * 1000:end_time * 1000]

        # Generate output filename
        filename = f"{base_name_no_ext}_{i:03}.wav"
        output_path = os.path.join(output_dir, 'clips', filename)

        # Export audio segment
        segment.export(output_path, format="wav")

        # Add entry to manifest
        manifest_data.append({
            "audio_filepath": output_path,
            "text": text,
            "duration": duration,
            "score": score
        })

    # Write manifest file
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, 'a', encoding='utf-8') as f:
        for entry in manifest_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def align_audio(audio_path, text_path, alignment_model, alignment_tokenizer, language, star_frequency):
    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)

    emissions, stride = generate_emissions(
        alignment_model,
        audio_waveform,
    )

    with open(text_path, "r") as f:
        lines = f.readlines()

    # if a line doesn't end with punctuation, add a period
    lines = [line.strip() + "." if not line.strip().endswith(('!', "'", ',', '.', ':', ';', '?', '’')) else line.strip() for line in lines]

    text = " ".join(lines)
    text = text.lower()

    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
        split_size="sentence",
        star_frequency=star_frequency,
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    return word_timestamps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CTC forced alignment and create NeMo manifests')
    parser.add_argument('text_dir', help='Path to the directory containing text files')
    parser.add_argument('audio_dir', help='Path to the directory containing audio files')
    parser.add_argument('output_dir', help='Path to the output directory')
    parser.add_argument('language', help='Language of the audio files', choices=['hausa', 'luo', 'chichewa'])
    parser.add_argument('--star_frequency', default='edges', help='Frequency of <star> token')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for audio processing')

    args = parser.parse_args()

    lang = LANG_TO_ISO[args.language]

    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'clips'), exist_ok=True)

    # Check whether manifest file already exists
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    if os.path.exists(manifest_path):
        raise FileExistsError(f"Manifest file already exists in {manifest_path}")

    # Get list of files in the timing directory
    text_files = os.listdir(args.text_dir)

    # Process each timing file
    for text_file in tqdm(text_files):
        base_name = os.path.basename(text_file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        text_file = os.path.join(args.text_dir, base_name)
        audio_file = os.path.join(args.audio_dir, base_name_no_ext + '.mp3')

        # Run CTC forced alignment
        timestamps = align_audio(
            audio_file,
            text_file,
            alignment_model,
            alignment_tokenizer,
            lang,
            args.star_frequency
            )

        # Cut audio segments and create manifest
        process_timestamps(timestamps, base_name_no_ext, audio_file, args.output_dir, args.sample_rate)
