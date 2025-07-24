import argparse
import os
import json


def convert_to_coqui(manifest_path, max_duration=12.0, min_duration=0.8):
    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Filter out audio files with duration greater than max_duration
    org_len = len(data)
    data = [entry for entry in data if entry['duration'] <= max_duration and entry['duration'] >= min_duration]
    print(f"Filtered out {org_len - len(data)} audio files with duration greater than {max_duration} seconds or less than {min_duration} seconds")

    # Write manifest in Coqui csv format: audio_file|text|speaker_name
    coqui_manifest_path = os.path.splitext(manifest_path)[0] + ".csv"
    with open(coqui_manifest_path, 'w', encoding='utf-8') as f:
        f.write("audio_file|text|speaker_name\n")
        for entry in data:
            speaker_name = entry['speaker_name'] if 'speaker_name' in entry else 'one'
            f.write(f"{entry['audio_filepath']}|{entry['text']}|{speaker_name}\n")
    print(f"Coqui manifest saved to: {coqui_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert manifest to Coqui TTS format')
    parser.add_argument('manifest_path', help='Path to the manifest file')
    parser.add_argument('--max_duration', help='Maximum duration of the audio files in seconds', type=float, default=12.0)
    parser.add_argument('--min_duration', help='Minimum duration of the audio files in seconds', type=float, default=0.8)
    args = parser.parse_args()
    convert_to_coqui(args.manifest_path, args.max_duration)
