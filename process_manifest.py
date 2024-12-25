import argparse
import os
import json


def process_manifest(manifest_path, score_threshold):
    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = [json.loads(line) for line in f]
    original_len = len(manifest)

    # Filter low score segments
    filtered_manifest = [entry for entry in manifest if entry['score'] >= score_threshold]
    len_filtered = len(filtered_manifest)
    print(f"Filtered {original_len - len_filtered} low score segments ({(original_len - len_filtered) / original_len * 100:.2f}%)")

    # Isolate speakers
    speaker_two_books = ['JOS', 'JDG', 'RUT', '1SA', '2SA', '1KI', '2KI', '1CH', '2CH', 'EZR', 'NEH', 'EST', 'ISA', 'JER', 'LAM', 'EZK', 'DAN', 'ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM', 'HEB', 'JAS', '1PE', '2PE', '1JN', '2JN', '3JN', 'JUD']
    for entry in filtered_manifest:
        book = os.path.basename(entry['audio_filepath']).split('_')[0]
        speaker = 'two' if book in speaker_two_books else 'one'
        entry['speaker_name'] = speaker

    # Separate dev and test sets
    dev_set = [entry for entry in filtered_manifest if os.path.basename(entry['audio_filepath']).split('_')[0] == 'EZR']
    test_set = [entry for entry in filtered_manifest if os.path.basename(entry['audio_filepath']).split('_')[0] == 'COL']
    train_set = [entry for entry in filtered_manifest if entry not in dev_set and entry not in test_set]

    # Write processed manifest for train set
    train_manifest_path = os.path.splitext(manifest_path)[0] + "_train.jsonl"
    with open(train_manifest_path, 'w', encoding='utf-8') as f:
        for entry in train_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Processed train manifest saved to: {train_manifest_path}")

    # Write processed manifest for dev set
    dev_manifest_path = os.path.splitext(manifest_path)[0] + "_dev.jsonl"
    with open(dev_manifest_path, 'w', encoding='utf-8') as f:
        for entry in dev_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Processed dev manifest saved to: {dev_manifest_path}")

    # Write processed manifest for test set
    test_manifest_path = os.path.splitext(manifest_path)[0] + "_test.jsonl"
    with open(test_manifest_path, 'w', encoding='utf-8') as f:
        for entry in test_set:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Processed test manifest saved to: {test_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter low score segments and isolate speakers in manifest')
    parser.add_argument('manifest_path', help='Path to the manifest file')
    parser.add_argument('min_score', type=float, help='Threshold for filtering low score segments')

    args = parser.parse_args()

    process_manifest(args.manifest_path, args.min_score)
