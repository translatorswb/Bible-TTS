from bs4 import BeautifulSoup
import os
import argparse
from text_utils import normalize_text


NON_SPEECH_ELEM_STYLES = ['b', 'r', 'iex', 'ms', 'mr', 'cl']


def process_usx_file(usx_path, out_text_path, chapter_utterance):
    usx_content = open(usx_path, 'r')
    soup = BeautifulSoup(usx_content, features="lxml")

    # Remove <note> elements before processing
    for note in soup.find_all('note'):
        note.decompose()

    title = soup.find("para", {"style": "h"}).text
    code = soup.find("book")['code']

    chapters_original = {}
    chapters_segmented = {}
    chapter_text = ""

    for elem in soup.usx.children:
        if elem.name == 'chapter' and 'sid' in elem.attrs:
            chapter_text = ""
            chapter_no = elem['number']
            chapter_id = code + "_" + chapter_no.zfill(3)

            chapter_text = title + " " + chapter_utterance + " " + chapter_no + "\n"
        elif chapter_text and elem.name == 'para':
            if elem['style'] in ["s1", "ms1"]:
                chapter_text += elem.text + "\n"
            elif elem['style'] not in NON_SPEECH_ELEM_STYLES:
                for v in elem.children:
                    if not v.name and not v.isspace():
                        chapter_text += v.strip() + " "
                    elif v.name == "verse":
                        chapter_text += v.text
                        if 'eid' in v.attrs:
                            chapter_text += '\n'
                        else:
                            chapter_text += ' '
                    elif v.name == "char":
                        chapter_text += v.text + " "
        elif chapter_text and elem.name == 'table':
            for row in elem.find_all('row'):
                for cell in row.find_all('cell'):
                    for v in cell.children:
                        if not v.name and not v.isspace():
                            chapter_text += v.strip() + " "
                        elif v.name == "verse":
                            chapter_text += v.text
                            if 'eid' in v.attrs:
                                chapter_text += '\n'
                            else:
                                chapter_text += ' '
                        elif v.name == "char":
                            chapter_text += v.text
        elif elem.name == 'chapter' and 'eid' in elem.attrs:
            chapters_original[chapter_id] = chapter_text
            chapters_segmented[chapter_id] = normalize_text(chapter_text, remove_punc=False)

    # There is no Daniel 14, it has only 12 chapters
    if "DAN_014" in chapters_original:
        print("Removing Daniel 14")
        del chapters_original["DAN_014"]
        del chapters_segmented["DAN_014"]

    book_id = os.path.basename(usx_path).split(".")[0]
    original_out_dir = os.path.join(out_text_path, "original")
    segmented_out_dir = os.path.join(out_text_path, "processed")

    if not os.path.exists(original_out_dir):
        os.makedirs(original_out_dir)
    if not os.path.exists(segmented_out_dir):
        os.makedirs(segmented_out_dir)

    for chap_id in chapters_segmented:
        out_original_path = os.path.join(original_out_dir, chap_id + ".txt")
        with open(out_original_path, 'w') as f:
            f.write(chapters_original[chap_id])

        out_segmented_path = os.path.join(segmented_out_dir, chap_id + ".txt")
        with open(out_segmented_path, 'w') as f:
            f.write(chapters_segmented[chap_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract chapters from a USX file and save them into separate text files.")
    parser.add_argument("usx_file", help="Path to the input USX file.")
    parser.add_argument("output_dir", help="Path to the output directory.")
    parser.add_argument("chapter_utterance", help="Chapter utterance to prepend to each chapter.")

    args = parser.parse_args()

    process_usx_file(args.usx_file, args.output_dir, args.chapter_utterance)
