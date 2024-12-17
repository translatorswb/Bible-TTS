# Unzip archives and prepare data for training

for f in raw_audio/*.zip; do
    unzip $f -d chapters/
done

unzip raw_text/*.zip -d text_by_book/

# Process text data
for f in text_by_book/release/USX_2/*.usx; do
    python process_chapters_text.py $f text_by_chapter/ Sura
done
