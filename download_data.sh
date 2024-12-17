# Script to download the raw data. Archives containing audio data are in audio_links.txt, and the text data is in text_link.txt.

# Download audio data
wget -i audio_links.txt --content-disposition -P raw_audio/

# Download text data
wget -i text_link.txt --content-disposition -P raw_text/
