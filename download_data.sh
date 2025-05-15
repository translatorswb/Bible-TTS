#!/bin/bash

# Check if the directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory containing the txt files
DIR=$1

# Download audio data
wget -i "$DIR/audio_links.txt" --content-disposition -P "$DIR/raw_audio/"

# Download text data
wget -i "$DIR/text_link.txt" --content-disposition -P "$DIR/raw_text/"
