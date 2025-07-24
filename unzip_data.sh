#!/bin/bash

# Check if the directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR=$1

# Unzip archives
for f in $DIR/raw_audio/*.zip; do
    unzip $f -d "$DIR/chapters/" -x metadata.xml
done

unzip "$DIR/raw_text/*.zip" -d "$DIR/text_by_book/"
