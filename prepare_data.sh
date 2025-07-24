#!/bin/bash

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <USX directory> <language>"
    exit 1
fi

DIR=$1
LANG=$2

# Process text data
for f in $DIR/*.usx; do
    python process_chapters_text.py $f $DIR/../../../text_by_chapter/ $LANG
done
