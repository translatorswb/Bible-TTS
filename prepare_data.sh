# Unzip archives and prepare data for training

for f in raw_audio/*.zip; do
    unzip $f -d chapters/
done
