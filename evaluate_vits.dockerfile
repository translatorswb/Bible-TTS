FROM ghcr.io/idiap/coqui-tts:v0.23.1

# Set working directory
WORKDIR /app

# Install pymcd
RUN pip install pymcd

# Copy the scripts
COPY synthesize.py .
COPY synthesize_csv.py .
COPY evaluate_run.py .

# Set the entrypoint to run the script
ENTRYPOINT ["python3"]
