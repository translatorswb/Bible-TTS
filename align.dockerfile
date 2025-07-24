FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pydub \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@555ddc9a401b8ddec9cb92f1c74af67f4cba1572

# Copy the script
COPY ctc-alignment.py .

# Set the entrypoint to run the script
ENTRYPOINT ["python", "ctc-alignment.py"]
