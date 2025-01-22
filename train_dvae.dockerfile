FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages app

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY train_dvae_xtts.py .
COPY train_dvae.sh .

# Set the entrypoint to run the script
ENTRYPOINT ["bash", "train_dvae.sh"]
