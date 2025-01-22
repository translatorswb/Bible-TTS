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

# Install deepspeed
RUN pip install --no-cache-dir deepspeed==0.10.3

# Install pymcd
RUN pip install pymcd

# Copy the scripts
COPY synthesize.py .
COPY synthesize_csv.py .
COPY evaluate_run.py .

# Set the entrypoint to run the script
ENTRYPOINT ["python"]
