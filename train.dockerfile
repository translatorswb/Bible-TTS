FROM ghcr.io/idiap/coqui-tts:v0.23.1

# Set working directory
WORKDIR /app

# Copy the script
COPY train_vits.py .

# Set the environment variables
ENV CUDA_VISIBLE_DEVICES=0,1

# Set the entrypoint to run the script
ENTRYPOINT ["python3", "-m", "trainer.distribute", "--script", "train_vits.py"]
