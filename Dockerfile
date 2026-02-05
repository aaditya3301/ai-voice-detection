FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 7860

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
