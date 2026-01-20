FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY FE/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"
CMD ["python3", "FE/app.py"]