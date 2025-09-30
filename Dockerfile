# -----------------------------
# Base Image
# -----------------------------
FROM python:3.13-slim

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy project files
# -----------------------------
COPY . /app

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless

# -----------------------------
# Environment variables
# -----------------------------

ENV POPPLER_PATH=/usr/bin

# -----------------------------
# Run both bots in parallel
# -----------------------------
# Use a shell script to manage both processes
COPY run_bots.sh /app/run_bots.sh
RUN chmod +x /app/run_bots.sh

CMD ["/app/run_bots.sh"]
