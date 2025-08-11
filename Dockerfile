# Use Python 3.12 base image (matching your environment)
# TensorFlow 2.19.0 will be installed via requirements.txt
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-tk \
    python3-dev \
    gcc \
    g++ \
    git \
    wget \
    vim \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Create directories for results and data persistence
RUN mkdir -p /app/fault_injection/results \
    && mkdir -p /app/fault_injection/optimizer_comparison_results \
    && mkdir -p /app/data \
    && mkdir -p /app/output \
    && mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV PYTHONUNBUFFERED=1

# Make the app directory writable by all users (for bind mount compatibility)
RUN chmod -R 777 /app

# Don't switch to a specific user - let docker run command handle it
# This allows --user flag to work properly

# Default command - shows help
CMD ["python", "fault_injection/scripts/test_optimizer_mitigation_v3.py", "--help"]