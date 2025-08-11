# Use TensorFlow base image with GPU support (change to tensorflow:2.15.0 for CPU-only)
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-tk \
    git \
    wget \
    vim \
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

# Create a non-root user for security (optional but recommended)
RUN useradd -m -s /bin/bash researcher && \
    chown -R researcher:researcher /app

# Switch to non-root user
USER researcher

# Default command - shows help
CMD ["python", "fault_injection/scripts/test_optimizer_mitigation_v3.py", "--help"]