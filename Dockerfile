# Use official TensorFlow image as base
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install \
    numpy==1.24.3 \
    matplotlib==3.7.1 \
    scikit-learn \
    pandas \
    Pillow

# Copy the project files
COPY . /app/

# Create results directory
RUN mkdir -p /app/results

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (can be overridden)
CMD ["python", "fault_injection/scripts/test_optimizer_mitigation_v3.py", "--help"]
