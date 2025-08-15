#!/bin/bash

# Singularity run script for HPC clusters
# Alternative to Docker for systems where Docker isn't available

set -e

# Configuration
IMAGE_NAME="fault-injection.sif"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Singularity is available
if ! command -v singularity &> /dev/null; then
    print_error "Singularity is not installed"
    exit 1
fi

# Function to build Singularity image from Dockerfile
build_from_docker() {
    print_message "Building Singularity image from Dockerfile..."
    
    # First build Docker image
    if command -v docker &> /dev/null; then
        print_message "Building Docker image first..."
        docker build -t fault-injection-experiment:latest . || {
            print_error "Failed to build Docker image"
            exit 1
        }
        
        print_message "Converting to Singularity format..."
        singularity build --force $IMAGE_NAME docker-daemon://fault-injection-experiment:latest
    else
        print_warning "Docker not available. Building directly from Dockerfile..."
        # Create a Singularity definition file from Dockerfile
        create_singularity_def
        singularity build --force $IMAGE_NAME fault-injection.def
    fi
    
    print_message "Singularity image built: $IMAGE_NAME"
}

# Function to create Singularity definition file
create_singularity_def() {
    cat > fault-injection.def << 'EOF'
Bootstrap: docker
From: python:3.12-slim

%files
    requirements.txt /app/requirements.txt
    . /app

%post
    # Install system dependencies
    apt-get update && apt-get install -y \
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
    
    # Install Python dependencies
    cd /app
    pip install --no-cache-dir -r requirements.txt
    
    # Create necessary directories
    mkdir -p /app/fault_injection/results
    mkdir -p /app/fault_injection/optimizer_comparison_results
    mkdir -p /app/output
    chmod -R 777 /app

%environment
    export PYTHONPATH=/app:$PYTHONPATH
    export TF_CPP_MIN_LOG_LEVEL=1
    export PYTHONUNBUFFERED=1

%runscript
    cd /app
    exec python "$@"
EOF
    print_message "Created Singularity definition file"
}

# Function to run interactive shell
run_interactive() {
    print_message "Starting interactive Singularity shell..."
    singularity shell \
        --bind "$SCRIPT_DIR:/app" \
        --pwd /app \
        $IMAGE_NAME
}

# Function to run experiment
run_experiment() {
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment: $script_path"
    print_message "Arguments: $args"
    
    singularity exec \
        --bind "$SCRIPT_DIR:/app" \
        --pwd /app \
        $IMAGE_NAME \
        python $script_path $args
}

# Function to run with GPU
run_gpu() {
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running with GPU support: $script_path"
    
    singularity exec \
        --nv \
        --bind "$SCRIPT_DIR:/app" \
        --pwd /app \
        $IMAGE_NAME \
        python $script_path $args
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build                   Build Singularity image
    interactive             Run interactive shell
    run SCRIPT [ARGS]       Run a specific script
    gpu SCRIPT [ARGS]       Run with GPU support
    optimizer [ARGS]        Run optimizer mitigation experiment
    help                    Show this help message

Examples:
    # Build the image
    $0 build
    
    # Run interactive shell
    $0 interactive
    
    # Run optimizer experiment
    $0 optimizer --baseline adam --test-optimizers sgd rmsprop --num-experiments 10
    
    # Run with GPU
    $0 gpu fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 100

Note: This script uses Singularity instead of Docker, which is more suitable
      for HPC clusters and shared computing environments.
EOF
}

# Check if image exists
check_image() {
    if [ ! -f "$IMAGE_NAME" ]; then
        print_warning "Singularity image not found: $IMAGE_NAME"
        print_message "Run '$0 build' to create it"
        exit 1
    fi
}

# Main script logic
case "$1" in
    build)
        build_from_docker
        ;;
    interactive)
        check_image
        run_interactive
        ;;
    run)
        check_image
        shift
        run_experiment $@
        ;;
    gpu)
        check_image
        shift
        run_gpu $@
        ;;
    optimizer)
        check_image
        shift
        run_experiment "fault_injection/scripts/test_optimizer_mitigation_v3.py" $@
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac