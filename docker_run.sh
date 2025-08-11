#!/bin/bash

# Docker run script for fault injection experiments
# This script provides different modes for running the experiments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="fault-injection-experiment:latest"
CONTAINER_NAME="fault_injection_runner"
RESULTS_DIR="./fault_injection/results"
OPTIMIZER_RESULTS_DIR="./fault_injection/optimizer_comparison_results"
OUTPUT_DIR="./output"

# Create necessary directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$OPTIMIZER_RESULTS_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to build the Docker image
build_image() {
    print_message "Building Docker image..."
    docker build -t $IMAGE_NAME . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    print_message "Docker image built successfully!"
}

# Function to run interactive shell
run_interactive() {
    print_message "Starting interactive container..."
    docker run -it --rm \
        --name $CONTAINER_NAME \
        -v "$(pwd)/$RESULTS_DIR:/app/fault_injection/results" \
        -v "$(pwd)/$OPTIMIZER_RESULTS_DIR:/app/fault_injection/optimizer_comparison_results" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        $IMAGE_NAME \
        /bin/bash
}

# Function to run experiment
run_experiment() {
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment: $script_path"
    print_message "Arguments: $args"
    
    docker run -it --rm \
        --name $CONTAINER_NAME \
        -v "$(pwd)/$RESULTS_DIR:/app/fault_injection/results" \
        -v "$(pwd)/$OPTIMIZER_RESULTS_DIR:/app/fault_injection/optimizer_comparison_results" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        $IMAGE_NAME \
        python $script_path $args
}

# Function to run with GPU support
run_with_gpu() {
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment with GPU support: $script_path"
    
    docker run -it --rm \
        --gpus all \
        --name $CONTAINER_NAME \
        -v "$(pwd)/$RESULTS_DIR:/app/fault_injection/results" \
        -v "$(pwd)/$OPTIMIZER_RESULTS_DIR:/app/fault_injection/optimizer_comparison_results" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        $IMAGE_NAME \
        python $script_path $args
}

# Function to run in background (detached)
run_background() {
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment in background: $script_path"
    
    container_id=$(docker run -d \
        --name $CONTAINER_NAME \
        -v "$(pwd)/$RESULTS_DIR:/app/fault_injection/results" \
        -v "$(pwd)/$OPTIMIZER_RESULTS_DIR:/app/fault_injection/optimizer_comparison_results" \
        -v "$(pwd)/$OUTPUT_DIR:/app/output" \
        $IMAGE_NAME \
        python $script_path $args)
    
    print_message "Container started with ID: $container_id"
    print_message "Use 'docker logs -f $CONTAINER_NAME' to follow logs"
    print_message "Use 'docker stop $CONTAINER_NAME' to stop"
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build                   Build the Docker image
    interactive             Run interactive bash shell in container
    run SCRIPT [ARGS]       Run a specific script with arguments
    gpu SCRIPT [ARGS]       Run with GPU support
    background SCRIPT [ARGS] Run in background (detached)
    optimizer [ARGS]        Run optimizer mitigation experiment
    help                    Show this help message

Examples:
    # Build the image
    $0 build
    
    # Run interactive shell
    $0 interactive
    
    # Run optimizer mitigation experiment
    $0 optimizer --baseline adam --test-optimizers sgd rmsprop --num-experiments 10
    
    # Run with GPU
    $0 gpu fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 100
    
    # Run in background
    $0 background fault_injection/scripts/random_injection.py

EOF
}

# Main script logic
case "$1" in
    build)
        build_image
        ;;
    interactive)
        run_interactive
        ;;
    run)
        shift
        run_experiment $@
        ;;
    gpu)
        shift
        run_with_gpu $@
        ;;
    background)
        shift
        run_background $@
        ;;
    optimizer)
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