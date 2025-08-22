#!/bin/bash

# Script that uses TensorFlow Docker image with named volumes

echo "=================================================="
echo "OPTIMIZER STATE ANALYSIS (Docker with Volumes)"
echo "=================================================="

# Parse command
COMMAND=${1:-help}

# Create named volumes if they don't exist
create_volumes() {
    echo "Ensuring Docker volumes exist..."
    docker volume create fault_injection_optimizer 2>/dev/null
    docker volume create fault_injection_results 2>/dev/null
    docker volume create fault_injection_output 2>/dev/null
    docker volume create fault_injection_checkpoints 2>/dev/null
    docker volume create fault_injection_source 2>/dev/null
}

# Copy source code to volume
copy_source_to_volume() {
    echo "Copying source code to Docker volume..."
    # Create a temporary container to copy files
    docker run --rm \
        -v fault_injection_source:/app \
        -v "$(pwd)":/source:ro \
        alpine:latest \
        sh -c "cp -r /source/* /app/ 2>/dev/null || true"
}

# Function to print usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup       - Setup volumes and copy source code"
    echo "  discover    - Discover optimizer slot names"
    echo "  experiment  - Run optimizer comparison experiment"
    echo "  analyze     - Analyze state from latest experiment"
    echo "  extract     - Extract results from Docker volumes to local directory"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # First time setup"
    echo "  $0 discover"
    echo "  $0 experiment"
    echo "  $0 extract    # Get results out"
}

# Use official TensorFlow image with volumes
run_tf_docker() {
    docker run --rm \
        -v fault_injection_source:/app:ro \
        -v fault_injection_optimizer:/app/fault_injection/optimizer_comparison_results \
        -v fault_injection_results:/app/fault_injection/results \
        -v fault_injection_output:/app/output \
        -v fault_injection_checkpoints:/app/checkpoints \
        -w /app \
        -e PYTHONUNBUFFERED=1 \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        tensorflow/tensorflow:2.13.0-gpu \
        python "$@"
}

# Extract results from volumes
extract_results() {
    echo "Extracting results from Docker volumes..."
    
    # Create local results directory
    mkdir -p ./extracted_results
    
    # Use a container to copy files from volumes
    docker run --rm \
        -v fault_injection_optimizer:/optimizer \
        -v fault_injection_output:/output \
        -v "$(pwd)/extracted_results":/extract \
        alpine:latest \
        sh -c "
            echo 'Copying optimizer comparison results...'
            cp -r /optimizer/* /extract/ 2>/dev/null || echo 'No optimizer results found'
            echo 'Copying analysis output...'
            cp -r /output/* /extract/ 2>/dev/null || echo 'No analysis output found'
            echo 'Done!'
        "
    
    echo ""
    echo "Results extracted to ./extracted_results/"
    ls -la ./extracted_results/
}

case $COMMAND in
    setup)
        echo "Setting up Docker volumes and copying source code..."
        create_volumes
        copy_source_to_volume
        echo "Setup complete!"
        ;;
    
    discover)
        echo "Discovering optimizer slot names..."
        create_volumes
        copy_source_to_volume
        run_tf_docker fault_injection/scripts/discover_optimizer_slots.py
        ;;
    
    experiment)
        echo "Running optimizer comparison experiment..."
        echo "This will compare SGD (with momentum), SGD_vanilla (no momentum), and Adam"
        create_volumes
        copy_source_to_volume
        run_tf_docker fault_injection/scripts/test_optimizer_mitigation_v4.py \
            --num-experiments 1 \
            --steps-after-injection 50 \
            --optimizers sgd sgd_vanilla adam
        ;;
    
    analyze)
        echo "Analyzing optimizer states from latest experiment..."
        create_volumes
        # Find latest experiment in volume
        docker run --rm \
            -v fault_injection_optimizer:/optimizer \
            alpine:latest \
            sh -c "ls -td /optimizer/experiment_* 2>/dev/null | head -1" > /tmp/latest_exp.txt
        
        LATEST_DIR=$(cat /tmp/latest_exp.txt | sed 's|/optimizer/|fault_injection/optimizer_comparison_results/|')
        rm /tmp/latest_exp.txt
        
        if [ -z "$LATEST_DIR" ]; then
            echo "No experiment results found in Docker volume!"
            echo "Run '$0 experiment' first"
            exit 1
        fi
        
        echo "Analyzing: $LATEST_DIR"
        run_tf_docker fault_injection/scripts/analyze_optimizer_states.py \
            "$LATEST_DIR" -o output/state_analysis
        echo ""
        echo "Results saved to output/state_analysis/ in Docker volume"
        echo "Run '$0 extract' to get results"
        ;;
    
    extract)
        extract_results
        ;;
    
    help|*)
        usage
        ;;
esac