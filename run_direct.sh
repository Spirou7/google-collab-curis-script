#!/bin/bash

# Simpler script that uses TensorFlow Docker image directly without building

echo "=================================================="
echo "OPTIMIZER STATE ANALYSIS (Direct Docker)"
echo "=================================================="

# Parse command
COMMAND=${1:-help}

# Function to print usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  discover    - Discover optimizer slot names"
    echo "  experiment  - Run optimizer comparison experiment"
    echo "  analyze     - Analyze state from latest experiment"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 discover"
    echo "  $0 experiment"
    echo "  $0 analyze"
}

# Use official TensorFlow image directly
run_tf_docker() {
    docker run --rm \
        -v "$(pwd):/workspace" \
        -w /workspace \
        -e PYTHONUNBUFFERED=1 \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        tensorflow/tensorflow:2.13.0-gpu \
        python "$@"
}

case $COMMAND in
    discover)
        echo "Discovering optimizer slot names..."
        run_tf_docker fault_injection/scripts/discover_optimizer_slots.py
        ;;
    
    experiment)
        echo "Running optimizer comparison experiment..."
        echo "This will compare SGD (with momentum), SGD_vanilla (no momentum), and Adam"
        run_tf_docker fault_injection/scripts/test_optimizer_mitigation_v4.py \
            --num-experiments 1 \
            --steps-after-injection 50 \
            --optimizers sgd sgd_vanilla adam
        ;;
    
    analyze)
        echo "Analyzing optimizer states from latest experiment..."
        LATEST_DIR=$(ls -td fault_injection/optimizer_comparison_results/experiment_* 2>/dev/null | head -1)
        if [ -z "$LATEST_DIR" ]; then
            echo "No experiment results found!"
            echo "Run '$0 experiment' first"
            exit 1
        fi
        echo "Analyzing: $LATEST_DIR"
        run_tf_docker fault_injection/scripts/analyze_optimizer_states.py \
            "$LATEST_DIR" -o output/state_analysis
        echo ""
        echo "Results saved to output/state_analysis/"
        ;;
    
    help|*)
        usage
        ;;
esac