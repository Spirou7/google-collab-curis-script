#!/bin/bash

# Script to run optimizer analysis with Docker

echo "=================================================="
echo "OPTIMIZER STATE ANALYSIS TOOL"
echo "=================================================="

# Check if docker-compose is available, otherwise use docker
if command -v docker-compose &> /dev/null; then
    DOCKER_CMD="docker-compose"
    USE_COMPOSE=true
else
    DOCKER_CMD="docker"
    USE_COMPOSE=false
    echo "Note: docker-compose not found, using docker directly"
    echo ""
fi

# Function to print usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  discover    - Discover optimizer slot names"
    echo "  experiment  - Run optimizer comparison experiment"
    echo "  analyze     - Analyze state from latest experiment"
    echo "  full        - Run full pipeline (discover + experiment + analyze)"
    echo "  help        - Show this help message"
    echo ""
    echo "Options for experiment:"
    echo "  --optimizers  - List of optimizers (default: adam sgd sgd_vanilla)"
    echo "  --experiments - Number of experiments (default: 1)"
    echo "  --steps       - Steps after injection (default: 50)"
    echo ""
    echo "Examples:"
    echo "  $0 discover"
    echo "  $0 experiment --optimizers 'adam sgd sgd_vanilla rmsprop' --experiments 5"
    echo "  $0 analyze"
    echo "  $0 full"
}

# Parse command
COMMAND=${1:-help}
shift

# Default values
OPTIMIZERS="adam sgd sgd_vanilla"
NUM_EXPERIMENTS=1
STEPS_AFTER=50

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --optimizers)
            OPTIMIZERS="$2"
            shift 2
            ;;
        --experiments)
            NUM_EXPERIMENTS="$2"
            shift 2
            ;;
        --steps)
            STEPS_AFTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Helper function to run docker commands
run_docker() {
    if [ "$USE_COMPOSE" = true ]; then
        docker-compose run --rm "$@"
    else
        # Build image if needed
        if ! docker images | grep -q "curis-optimizer"; then
            echo "Building Docker image..."
            docker build -t curis-optimizer:latest .
        fi
        
        # Run with docker directly
        docker run --rm \
            -v "$(pwd):/app" \
            -w /app \
            -e PYTHONUNBUFFERED=1 \
            -e TF_CPP_MIN_LOG_LEVEL=2 \
            curis-optimizer:latest \
            "$@"
    fi
}

# Execute command
case $COMMAND in
    discover)
        echo "Discovering optimizer slot names..."
        run_docker python fault_injection/scripts/discover_optimizer_slots.py
        ;;
    
    experiment)
        echo "Running optimizer comparison experiment..."
        echo "Optimizers: $OPTIMIZERS"
        echo "Experiments: $NUM_EXPERIMENTS"
        echo "Steps after injection: $STEPS_AFTER"
        
        run_docker python fault_injection/scripts/test_optimizer_mitigation_v4.py \
            --num-experiments $NUM_EXPERIMENTS \
            --steps-after-injection $STEPS_AFTER \
            --optimizers $OPTIMIZERS
        ;;
    
    analyze)
        echo "Analyzing optimizer states from latest experiment..."
        if [ "$USE_COMPOSE" = true ]; then
            docker-compose run --rm analyze-states
        else
            # Find latest experiment directory
            LATEST_DIR=$(ls -td fault_injection/optimizer_comparison_results/experiment_* 2>/dev/null | head -1)
            if [ -z "$LATEST_DIR" ]; then
                echo "No experiment results found!"
                exit 1
            fi
            echo "Analyzing: $LATEST_DIR"
            run_docker python fault_injection/scripts/analyze_optimizer_states.py \
                "$LATEST_DIR" -o output/state_analysis
        fi
        echo ""
        echo "Results saved to output/state_analysis/"
        ;;
    
    full)
        echo "Running full analysis pipeline..."
        
        # Step 1: Discover slots
        echo ""
        echo "Step 1/3: Discovering optimizer slots..."
        run_docker python fault_injection/scripts/discover_optimizer_slots.py
        
        # Step 2: Run experiment
        echo ""
        echo "Step 2/3: Running experiment..."
        run_docker python fault_injection/scripts/test_optimizer_mitigation_v4.py \
            --num-experiments $NUM_EXPERIMENTS \
            --steps-after-injection $STEPS_AFTER \
            --optimizers $OPTIMIZERS
        
        # Step 3: Analyze results
        echo ""
        echo "Step 3/3: Analyzing results..."
        if [ "$USE_COMPOSE" = true ]; then
            docker-compose run --rm analyze-states
        else
            LATEST_DIR=$(ls -td fault_injection/optimizer_comparison_results/experiment_* 2>/dev/null | head -1)
            if [ -n "$LATEST_DIR" ]; then
                echo "Analyzing: $LATEST_DIR"
                run_docker python fault_injection/scripts/analyze_optimizer_states.py \
                    "$LATEST_DIR" -o output/state_analysis
            fi
        fi
        
        echo ""
        echo "=================================================="
        echo "Full pipeline complete!"
        echo "Results saved in output/"
        echo "=================================================="
        ;;
    
    help|*)
        usage
        ;;
esac