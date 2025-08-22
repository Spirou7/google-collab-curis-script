#!/bin/bash

# Script to run optimizer analysis with Docker

echo "=================================================="
echo "OPTIMIZER STATE ANALYSIS TOOL"
echo "=================================================="

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

# Execute command
case $COMMAND in
    discover)
        echo "Discovering optimizer slot names..."
        docker-compose run --rm discover-slots
        ;;
    
    experiment)
        echo "Running optimizer comparison experiment..."
        echo "Optimizers: $OPTIMIZERS"
        echo "Experiments: $NUM_EXPERIMENTS"
        echo "Steps after injection: $STEPS_AFTER"
        
        docker-compose run --rm optimizer-experiment \
            python fault_injection/scripts/test_optimizer_mitigation_v4.py \
            --num-experiments $NUM_EXPERIMENTS \
            --steps-after-injection $STEPS_AFTER \
            --optimizers $OPTIMIZERS
        ;;
    
    analyze)
        echo "Analyzing optimizer states from latest experiment..."
        docker-compose run --rm analyze-states
        echo ""
        echo "Results saved to output/state_analysis/"
        echo "To view results locally, copy them from Docker volume:"
        echo "  docker cp \$(docker-compose ps -q analyze-states):/app/output/state_analysis ./state_analysis"
        ;;
    
    full)
        echo "Running full analysis pipeline..."
        
        # Step 1: Discover slots
        echo ""
        echo "Step 1/3: Discovering optimizer slots..."
        docker-compose run --rm discover-slots
        
        # Step 2: Run experiment
        echo ""
        echo "Step 2/3: Running experiment..."
        docker-compose run --rm optimizer-experiment \
            python fault_injection/scripts/test_optimizer_mitigation_v4.py \
            --num-experiments $NUM_EXPERIMENTS \
            --steps-after-injection $STEPS_AFTER \
            --optimizers $OPTIMIZERS
        
        # Step 3: Analyze results
        echo ""
        echo "Step 3/3: Analyzing results..."
        docker-compose run --rm analyze-states
        
        echo ""
        echo "=================================================="
        echo "Full pipeline complete!"
        echo "Results saved in Docker volumes"
        echo "=================================================="
        ;;
    
    help|*)
        usage
        ;;
esac