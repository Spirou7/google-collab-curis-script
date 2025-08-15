#!/bin/bash

# Wrapper script to run experiments with periodic checkpoint extractions
# This ensures you don't lose progress if the script is interrupted

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_message() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[CHECKPOINT]${NC} $1"; }

# Start experiment in background
print_message "Starting experiment in background..."
./shell_scripts/docker_run.sh background optimizer "$@"

# Wait a bit for it to start
sleep 5

# Monitor and periodically save
while docker ps | grep -q fault_injection_runner; do
    print_warning "Experiment running... (Ctrl+C to stop monitoring, experiment continues)"
    
    # Every 5 minutes, show what's saved
    sleep 300
    
    print_message "Auto-checkpoint: Checking saved files..."
    ./shell_scripts/docker_run.sh list-results | head -20
    
    # Optional: Auto-extract every 30 minutes
    # ./shell_scripts/docker_run.sh extract-safe
done

print_message "Experiment completed or stopped!"
print_message "Extracting final results..."
./shell_scripts/docker_run.sh extract-safe

print_message "Results saved to ./extracted_results/"