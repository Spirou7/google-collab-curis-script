#!/bin/bash

# Docker run script for fault injection experiments
# This script uses Docker named volumes to avoid host permission issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Configuration
IMAGE_NAME="fault-injection-experiment:latest"
CONTAINER_NAME="fault_injection_runner"

# Named volumes (Docker-managed, no host permissions needed)
VOLUME_RESULTS="fault_injection_results"
VOLUME_OPTIMIZER="fault_injection_optimizer"
VOLUME_OUTPUT="fault_injection_output"
VOLUME_CHECKPOINTS="fault_injection_checkpoints"

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running or not installed"
        exit 1
    fi
}

# Function to create volumes if they don't exist
create_volumes() {
    print_message "Ensuring Docker volumes exist..."
    docker volume create $VOLUME_RESULTS >/dev/null 2>&1 || true
    docker volume create $VOLUME_OPTIMIZER >/dev/null 2>&1 || true
    docker volume create $VOLUME_OUTPUT >/dev/null 2>&1 || true
    docker volume create $VOLUME_CHECKPOINTS >/dev/null 2>&1 || true
    print_message "Volumes ready: $VOLUME_RESULTS, $VOLUME_OPTIMIZER, $VOLUME_OUTPUT, $VOLUME_CHECKPOINTS"
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
    create_volumes
    print_message "Starting interactive container with named volumes..."
    print_info "Files will be saved in Docker volumes (no host mounting)"
    print_info "Use 'exit' to leave the container"
    print_info "Use './docker_run.sh copy-results' to extract files after"
    
    docker run -it --rm \
        --name $CONTAINER_NAME \
        -v $VOLUME_RESULTS:/app/fault_injection/results \
        -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results \
        -v $VOLUME_OUTPUT:/app/output \
        -v $VOLUME_CHECKPOINTS:/app/checkpoints \
        -e HOME=/app \
        -w /app \
        $IMAGE_NAME \
        /bin/bash
}

# Function to run experiment
run_experiment() {
    create_volumes
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment: $script_path"
    print_message "Arguments: $args"
    print_info "Results will be saved in Docker volumes"
    
    docker run -it --rm \
        --name $CONTAINER_NAME \
        -v $VOLUME_RESULTS:/app/fault_injection/results \
        -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results \
        -v $VOLUME_OUTPUT:/app/output \
        -v $VOLUME_CHECKPOINTS:/app/checkpoints \
        -e HOME=/app \
        -w /app \
        $IMAGE_NAME \
        python $script_path $args
    
    print_message "Experiment completed!"
    print_info "To extract results, run: ./docker_run.sh copy-results"
}

# Function to run with GPU support
run_with_gpu() {
    create_volumes
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment with GPU support: $script_path"
    
    docker run -it --rm \
        --gpus all \
        --name $CONTAINER_NAME \
        -v $VOLUME_RESULTS:/app/fault_injection/results \
        -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results \
        -v $VOLUME_OUTPUT:/app/output \
        -v $VOLUME_CHECKPOINTS:/app/checkpoints \
        -e HOME=/app \
        -w /app \
        $IMAGE_NAME \
        python $script_path $args
    
    print_info "To extract results, run: ./docker_run.sh copy-results"
}

# Function to run in background (detached)
run_background() {
    create_volumes
    local script_path=$1
    shift
    local args=$@
    
    print_message "Running experiment in background: $script_path"
    
    container_id=$(docker run -d \
        --name $CONTAINER_NAME \
        -v $VOLUME_RESULTS:/app/fault_injection/results \
        -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results \
        -v $VOLUME_OUTPUT:/app/output \
        -v $VOLUME_CHECKPOINTS:/app/checkpoints \
        -e HOME=/app \
        -w /app \
        $IMAGE_NAME \
        python $script_path $args)
    
    print_message "Container started with ID: $container_id"
    print_message "Use 'docker logs -f $CONTAINER_NAME' to follow logs"
    print_message "Use 'docker stop $CONTAINER_NAME' to stop"
    print_info "To extract results, run: ./docker_run.sh copy-results"
}

# Function to copy all results from volumes to current directory
copy_results() {
    print_message "Copying results from Docker volumes to current directory..."
    
    # Create local directories
    mkdir -p ./extracted_results/fault_injection/results
    mkdir -p ./extracted_results/fault_injection/optimizer_comparison_results
    mkdir -p ./extracted_results/output
    mkdir -p ./extracted_results/checkpoints
    
    print_info "Extracting results using tar stream (no mounting required)..."
    
    # Create a tar archive from volumes and extract locally
    docker run --rm \
        -v $VOLUME_RESULTS:/source/results:ro \
        -v $VOLUME_OPTIMIZER:/source/optimizer:ro \
        -v $VOLUME_OUTPUT:/source/output:ro \
        -v $VOLUME_CHECKPOINTS:/source/checkpoints:ro \
        alpine tar czf - -C /source . 2>/dev/null | tar xzf - -C ./extracted_results/ 2>/dev/null || {
            print_warning "Standard extraction failed, trying alternative method..."
            
            # Alternative: Create container, copy files, then extract
            print_info "Creating temporary container for extraction..."
            docker run -d --name temp_extractor \
                -v $VOLUME_RESULTS:/app/fault_injection/results:ro \
                -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results:ro \
                -v $VOLUME_OUTPUT:/app/output:ro \
                -v $VOLUME_CHECKPOINTS:/app/checkpoints:ro \
                alpine sleep 300
            
            # Copy from container to local
            docker cp temp_extractor:/app/fault_injection/results/. ./extracted_results/fault_injection/results/ 2>/dev/null || true
            docker cp temp_extractor:/app/fault_injection/optimizer_comparison_results/. ./extracted_results/fault_injection/optimizer_comparison_results/ 2>/dev/null || true
            docker cp temp_extractor:/app/output/. ./extracted_results/output/ 2>/dev/null || true
            docker cp temp_extractor:/app/checkpoints/. ./extracted_results/checkpoints/ 2>/dev/null || true
            
            # Cleanup
            docker rm -f temp_extractor >/dev/null 2>&1
    }
    
    print_message "Results extracted to ./extracted_results/"
    print_info "Listing extracted files:"
    find ./extracted_results -type f -name "*.json" -o -name "*.csv" -o -name "*.pkl" 2>/dev/null | head -20 || echo "No files found yet"
}

# Function to copy a specific file from container
copy_single() {
    local file_path=$1
    if [ -z "$file_path" ]; then
        print_error "Please specify a file path to copy"
        print_info "Example: ./docker_run.sh copy-single /app/fault_injection/results/experiment.json"
        exit 1
    fi
    
    # Try to copy from running container first
    if docker ps -q -f name=$CONTAINER_NAME >/dev/null 2>&1; then
        print_message "Copying $file_path from running container..."
        docker cp $CONTAINER_NAME:$file_path ./$(basename $file_path) && \
            print_message "File copied to ./$(basename $file_path)" || \
            print_error "Failed to copy file"
    else
        print_warning "No running container found"
        print_info "Starting temporary container to extract file..."
        
        # Extract from volume using temporary container
        docker run --rm \
            -v $VOLUME_RESULTS:/app/fault_injection/results:ro \
            -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results:ro \
            -v $VOLUME_OUTPUT:/app/output:ro \
            -v $VOLUME_CHECKPOINTS:/app/checkpoints:ro \
            -v "$(pwd)":/dest \
            alpine cp $file_path /dest/$(basename $file_path) && \
            print_message "File copied to ./$(basename $file_path)" || \
            print_error "Failed to copy file"
    fi
}

# Function to list files in volumes
list_results() {
    print_message "Listing files in Docker volumes..."
    
    docker run --rm \
        -v $VOLUME_RESULTS:/results:ro \
        -v $VOLUME_OPTIMIZER:/optimizer:ro \
        -v $VOLUME_OUTPUT:/output:ro \
        -v $VOLUME_CHECKPOINTS:/checkpoints:ro \
        alpine sh -c "
            echo '=== Results Volume ==='
            find /results -type f 2>/dev/null | head -20 || echo 'No files found'
            echo ''
            echo '=== Optimizer Results Volume ==='
            find /optimizer -type f 2>/dev/null | head -20 || echo 'No files found'
            echo ''
            echo '=== Output Volume ==='
            find /output -type f 2>/dev/null | head -20 || echo 'No files found'
            echo ''
            echo '=== Checkpoints Volume ==='
            find /checkpoints -type f 2>/dev/null | head -10 || echo 'No files found'
        "
}

# Function to clean volumes (with confirmation)
clean_volumes() {
    print_warning "This will DELETE all data in Docker volumes!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_message "Removing Docker volumes..."
        docker volume rm $VOLUME_RESULTS $VOLUME_OPTIMIZER $VOLUME_OUTPUT $VOLUME_CHECKPOINTS 2>/dev/null || true
        print_message "Volumes removed"
    else
        print_message "Cancelled"
    fi
}

# Function to backup all volumes to a tar archive (no mounting)
backup_volumes() {
    local backup_name="fault_injection_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    print_message "Creating backup: $backup_name"
    
    # Stream tar directly to local file without mounting
    docker run --rm \
        -v $VOLUME_RESULTS:/backup/results:ro \
        -v $VOLUME_OPTIMIZER:/backup/optimizer:ro \
        -v $VOLUME_OUTPUT:/backup/output:ro \
        -v $VOLUME_CHECKPOINTS:/backup/checkpoints:ro \
        alpine tar czf - -C /backup . > $backup_name && \
        print_message "Backup created: ./$backup_name" || \
        print_error "Backup failed"
}

# Function to extract results without ANY mounting (safest method)
extract_safe() {
    print_message "Extracting results (completely mount-free method)..."
    
    # Create directories
    mkdir -p ./extracted_results
    
    # Create temporary container with volumes attached
    print_info "Creating temporary extraction container..."
    docker create --name temp_extract \
        -v $VOLUME_RESULTS:/data/results:ro \
        -v $VOLUME_OPTIMIZER:/data/optimizer:ro \
        -v $VOLUME_OUTPUT:/data/output:ro \
        -v $VOLUME_CHECKPOINTS:/data/checkpoints:ro \
        alpine sh >/dev/null 2>&1
    
    # Stream tar archive from container to local file
    print_info "Streaming data from Docker volumes..."
    docker run --rm \
        -v $VOLUME_RESULTS:/data/results:ro \
        -v $VOLUME_OPTIMIZER:/data/optimizer:ro \
        -v $VOLUME_OUTPUT:/data/output:ro \
        -v $VOLUME_CHECKPOINTS:/data/checkpoints:ro \
        alpine tar czf - -C /data . | tar xzf - -C ./extracted_results/
    
    # Cleanup
    docker rm -f temp_extract >/dev/null 2>&1 || true
    
    print_message "Results extracted to ./extracted_results/"
    print_info "Contents:"
    ls -la ./extracted_results/ 2>/dev/null || echo "Check ./extracted_results/ directory"
}

# Function to show volume info
volume_info() {
    print_message "Docker Volume Information:"
    echo ""
    for vol in $VOLUME_RESULTS $VOLUME_OPTIMIZER $VOLUME_OUTPUT $VOLUME_CHECKPOINTS; do
        echo "Volume: $vol"
        docker volume inspect $vol 2>/dev/null | grep -E '"CreatedAt"|"Mountpoint"' || echo "  Not created yet"
        echo ""
    done
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
    
    === File Management (NEW) ===
    copy-results            Copy ALL results from volumes to ./extracted_results/
    extract-safe            Extract results (safest, no mount method)
    copy-single FILE        Copy specific file from container/volume
    list-results            List files in Docker volumes
    backup                  Create tar.gz backup of all volumes
    
    === Volume Management ===
    volume-info             Show information about Docker volumes
    clean-volumes           Delete all Docker volumes (WARNING: deletes data)
    
    help                    Show this help message

Examples:
    # Build the image
    $0 build
    
    # Run interactive shell
    $0 interactive
    
    # Run optimizer mitigation experiment
    $0 optimizer --baseline adam --test-optimizers sgd rmsprop --num-experiments 10
    
    # Extract all results after experiment
    $0 copy-results
    
    # Copy specific file
    $0 copy-single /app/fault_injection/results/experiment_001.json
    
    # List what's in the volumes
    $0 list-results
    
    # Backup all data
    $0 backup
    
    # Clean up volumes (careful!)
    $0 clean-volumes

Notes:
    - Files are saved in Docker named volumes (no host mounting required)
    - Volumes persist data between container runs
    - Use 'copy-results' to extract files to your current directory
    - No permission issues since Docker manages the volumes internally

EOF
}

# Main script logic
check_docker

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
    copy-results)
        copy_results
        ;;
    extract-safe)
        extract_safe
        ;;
    copy-single)
        shift
        copy_single $@
        ;;
    list-results)
        list_results
        ;;
    clean-volumes)
        clean_volumes
        ;;
    backup)
        backup_volumes
        ;;
    volume-info)
        volume_info
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -z "$1" ]; then
            show_help
        else
            print_error "Unknown command: $1"
            show_help
            exit 1
        fi
        ;;
esac