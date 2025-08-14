#!/bin/bash

# Script to find and recover results from old Docker runs

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_message() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

print_message "Searching for your experiment results..."

# 1. Check for stopped containers that might have results
print_message "Checking for stopped containers..."
stopped_containers=$(docker ps -a --filter "ancestor=fault-injection-experiment:latest" --format "{{.ID}} {{.Names}} {{.Status}}" 2>/dev/null || true)

if [ ! -z "$stopped_containers" ]; then
    print_info "Found stopped containers:"
    echo "$stopped_containers"
    
    # Try to copy from each stopped container
    docker ps -a --filter "ancestor=fault-injection-experiment:latest" -q | while read container_id; do
        if [ ! -z "$container_id" ]; then
            print_info "Checking container: $container_id"
            
            # Try to copy results directory
            mkdir -p ./recovered_results/$container_id
            docker cp $container_id:/app/fault_injection/results ./recovered_results/$container_id/ 2>/dev/null && \
                print_message "Recovered results from container $container_id" || \
                print_warning "No results in container $container_id"
            
            docker cp $container_id:/app/fault_injection/optimizer_comparison_results ./recovered_results/$container_id/ 2>/dev/null && \
                print_message "Recovered optimizer results from container $container_id" || true
        fi
    done
else
    print_warning "No stopped containers found"
fi

# 2. Check Docker volumes (including old unnamed ones)
print_message "Checking all Docker volumes..."
docker volume ls --format "{{.Name}}" | while read vol; do
    # Check volume size
    size=$(docker run --rm -v $vol:/data:ro alpine du -sh /data 2>/dev/null | cut -f1)
    if [ "$size" != "4.0K" ] && [ ! -z "$size" ]; then
        print_info "Found non-empty volume: $vol (size: $size)"
        
        # List contents
        echo "  Contents:"
        docker run --rm -v $vol:/data:ro alpine find /data -type f | head -5
    fi
done

# 3. Check if results were saved in container's filesystem layers
print_message "Checking container filesystem layers..."
image_id=$(docker images -q fault-injection-experiment:latest | head -1)
if [ ! -z "$image_id" ]; then
    # Create a temporary container from the image
    temp_container=$(docker create fault-injection-experiment:latest)
    
    # Check for results
    mkdir -p ./recovered_results/image_check
    docker cp $temp_container:/app/fault_injection/results ./recovered_results/image_check/ 2>/dev/null && \
        print_info "Found results in image layers" || \
        print_warning "No results in image layers"
    
    # Cleanup
    docker rm $temp_container >/dev/null 2>&1
fi

# 4. Look for any anonymous volumes
print_message "Checking for anonymous volumes..."
docker volume ls -f dangling=true --format "{{.Name}}" | while read vol; do
    size=$(docker run --rm -v $vol:/data:ro alpine du -sh /data 2>/dev/null | cut -f1)
    if [ "$size" != "4.0K" ] && [ ! -z "$size" ]; then
        print_info "Found non-empty anonymous volume: $vol (size: $size)"
        
        # Try to extract it
        mkdir -p ./recovered_results/anonymous_volumes/$vol
        docker run --rm -v $vol:/data:ro alpine tar cf - -C /data . | tar xf - -C ./recovered_results/anonymous_volumes/$vol/ 2>/dev/null && \
            print_message "Extracted anonymous volume: $vol"
    fi
done

# 5. Check if there's a running container
print_message "Checking for running containers..."
running=$(docker ps --filter "ancestor=fault-injection-experiment:latest" --format "{{.ID}} {{.Names}}")
if [ ! -z "$running" ]; then
    print_info "Found running container(s):"
    echo "$running"
    
    # Get the container ID
    container_id=$(echo "$running" | head -1 | cut -d' ' -f1)
    
    print_info "Attempting to copy from running container: $container_id"
    mkdir -p ./recovered_results/running_container
    docker cp $container_id:/app/fault_injection/results ./recovered_results/running_container/ 2>/dev/null && \
        print_message "Copied results from running container" || \
        print_warning "No results in running container"
fi

# 6. Summary
print_message "Recovery scan complete!"

if [ -d "./recovered_results" ] && [ "$(ls -A ./recovered_results)" ]; then
    print_message "Recovered files are in ./recovered_results/"
    print_info "Directory structure:"
    find ./recovered_results -type f -name "*.json" -o -name "*.pkl" -o -name "*.csv" 2>/dev/null | head -20
else
    print_warning "No results recovered. Your experiment might have:"
    print_info "1. Failed to save due to permission issues (if using old bind mounts)"
    print_info "2. Been run in a container that was removed with 'docker rm'"
    print_info "3. Used a different image name or tag"
    
    print_message ""
    print_message "Try checking:"
    print_info "- Docker logs: docker logs <container_name>"
    print_info "- All containers: docker ps -a"
    print_info "- All volumes: docker volume ls"
fi