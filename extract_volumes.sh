#!/bin/bash

# Robust extraction script for Docker volumes
# This script extracts data from Docker volumes without requiring any host mount permissions

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

# Volume names
VOLUME_RESULTS="fault_injection_results"
VOLUME_OPTIMIZER="fault_injection_optimizer"
VOLUME_OUTPUT="fault_injection_output"
VOLUME_CHECKPOINTS="fault_injection_checkpoints"

# Check if volumes exist
check_volumes() {
    print_message "Checking Docker volumes..."
    local volumes_found=false
    
    for vol in $VOLUME_RESULTS $VOLUME_OPTIMIZER $VOLUME_OUTPUT $VOLUME_CHECKPOINTS; do
        if docker volume inspect $vol >/dev/null 2>&1; then
            print_info "Found volume: $vol"
            volumes_found=true
            
            # Check volume size (approximate)
            docker run --rm -v $vol:/data:ro alpine sh -c "du -sh /data 2>/dev/null || echo 'Empty'" | while read size path; do
                print_info "  Size: $size"
            done
        else
            print_warning "Volume not found: $vol"
        fi
    done
    
    if [ "$volumes_found" = false ]; then
        print_error "No volumes found! Have you run any experiments yet?"
        exit 1
    fi
}

# Method 1: Direct tar streaming (most reliable)
extract_tar_stream() {
    print_message "Method 1: Extracting via tar stream..."
    
    # Create output directory
    mkdir -p ./extracted_results
    
    # Extract each volume separately to avoid issues
    for vol in $VOLUME_RESULTS $VOLUME_OPTIMIZER $VOLUME_OUTPUT $VOLUME_CHECKPOINTS; do
        if docker volume inspect $vol >/dev/null 2>&1; then
            case $vol in
                $VOLUME_RESULTS)
                    output_dir="./extracted_results/results"
                    ;;
                $VOLUME_OPTIMIZER)
                    output_dir="./extracted_results/optimizer"
                    ;;
                $VOLUME_OUTPUT)
                    output_dir="./extracted_results/output"
                    ;;
                $VOLUME_CHECKPOINTS)
                    output_dir="./extracted_results/checkpoints"
                    ;;
            esac
            
            mkdir -p $output_dir
            print_info "Extracting $vol to $output_dir..."
            
            # Stream tar directly without any mounting
            docker run --rm -v $vol:/data:ro alpine tar cf - -C /data . 2>/dev/null | tar xf - -C $output_dir 2>/dev/null || {
                print_warning "Could not extract $vol (might be empty)"
            }
        fi
    done
    
    print_message "Extraction complete! Files are in ./extracted_results/"
    print_info "Contents:"
    find ./extracted_results -type f 2>/dev/null | head -20 || echo "No files found"
}

# Method 2: Using docker cp with temporary container
extract_docker_cp() {
    print_message "Method 2: Extracting via docker cp..."
    
    # Remove any existing temp container
    docker rm -f volume_extractor >/dev/null 2>&1 || true
    
    # Create a container with all volumes mounted
    print_info "Creating temporary container..."
    docker create --name volume_extractor \
        -v $VOLUME_RESULTS:/volumes/results:ro \
        -v $VOLUME_OPTIMIZER:/volumes/optimizer:ro \
        -v $VOLUME_OUTPUT:/volumes/output:ro \
        -v $VOLUME_CHECKPOINTS:/volumes/checkpoints:ro \
        alpine sh
    
    # Create output directory
    mkdir -p ./extracted_results
    
    # Copy each volume's contents
    print_info "Copying files from container..."
    docker cp volume_extractor:/volumes/results ./extracted_results/ 2>/dev/null || print_warning "No results to copy"
    docker cp volume_extractor:/volumes/optimizer ./extracted_results/ 2>/dev/null || print_warning "No optimizer results to copy"
    docker cp volume_extractor:/volumes/output ./extracted_results/ 2>/dev/null || print_warning "No output to copy"
    docker cp volume_extractor:/volumes/checkpoints ./extracted_results/ 2>/dev/null || print_warning "No checkpoints to copy"
    
    # Cleanup
    docker rm -f volume_extractor >/dev/null 2>&1
    
    print_message "Extraction complete! Files are in ./extracted_results/"
    ls -la ./extracted_results/ 2>/dev/null
}

# Method 3: Create a tar backup file
create_backup() {
    print_message "Method 3: Creating backup tar file..."
    
    local backup_name="volumes_backup_$(date +%Y%m%d_%H%M%S).tar"
    
    # Remove any existing temp container
    docker rm -f volume_backup >/dev/null 2>&1 || true
    
    print_info "Creating backup container..."
    docker create --name volume_backup \
        -v $VOLUME_RESULTS:/backup/results:ro \
        -v $VOLUME_OPTIMIZER:/backup/optimizer:ro \
        -v $VOLUME_OUTPUT:/backup/output:ro \
        -v $VOLUME_CHECKPOINTS:/backup/checkpoints:ro \
        alpine tar cf /backup.tar -C /backup .
    
    # Start container to create the tar
    docker start volume_backup
    docker wait volume_backup
    
    # Copy the tar file out
    print_info "Copying backup file..."
    docker cp volume_backup:/backup.tar ./$backup_name
    
    # Cleanup
    docker rm -f volume_backup >/dev/null 2>&1
    
    print_message "Backup created: $backup_name"
    print_info "To extract: tar xf $backup_name"
}

# Main menu
show_menu() {
    echo ""
    echo "Docker Volume Extraction Tool"
    echo "=============================="
    echo "1) Check volumes (see what data exists)"
    echo "2) Extract via tar stream (recommended)"
    echo "3) Extract via docker cp"
    echo "4) Create backup tar file"
    echo "5) Exit"
    echo ""
    read -p "Choose method [1-5]: " choice
    
    case $choice in
        1) check_volumes ;;
        2) check_volumes && extract_tar_stream ;;
        3) check_volumes && extract_docker_cp ;;
        4) check_volumes && create_backup ;;
        5) exit 0 ;;
        *) print_error "Invalid choice"; show_menu ;;
    esac
}

# If run with argument, execute that method directly
if [ "$1" = "check" ]; then
    check_volumes
elif [ "$1" = "tar" ]; then
    check_volumes && extract_tar_stream
elif [ "$1" = "cp" ]; then
    check_volumes && extract_docker_cp
elif [ "$1" = "backup" ]; then
    check_volumes && create_backup
elif [ "$1" = "auto" ]; then
    # Try all methods until one works
    print_message "Trying all extraction methods..."
    check_volumes
    extract_tar_stream || extract_docker_cp || create_backup
else
    show_menu
fi