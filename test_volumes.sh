#!/bin/bash

# Quick test script to verify Docker volumes work correctly

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

print_message() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_info() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_success() { echo -e "${MAGENTA}[SUCCESS]${NC} $1"; }

echo ""
echo "=========================================="
echo "   DOCKER VOLUME TEST SUITE"
echo "=========================================="
echo ""

# Step 1: Build the image if needed
print_info "Step 1: Checking Docker image..."
if docker images | grep -q "fault-injection-experiment"; then
    print_success "Image exists"
else
    print_warning "Image not found, building..."
    ./docker_run.sh build
fi

# Step 2: Run the test script in container
print_info "Step 2: Running test script in container with volumes..."
echo ""
docker run --rm \
    -v fault_injection_results:/app/fault_injection/results \
    -v fault_injection_optimizer:/app/fault_injection/optimizer_comparison_results \
    -v fault_injection_output:/app/output \
    -v fault_injection_checkpoints:/app/checkpoints \
    fault-injection-experiment:latest \
    python test_volume_save.py

echo ""
print_info "Step 3: Checking what's in the volumes..."
./docker_run.sh list-results

echo ""
print_info "Step 4: Extracting files from volumes to local directory..."
echo ""

# Try the safest extraction method
print_message "Using extract-safe method..."
./docker_run.sh extract-safe

echo ""
print_info "Step 5: Verifying extracted files..."
echo ""

if [ -d "./extracted_results" ]; then
    print_success "Extraction successful! Files found:"
    echo ""
    find ./extracted_results -type f \( -name "*.json" -o -name "*.txt" -o -name "*.csv" -o -name "*.pkl" \) | while read file; do
        echo "  ✓ $file ($(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown") bytes)"
    done
    
    echo ""
    print_success "Reading a test file to verify content:"
    if [ -f "./extracted_results/output/test_output.txt" ]; then
        echo "---"
        cat ./extracted_results/output/test_output.txt
        echo "---"
    fi
else
    print_error "No extracted_results directory found!"
    print_info "Trying alternative extraction method..."
    ./extract_volumes.sh auto
fi

echo ""
echo "=========================================="
echo "   TEST COMPLETE"
echo "=========================================="
echo ""
print_success "If you see files above, the volume system is working!"
print_info "Your 10-hour experiment can now be run with:"
print_info "./docker_run.sh optimizer --your-parameters"
echo ""