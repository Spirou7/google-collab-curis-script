#!/bin/bash

# Setup script for running on computing clusters
# This handles permission issues and directory creation

set -e

echo "==================================="
echo "Cluster Setup for Fault Injection"
echo "==================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Create necessary directories
echo ""
echo "Creating required directories..."

directories=(
    "fault_injection/results"
    "fault_injection/optimizer_comparison_results"
    "output"
    "data"
    "checkpoints"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating: $dir"
        mkdir -p "$dir" || {
            echo "ERROR: Could not create $dir"
            echo "Please create it manually with: mkdir -p $SCRIPT_DIR/$dir"
        }
    else
        echo "✓ Exists: $dir"
    fi
    
    # Check if writable
    if [ -w "$dir" ]; then
        echo "  ✓ Writable"
    else
        echo "  ⚠ Not writable - fixing permissions..."
        chmod 755 "$dir" 2>/dev/null || {
            echo "  ERROR: Could not change permissions"
            echo "  Try: chmod 755 $SCRIPT_DIR/$dir"
        }
    fi
done

echo ""
echo "==================================="
echo "Docker Build Instructions"
echo "==================================="

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
    
    # Check if user can run Docker
    if docker ps &> /dev/null; then
        echo "✓ Docker daemon is accessible"
    else
        echo "⚠ Docker daemon not accessible. You may need to:"
        echo "  - Add yourself to the docker group: sudo usermod -aG docker $USER"
        echo "  - Or use sudo for docker commands"
    fi
else
    echo "⚠ Docker not found. Please install Docker or use Singularity"
fi

echo ""
echo "To build the Docker image:"
echo "  docker build -t fault-injection-experiment:latest ."
echo ""
echo "To run experiments:"
echo "  ./docker_run.sh interactive"
echo ""

# Alternative: Singularity setup (common on HPC clusters)
if command -v singularity &> /dev/null; then
    echo "==================================="
    echo "Singularity Alternative"
    echo "==================================="
    echo "Singularity is available on this system."
    echo ""
    echo "To convert Docker image to Singularity:"
    echo "  singularity build fault-injection.sif docker-daemon://fault-injection-experiment:latest"
    echo ""
    echo "To run with Singularity:"
    echo "  singularity exec --bind $PWD:/app fault-injection.sif python /app/fault_injection/scripts/test_optimizer_mitigation_v3.py"
fi

echo ""
echo "==================================="
echo "Environment Info"
echo "==================================="
echo "User: $(whoami)"
echo "UID: $(id -u)"
echo "GID: $(id -g)"
echo "Home: $HOME"
echo "Current dir: $PWD"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "Could not query GPU"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "===================================" 