#!/bin/bash

# Test script to verify Docker setup works correctly
# Uses Docker named volumes to avoid permission issues

echo "========================================="
echo "Docker Setup Test for Optimizer Script v3"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Named volumes (same as in shell_scripts/docker_run.sh)
VOLUME_RESULTS="fault_injection_results"
VOLUME_OPTIMIZER="fault_injection_optimizer"
VOLUME_OUTPUT="fault_injection_output"
VOLUME_CHECKPOINTS="fault_injection_checkpoints"

# Step 1: Check Docker is installed
echo "1. Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
    docker --version
else
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi
echo ""

# Step 2: Check Docker Compose is installed
echo "2. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓ Docker Compose is installed${NC}"
    docker-compose --version
else
    echo -e "${YELLOW}⚠ Docker Compose not found, will use 'docker compose' command${NC}"
fi
echo ""

# Step 3: Create Docker volumes
echo "3. Creating Docker named volumes (no permission issues)..."
docker volume create $VOLUME_RESULTS >/dev/null 2>&1 || true
docker volume create $VOLUME_OPTIMIZER >/dev/null 2>&1 || true
docker volume create $VOLUME_OUTPUT >/dev/null 2>&1 || true
docker volume create $VOLUME_CHECKPOINTS >/dev/null 2>&1 || true
echo -e "${GREEN}✓ Docker volumes ready${NC}"
echo ""

# Step 4: Build Docker image
echo "4. Building Docker image..."
docker build -t curis-optimizer:test . || {
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Docker image built successfully${NC}"
echo ""

# Step 5: Run a minimal test
echo "5. Running minimal test (1 experiment, 10 steps, 2 optimizers)..."
echo "This should take about 2-5 minutes..."
echo -e "${BLUE}Note: Using Docker named volumes - no host mounting required${NC}"
echo ""

docker run --rm \
    -v $VOLUME_RESULTS:/app/fault_injection/results \
    -v $VOLUME_OPTIMIZER:/app/fault_injection/optimizer_comparison_results \
    -v $VOLUME_OUTPUT:/app/output \
    -v $VOLUME_CHECKPOINTS:/app/checkpoints \
    curis-optimizer:test \
    python fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 1 \
    --steps-after-injection 10 \
    --optimizers adam sgd

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test run completed successfully${NC}"
else
    echo -e "${RED}✗ Test run failed${NC}"
    exit 1
fi
echo ""

# Step 6: Check results were saved in Docker volumes
echo "6. Checking if results were saved in Docker volumes..."
echo -e "${BLUE}Checking volume contents...${NC}"

# List files in volumes using a temporary container
docker run --rm \
    -v $VOLUME_OPTIMIZER:/optimizer:ro \
    alpine sh -c "find /optimizer -name 'parallel_run_*' -type d 2>/dev/null | head -5" > /tmp/results_check.txt

RESULT_COUNT=$(wc -l < /tmp/results_check.txt)

if [ $RESULT_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Found result directories in Docker volume${NC}"
    echo ""
    echo "Results in volume:"
    cat /tmp/results_check.txt
    echo ""
    
    # Extract results to local directory for inspection
    echo "7. Extracting results from Docker volumes..."
    mkdir -p ./extracted_results
    
    # Use tar streaming to extract (no mounting required)
    docker run --rm \
        -v $VOLUME_OPTIMIZER:/data:ro \
        alpine tar cf - -C /data . 2>/dev/null | tar xf - -C ./extracted_results/ 2>/dev/null || {
            echo -e "${YELLOW}Partial extraction (some files may be empty)${NC}"
        }
    
    if [ -d "./extracted_results" ]; then
        echo -e "${GREEN}✓ Results extracted to ./extracted_results/${NC}"
        echo "Contents:"
        ls -la ./extracted_results/ | head -10
        
        # Check for key files
        LATEST_DIR=$(ls -td ./extracted_results/parallel_run_* 2>/dev/null | head -1)
        if [ -n "$LATEST_DIR" ]; then
            echo ""
            echo "Checking files in latest results:"
            [ -f "$LATEST_DIR/all_injection_configs.json" ] && echo -e "${GREEN}  ✓ all_injection_configs.json${NC}"
            [ -d "$LATEST_DIR/experiment_000" ] && echo -e "${GREEN}  ✓ experiment_000 directory${NC}"
            [ -f "$LATEST_DIR/final_report.md" ] && echo -e "${GREEN}  ✓ final_report.md${NC}"
        fi
    fi
else
    echo -e "${RED}✗ No results found in Docker volumes${NC}"
    echo "Checking what's in the volumes:"
    docker run --rm \
        -v $VOLUME_OPTIMIZER:/optimizer:ro \
        -v $VOLUME_RESULTS:/results:ro \
        alpine sh -c "echo '=== Optimizer Volume ==='; ls -la /optimizer 2>/dev/null | head -5; echo '=== Results Volume ==='; ls -la /results 2>/dev/null | head -5"
fi

rm -f /tmp/results_check.txt
echo ""

echo "========================================="
echo -e "${GREEN}Docker setup test complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Run experiments using docker-compose:"
echo "   docker-compose up --build"
echo ""
echo "2. Or use the shell script directly:"
echo "   ./shell_scripts/docker_run.sh optimizer --num-experiments 10 --steps-after-injection 100"
echo ""
echo "3. Extract results after experiments:"
echo "   ./shell_scripts/extract_volumes.sh"
echo ""
echo "IMPORTANT: Results are stored in Docker named volumes (no permission issues)"
echo "Use extraction scripts to copy results to your local machine"