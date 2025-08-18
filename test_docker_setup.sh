#!/bin/bash

# Test script to verify Docker setup works correctly

echo "========================================="
echo "Docker Setup Test for Optimizer Script v3"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Step 3: Create results directory
echo "3. Creating results directory..."
mkdir -p docker_results
echo -e "${GREEN}✓ Created ./docker_results/${NC}"
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
echo ""

docker run --rm \
    -v $(pwd)/docker_results:/app/fault_injection/optimizer_comparison_results \
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

# Step 6: Check results were saved
echo "6. Checking if results were saved..."
RESULT_COUNT=$(find docker_results -name "parallel_run_*" -type d | wc -l)

if [ $RESULT_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Found $RESULT_COUNT result directory(ies)${NC}"
    echo ""
    echo "Result directories:"
    ls -la docker_results/
    echo ""
    
    # Check for key files
    LATEST_DIR=$(ls -td docker_results/parallel_run_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "Checking files in latest results ($LATEST_DIR):"
        
        if [ -f "$LATEST_DIR/all_injection_configs.json" ]; then
            echo -e "${GREEN}  ✓ all_injection_configs.json found${NC}"
        else
            echo -e "${RED}  ✗ all_injection_configs.json missing${NC}"
        fi
        
        if [ -d "$LATEST_DIR/experiment_000" ]; then
            echo -e "${GREEN}  ✓ experiment_000 directory found${NC}"
            
            # Check for specific files in experiment directory
            if [ -f "$LATEST_DIR/experiment_000/results.json" ]; then
                echo -e "${GREEN}    ✓ results.json found${NC}"
            fi
            if [ -f "$LATEST_DIR/experiment_000/comparison_plots.png" ]; then
                echo -e "${GREEN}    ✓ comparison_plots.png found${NC}"
            fi
        else
            echo -e "${RED}  ✗ experiment_000 directory missing${NC}"
        fi
        
        if [ -f "$LATEST_DIR/final_report.md" ]; then
            echo -e "${GREEN}  ✓ final_report.md found${NC}"
            echo ""
            echo "Preview of final report:"
            head -20 "$LATEST_DIR/final_report.md"
        fi
    fi
else
    echo -e "${RED}✗ No result directories found in docker_results/${NC}"
    echo "Contents of docker_results/:"
    ls -la docker_results/
    exit 1
fi
echo ""

echo "========================================="
echo -e "${GREEN}All tests passed! Docker setup is working correctly.${NC}"
echo "========================================="
echo ""
echo "You can now run larger experiments with:"
echo "  docker-compose up --build"
echo ""
echo "Or customize parameters in docker-compose.yml"
echo ""
echo "Results will be saved to: ./docker_results/parallel_run_*/"