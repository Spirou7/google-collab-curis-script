#!/bin/bash

# Check for actual optimizer experiment results

echo "Checking for optimizer experiment results..."
echo "=========================================="

# Check optimizer volume for experiment directories
echo "Looking for optimizer_comparison_results directories:"
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine find /data -type d -name "optimizer_comparison_results_*" 2>/dev/null | head -20

echo ""
echo "Looking for experiment directories:"
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine find /data -type d -name "experiment_*" 2>/dev/null | head -20

echo ""
echo "Looking for results.json files:"
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine find /data -name "results.json" 2>/dev/null

echo ""
echo "Looking for final_report.md:"
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine find /data -name "final_report.md" 2>/dev/null

echo ""
echo "All files in optimizer volume (excluding test files):"
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine find /data -type f ! -name "test_*" 2>/dev/null | head -20

echo ""
echo "=========================================="
echo "If no experiment files found above, the optimizer experiment may have:"
echo "1. Failed to save (check container logs)"
echo "2. Not completed yet (still running?)"
echo "3. Saved to wrong location"