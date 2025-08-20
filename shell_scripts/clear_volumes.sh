#!/bin/bash

# Script to clear Docker volumes for this project
# Only removes volumes with the fault_injection prefix

echo "WARNING: This will delete fault_injection Docker volumes and their data!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Stop containers using fault_injection volumes
echo "Stopping containers using fault_injection volumes..."
docker stop fault_injection_runner 2>/dev/null

# Remove only fault_injection volumes
echo "Removing fault_injection Docker volumes..."
docker volume rm fault_injection_results 2>/dev/null
docker volume rm fault_injection_optimizer 2>/dev/null
docker volume rm fault_injection_output 2>/dev/null
docker volume rm fault_injection_checkpoints 2>/dev/null

echo "Fault injection Docker volumes have been removed."