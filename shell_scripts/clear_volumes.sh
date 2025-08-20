#!/bin/bash

# Script to clear ALL Docker volumes
# WARNING: This will delete all data stored in Docker volumes!

echo "WARNING: This will delete ALL Docker volumes and their data!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Stop all running containers first (volumes in use cannot be removed)
echo "Stopping all running containers..."
docker stop $(docker ps -aq) 2>/dev/null

# Remove all volumes
echo "Removing all Docker volumes..."
docker volume rm $(docker volume ls -q) 2>/dev/null

# Alternative: Use prune to remove all unused volumes
# docker volume prune -a -f

echo "All Docker volumes have been removed."