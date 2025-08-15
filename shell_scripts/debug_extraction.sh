#!/bin/bash

# Debug extraction issues

echo "=== Debugging Extraction Process ==="
echo ""

echo "1. Checking what's in the Docker volume:"
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine sh -c "ls -la /data/run_*/" | head -20

echo ""
echo "2. Testing extraction with simple docker cp:"
# Create temp container
docker create --name temp_debug \
    -v fault_injection_optimizer:/data:ro \
    alpine sh
    
# Try to copy
echo "Attempting to copy files..."
docker cp temp_debug:/data ./test_extract 2>&1

# Check what we got
echo ""
echo "3. What was extracted to ./test_extract:"
ls -la ./test_extract/ 2>/dev/null | head -10

# Cleanup
docker rm temp_debug >/dev/null 2>&1

echo ""
echo "4. Checking current directory for extracted_results:"
ls -la | grep extracted

echo ""
echo "5. Trying manual tar extraction:"
mkdir -p manual_extract
docker run --rm \
    -v fault_injection_optimizer:/data:ro \
    alpine tar cf - -C /data . | tar xf - -C ./manual_extract/

echo ""
echo "6. What's in manual_extract:"
find ./manual_extract -type f -name "*.json" -o -name "*.png" -o -name "*.md" | head -10

echo ""
echo "=== Summary ==="
echo "Your files should be in one of these locations:"
echo "  ./extracted_results/optimizer/"
echo "  ./test_extract/"
echo "  ./manual_extract/"