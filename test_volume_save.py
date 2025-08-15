#!/usr/bin/env python3
"""
Simple test script to verify Docker volume saving and extraction works.
This will create test files in each volume mount point.
"""

import os
import json
import datetime
import pickle
import csv
from pathlib import Path

def create_test_files():
    """Create test files in each volume mount location."""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_data = {
        "test_id": f"test_{timestamp}",
        "timestamp": timestamp,
        "message": "This is a test file to verify volume mounting works!",
        "experiment_info": {
            "duration": "10 seconds",
            "purpose": "volume test",
            "success": True
        }
    }
    
    # Test directories that map to our Docker volumes
    test_locations = [
        ("/app/fault_injection/results", "test_results.json"),
        ("/app/fault_injection/optimizer_comparison_results", "test_optimizer.json"),
        ("/app/output", "test_output.txt"),
        ("/app/checkpoints", "test_checkpoint.pkl")
    ]
    
    print("=" * 60)
    print("DOCKER VOLUME SAVE TEST")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print("-" * 60)
    
    results = []
    
    for directory, filename in test_locations:
        try:
            # Create directory if it doesn't exist
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            filepath = os.path.join(directory, filename)
            
            # Save different file types
            if filename.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(test_data, f, indent=2)
                print(f"✓ Created JSON: {filepath}")
                
            elif filename.endswith('.txt'):
                with open(filepath, 'w') as f:
                    f.write(f"Test file created at {timestamp}\n")
                    f.write("This proves that Docker volumes are working!\n")
                    f.write(f"Location: {directory}\n")
                print(f"✓ Created TXT: {filepath}")
                
            elif filename.endswith('.pkl'):
                with open(filepath, 'wb') as f:
                    pickle.dump(test_data, f)
                print(f"✓ Created PKL: {filepath}")
            
            # Also create a CSV for good measure
            csv_path = os.path.join(directory, f"test_{timestamp}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'test_id', 'status'])
                writer.writerow([timestamp, f"test_{timestamp}", 'success'])
            print(f"✓ Created CSV: {csv_path}")
            
            # Verify file was created and is readable
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                results.append(f"  {filepath}: {size} bytes")
            
        except Exception as e:
            print(f"✗ ERROR in {directory}: {str(e)}")
            results.append(f"  ERROR: Could not write to {directory}")
    
    print("-" * 60)
    print("VERIFICATION:")
    for result in results:
        print(result)
    
    print("-" * 60)
    print("TEST COMPLETE!")
    print("Now run: ./shell_scripts/docker_run.sh extract-safe")
    print("Or: ./shell_scripts/extract_volumes.sh")
    print("To extract these test files to your local machine.")
    print("=" * 60)
    
    # Create a summary file
    summary_path = "/app/TEST_RUN_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Volume Test Run Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Files created:\n")
        for result in results:
            f.write(f"{result}\n")
    
    return True

if __name__ == "__main__":
    success = create_test_files()
    exit(0 if success else 1)