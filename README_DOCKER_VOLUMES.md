# Docker Named Volumes Solution

## Overview
This setup uses Docker named volumes instead of bind mounts to completely avoid host filesystem permission issues.

## Quick Start

### Using docker_run.sh (Recommended)
```bash
# Build the image
./docker_run.sh build

# Run an experiment
./docker_run.sh optimizer --num-experiments 10

# Extract results to your current directory
./docker_run.sh copy-results

# View files in volumes
./docker_run.sh list-results
```

### Using docker-compose
```bash
# Build and start
docker-compose up -d

# Run interactive shell
docker-compose exec fault-injection bash

# Run an experiment
docker-compose exec fault-injection python fault_injection/scripts/test_optimizer_mitigation_v3.py

# Stop
docker-compose down
```

## How It Works

1. **Named Volumes**: Docker creates and manages storage internally at `/var/lib/docker/volumes/`
2. **No Host Permissions Needed**: You never mount your restricted directories
3. **Data Persistence**: Volumes persist between container runs
4. **Easy Extraction**: Copy files to your current directory when needed

## Key Commands

### File Management
- `./docker_run.sh copy-results` - Extract ALL results to `./extracted_results/`
- `./docker_run.sh copy-single FILE` - Copy specific file
- `./docker_run.sh list-results` - See what's in volumes
- `./docker_run.sh backup` - Create tar.gz backup

### Volume Management
- `./docker_run.sh volume-info` - Show volume details
- `./docker_run.sh clean-volumes` - Delete volumes (careful!)

## Workflow Example

```bash
# 1. Build image
./docker_run.sh build

# 2. Run experiment
./docker_run.sh optimizer --baseline adam --num-experiments 100

# 3. Check results
./docker_run.sh list-results

# 4. Extract to current directory
./docker_run.sh copy-results

# 5. Files are now in ./extracted_results/
ls ./extracted_results/
```

## Direct Docker Commands

If you prefer using Docker directly:

```bash
# Create volumes
docker volume create fault_injection_results
docker volume create fault_injection_optimizer
docker volume create fault_injection_output

# Run with volumes
docker run -it --rm \
  -v fault_injection_results:/app/fault_injection/results \
  -v fault_injection_optimizer:/app/fault_injection/optimizer_comparison_results \
  -v fault_injection_output:/app/output \
  fault-injection-experiment:latest \
  python your_script.py

# Extract files using helper container
docker run --rm \
  -v fault_injection_results:/source:ro \
  -v $(pwd):/dest \
  alpine cp -r /source/* /dest/
```

## Why This Works

- **Bind mounts** (`-v /host/path:/container/path`) require host permissions
- **Named volumes** (`-v volume_name:/container/path`) are Docker-managed
- Docker has full control over its own volume storage
- No interaction with restricted host directories
- Files copied out only to directories where you have permissions

## Troubleshooting

### If copy-results fails
Try using sudo or ensure current directory is writable:
```bash
mkdir extracted_results
chmod 755 extracted_results
./docker_run.sh copy-results
```

### To inspect volumes directly
```bash
docker volume inspect fault_injection_results
```

### To manually browse volume contents
```bash
docker run --rm -it \
  -v fault_injection_results:/data:ro \
  alpine sh
# Then: ls /data
```