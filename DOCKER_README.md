# Docker Setup for Fault Injection Experiments

This guide explains how to build and run the fault injection experiments in Docker on any machine.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)
- NVIDIA Docker runtime (for GPU support, optional)

## Quick Start

### 1. Build the Docker Image

```bash
# Make the run script executable
chmod +x shell_scripts/docker_run.sh

# Build the image
./shell_scripts/docker_run.sh build
```

Or using Docker directly:
```bash
docker build -t fault-injection-experiment:latest .
```

### 2. Run Experiments

#### Interactive Mode (Recommended for Development)
```bash
# Start an interactive shell in the container
./shell_scripts/docker_run.sh interactive

# Once inside the container, run experiments:
python fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --baseline adam \
    --test-optimizers sgd rmsprop \
    --num-experiments 10 \
    --steps-after-injection 200
```

#### Direct Execution
```bash
# Run optimizer mitigation experiment directly
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd rmsprop nadam \
    --num-experiments 100 \
    --steps-after-injection 200
```

#### With GPU Support
```bash
# Run with GPU acceleration (requires NVIDIA Docker)
./shell_scripts/docker_run.sh gpu fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 100
```

#### Background Execution (for long-running experiments)
```bash
# Run in background
./shell_scripts/docker_run.sh background fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 100

# Check logs
docker logs -f fault_injection_runner

# Stop the container
docker stop fault_injection_runner
```

## Using Docker Compose

### Start the Container
```bash
# Start the service
docker-compose up -d

# Execute commands in the running container
docker-compose exec fault-injection python fault_injection/scripts/test_optimizer_mitigation_v3.py --help

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Volume Mounts and Data Persistence

The Docker setup automatically mounts these directories:

| Host Directory | Container Directory | Purpose |
|---------------|-------------------|---------|
| `./fault_injection/results` | `/app/fault_injection/results` | Experiment results |
| `./fault_injection/optimizer_comparison_results` | `/app/fault_injection/optimizer_comparison_results` | Optimizer comparison outputs |
| `./output` | `/app/output` | General output files |

All files saved to these directories in the container will persist on the host machine.

## Transferring to Another Machine

### Method 1: Copy Project and Build
```bash
# On your machine: Create archive
tar -czf fault_injection_project.tar.gz \
    --exclude='*.h5' \
    --exclude='optimizer_comparison_results*' \
    --exclude='results/*' \
    --exclude='tf_env' \
    .

# Transfer to target machine
scp fault_injection_project.tar.gz user@remote-machine:/path/to/destination

# On target machine: Extract and build
tar -xzf fault_injection_project.tar.gz
./shell_scripts/docker_run.sh build
```

### Method 2: Export Docker Image
```bash
# On your machine: Save the Docker image
docker save fault-injection-experiment:latest | gzip > fault-injection-image.tar.gz

# Transfer to target machine
scp fault-injection-image.tar.gz user@remote-machine:/path/to/destination

# On target machine: Load the image
docker load < fault-injection-image.tar.gz
```

### Method 3: Use Docker Registry
```bash
# Tag and push to registry (e.g., Docker Hub)
docker tag fault-injection-experiment:latest yourusername/fault-injection:latest
docker push yourusername/fault-injection:latest

# On target machine: Pull and run
docker pull yourusername/fault-injection:latest
docker run -it yourusername/fault-injection:latest
```

## Configuration

### CPU vs GPU

The Dockerfile uses TensorFlow with GPU support by default. To use CPU-only:

1. Edit `Dockerfile`, change line 2:
   ```dockerfile
   FROM tensorflow/tensorflow:2.15.0  # CPU-only
   ```

2. Rebuild the image:
   ```bash
   ./shell_scripts/docker_run.sh build
   ```

### Resource Limits

Edit `docker-compose.yml` to adjust resource limits:
```yaml
mem_limit: 32g  # Increase memory limit
cpus: '16'      # Use more CPU cores
```

## Troubleshooting

### Permission Issues
If you encounter permission errors with saved files:
```bash
# Run container as current user
docker run -it --rm \
    --user $(id -u):$(id -g) \
    -v "$(pwd)/results:/app/results" \
    fault-injection-experiment:latest
```

### Out of Memory
If experiments fail due to memory:
1. Reduce batch size in experiments
2. Increase Docker memory limit
3. Use `--num-experiments` with smaller values

### GPU Not Detected
Ensure NVIDIA Docker runtime is installed:
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Example Workflows

### Running Multiple Experiments with Different Optimizers
```bash
# Test 1: Adam vs SGD
./shell_scripts/docker_run.sh optimizer --baseline adam --test-optimizers sgd --num-experiments 50

# Test 2: Adam vs RMSprop
./shell_scripts/docker_run.sh optimizer --baseline adam --test-optimizers rmsprop --num-experiments 50

# Test 3: All optimizers
./shell_scripts/docker_run.sh optimizer \
    --baseline adam \
    --test-optimizers sgd rmsprop nadam adadelta \
    --num-experiments 100
```

### Batch Processing
Create a script `run_experiments.sh`:
```bash
#!/bin/bash
for optimizer in sgd rmsprop nadam; do
    echo "Testing $optimizer..."
    ./shell_scripts/docker_run.sh optimizer \
        --baseline adam \
        --test-optimizers $optimizer \
        --num-experiments 20 \
        --steps-after-injection 200
    sleep 5
done
```

## Monitoring Progress

### Real-time Logs
```bash
# Follow container logs
docker logs -f fault_injection_runner

# With timestamps
docker logs -f -t fault_injection_runner
```

### Check Results
```bash
# List completed experiments
ls -la fault_injection/optimizer_comparison_results*/

# View summary reports
cat fault_injection/optimizer_comparison_results*/final_report.md
```

## Clean Up

```bash
# Remove containers
docker rm -f fault_injection_runner

# Remove unused images
docker image prune

# Clean up old results (careful!)
rm -rf fault_injection/optimizer_comparison_results_*
```

## Support

For issues or questions about the Docker setup, check:
1. Container logs: `docker logs fault_injection_runner`
2. Docker status: `docker ps -a`
3. Image list: `docker images`
4. Disk usage: `docker system df`