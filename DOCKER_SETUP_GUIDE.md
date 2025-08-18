# Docker Setup and Execution Guide

## Prerequisites
- Docker installed on your machine
- Docker Compose installed (usually comes with Docker Desktop)
- (Optional) NVIDIA Docker runtime for GPU support

## Quick Start

### 1. Build and Run with Docker Compose (Easiest)
```bash
# Navigate to project directory
cd /Users/michael/Documents/CURIS_Research/curis_version_script_4

# Build and run with default settings (quick test)
docker-compose up --build

# Run in background
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 2. Customize Experiment Parameters
Edit the `docker-compose.yml` file's command section:
```yaml
command: >
  python fault_injection/scripts/test_optimizer_mitigation_v3.py
  --num-experiments 10
  --steps-after-injection 100
  --optimizers adam sgd rmsprop adamw
```

Then rebuild and run:
```bash
docker-compose up --build
```

## Detailed Docker Instructions

### Build the Docker Image
```bash
# Build the image
docker build -t curis-optimizer:latest .

# Verify the image was created
docker images | grep curis-optimizer
```

### Run Different Experiment Configurations

#### Quick Test (1 experiment, 2 optimizers)
```bash
docker run --rm -it \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 1 \
  --steps-after-injection 10 \
  --optimizers adam sgd
```

#### Small Experiment (5 experiments, 3 optimizers)
```bash
docker run --rm -it \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 5 \
  --steps-after-injection 50 \
  --optimizers adam sgd rmsprop
```

#### Large Experiment in Background
```bash
docker run -d \
  --name optimizer_exp_large \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 50 \
  --steps-after-injection 200 \
  --optimizers adam sgd rmsprop adamw

# Monitor progress
docker logs -f optimizer_exp_large

# Stop if needed
docker stop optimizer_exp_large
docker rm optimizer_exp_large
```

### Interactive Mode for Debugging
```bash
# Start container with bash shell
docker run --rm -it \
  -v $(pwd)/docker_results:/app/fault_injection \
  --entrypoint /bin/bash \
  curis-optimizer:latest

# Inside container, run experiments manually
python fault_injection/scripts/test_optimizer_mitigation_v3.py --help
python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 1 \
  --steps-after-injection 10 \
  --optimizers adam sgd
```

## GPU Support

### 1. Install NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Create GPU-enabled Dockerfile
Create `Dockerfile.gpu`:
```dockerfile
# Use GPU-enabled TensorFlow image
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install numpy==1.24.3 matplotlib scikit-learn pandas Pillow

COPY . /app/

# Remove CPU-only configuration
RUN sed -i 's/tf.config.set_visible_devices(\[\], '\''GPU'\'')/#tf.config.set_visible_devices(\[\], '\''GPU'\'')/g' \
    /app/fault_injection/scripts/test_optimizer_mitigation_v3.py

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "fault_injection/scripts/test_optimizer_mitigation_v3.py", "--help"]
```

### 3. Run with GPU
```bash
# Build GPU image
docker build -f Dockerfile.gpu -t curis-optimizer:gpu .

# Run with GPU support
docker run --rm -it --gpus all \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:gpu \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 10 \
  --steps-after-injection 100 \
  --optimizers adam sgd rmsprop adamw
```

### 4. Docker Compose with GPU
Uncomment GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then run:
```bash
docker-compose up --build
```

## Managing Results

### Results Location
Results are saved to `./docker_results/` on your host machine, organized as:
```
docker_results/
├── parallel_optimizer_results_YYYYMMDD_HHMMSS/
│   ├── all_injection_configs.json
│   ├── experiment_000/
│   │   ├── results.json
│   │   ├── comparison_plots.png
│   │   └── recovery_zoom.png
│   ├── intermediate_summary.json
│   ├── final_report.md
│   └── summary_visualizations.png
```

### Copy Results from Container (if not using volumes)
```bash
# List running containers
docker ps

# Copy results
docker cp container_id:/app/fault_injection/parallel_optimizer_results_* ./local_results/
```

## Multi-Stage Experiments

### Create a Script for Multiple Runs
Create `run_experiments.sh`:
```bash
#!/bin/bash

# Small test
docker run --rm \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 5 \
  --steps-after-injection 50 \
  --optimizers adam sgd

# Medium experiment
docker run --rm \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 20 \
  --steps-after-injection 100 \
  --optimizers adam sgd rmsprop

# Large experiment
docker run --rm \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 50 \
  --steps-after-injection 200 \
  --optimizers adam sgd rmsprop adamw
```

Run with:
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## Docker on Remote Machines

### 1. Save Docker Image
```bash
# On local machine
docker save curis-optimizer:latest | gzip > curis-optimizer.tar.gz

# Transfer to remote
scp curis-optimizer.tar.gz user@remote-server:~/
```

### 2. Load on Remote Machine
```bash
# On remote machine
docker load < curis-optimizer.tar.gz

# Run experiment
docker run -d --name remote_exp \
  -v ~/results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 100 \
  --steps-after-injection 200 \
  --optimizers adam sgd rmsprop adamw
```

## Monitoring and Debugging

### Check Container Status
```bash
# List all containers
docker ps -a

# Check container logs
docker logs container_name

# Follow logs in real-time
docker logs -f container_name

# Check resource usage
docker stats container_name
```

### Debug Inside Container
```bash
# Execute bash in running container
docker exec -it container_name /bin/bash

# Inside container
cd /app
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
ls -la fault_injection/
tail -f fault_injection/parallel_optimizer_results_*/experiment_000/results.json
```

### Clean Up
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all experiment containers
docker rm $(docker ps -a | grep optimizer | awk '{print $1}')

# Full cleanup (careful!)
docker system prune -a
```

## Troubleshooting

### Issue: Permission Denied on Results
```bash
# Fix permissions on results directory
sudo chown -R $(whoami):$(whoami) docker_results/
```

### Issue: Out of Memory
```bash
# Limit container memory
docker run --rm -it \
  --memory="4g" \
  --memory-swap="4g" \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py
```

### Issue: Cannot Find Module
```bash
# Rebuild with no cache
docker build --no-cache -t curis-optimizer:latest .
```

### Issue: Container Exits Immediately
```bash
# Run with interactive terminal
docker run --rm -it \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:latest \
  /bin/bash

# Then manually run the script to see errors
python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 1 \
  --steps-after-injection 10 \
  --optimizers adam sgd
```

## Best Practices

1. **Always use volumes** to persist results outside the container
2. **Name your containers** for easier management
3. **Use docker-compose** for complex configurations
4. **Tag images with versions** for reproducibility
5. **Monitor resource usage** with `docker stats`
6. **Clean up regularly** to save disk space

## Example Production Workflow

```bash
# 1. Build image with version tag
docker build -t curis-optimizer:v1.0 .

# 2. Run experiment with descriptive name
docker run -d \
  --name exp_$(date +%Y%m%d_%H%M%S) \
  -v $(pwd)/docker_results:/app/fault_injection \
  curis-optimizer:v1.0 \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 100 \
  --steps-after-injection 200 \
  --optimizers adam sgd rmsprop adamw nadam

# 3. Monitor progress
watch -n 10 'docker logs --tail 20 exp_*'

# 4. Check results
ls -la docker_results/parallel_optimizer_results_*/final_report.md

# 5. Archive results
tar -czf results_$(date +%Y%m%d).tar.gz docker_results/
```