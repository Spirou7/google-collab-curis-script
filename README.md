# Optimizer Mitigation Experiment

## Platform-Specific Instructions

### For Intel/AMD x86_64 Systems (Linux, Windows WSL2, Intel Macs)
```bash
# Build Docker Image
./shell_scripts/docker_run.sh build

# Run Experiment
./shell_scripts/docker_run.sh optimizer \
  --num-experiments 1 \
  --steps-after-injection 10 \
  --optimizers adam sgd rmsprop
```

### For Apple Silicon Macs (M1/M2/M3)
```bash
# Option 1: Force x86_64 emulation (slower but works with current image)
# Build Docker Image with platform flag
docker build --platform linux/amd64 -t fault-injection-experiment:latest .

# Run Experiment with platform flag
docker run -it --rm --platform linux/amd64 \
  --name fault_injection_runner \
  -v fault_injection_results:/app/fault_injection/results \
  -v fault_injection_optimizer:/app/fault_injection/optimizer_comparison_results \
  -v fault_injection_output:/app/output \
  -v fault_injection_checkpoints:/app/checkpoints \
  -e HOME=/app \
  -w /app \
  fault-injection-experiment:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 5 \
  --steps-after-injection 100 \
  --optimizers adam sgd rmsprop

# Option 2: Modify Dockerfile for ARM64 (faster, recommended)
# Change line 2 in Dockerfile from:
#   FROM tensorflow/tensorflow:2.13.0-gpu
# To:
#   FROM tensorflow/tensorflow:2.13.0
# Then run normally:
./shell_scripts/docker_run.sh build
./shell_scripts/docker_run.sh optimizer \
  --num-experiments 5 \
  --steps-after-injection 100 \
  --optimizers adam sgd rmsprop
```

## Quick Start (All Platforms)

### Extract Results
```bash
./shell_scripts/docker_run.sh copy-results
# or
./shell_scripts/extract_volumes.sh
```

## Available Commands

- `build`: Build the Docker image
- `optimizer [ARGS]`: Run optimizer mitigation experiment
- `interactive`: Start interactive bash shell
- `gpu [SCRIPT] [ARGS]`: Run with GPU support
- `copy-results`: Extract all results to ./extracted_results/
- `list-results`: List files in Docker volumes
- `backup`: Create tar.gz backup of volumes

## Configuration Parameters

- `--num-experiments`: Number of experiments to run (default: 10)
- `--steps-after-injection`: Training steps after fault injection (default: 100)
- `--optimizers`: Space-separated list of optimizers to test
- `--learning-rate`: Learning rate (default: 0.001)
- `--base-seed`: Random seed (default: 42)

## Results

Results are stored in Docker volumes and can be extracted using:
```bash
./shell_scripts/docker_run.sh copy-results
```

Files will be saved to ./extracted_results/ containing:
- CSV files with experiment metrics
- JSON metadata files
- Checkpoint files for model recovery
