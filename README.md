# Optimizer Mitigation Experiment

## Quick Start

### Build Docker Image
```bash
docker build -t curis-optimizer:latest .
```

### Run Experiment
```bash
docker run -v $(pwd)/results:/app/results curis-optimizer:latest \
  python fault_injection/scripts/test_optimizer_mitigation_v3.py \
  --num-experiments 5 \
  --steps-after-injection 100 \
  --optimizers adam sgd rmsprop
```

## Configuration Parameters

- `--num-experiments`: Number of experiments to run (default: 10)
- `--steps-after-injection`: Training steps after fault injection (default: 100)
- `--optimizers`: Space-separated list of optimizers to test (e.g., adam sgd rmsprop)
- `--learning-rate`: Learning rate for optimizers (default: 0.001)
- `--base-seed`: Random seed for reproducibility (default: 42)

## Using Docker Compose

```bash
# Edit docker-compose.yml to modify experiment parameters
docker-compose up
```

## Results

Results are saved to the mounted volume at `/app/results` containing:
- CSV files with experiment metrics
- JSON metadata files
- Checkpoint files for model recovery