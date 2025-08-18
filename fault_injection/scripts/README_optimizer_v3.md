# Parallel Optimizer Mitigation Experiment (V3)

## Overview
This script (`test_optimizer_mitigation_v3.py`) tests how different optimizers recover from fault injection by training ALL optimizers from scratch for each experiment, ensuring a fair comparison with identical training context.

## Key Features
- **Parallel Training**: All optimizers train from scratch to the injection point using identical data batches
- **Synchronized Injection**: Same fault applied to all models at the exact same point
- **Fair Comparison**: All optimizers have full training context before injection
- **Comprehensive Analysis**: Detailed visualizations and metrics for recovery comparison
- **Docker Support**: Fully integrated with Docker named volumes to avoid permission issues

## Running with Docker (Recommended for Remote/Cluster)

### Quick Start with Docker
```bash
# 1. Build the Docker image
./shell_scripts/docker_run.sh build

# 2. Run the experiment (no permission issues!)
./shell_scripts/docker_run.sh optimizer \
    --optimizers adam sgd \
    --num-experiments 1 \
    --steps-after-injection 10

# 3. Extract results after completion
./shell_scripts/extract_volumes.sh

# Results will be in ./extracted_results/optimizer/parallel_run_*/
```

### Docker Examples

#### Quick Test
```bash
./shell_scripts/docker_run.sh optimizer \
    --optimizers adam sgd \
    --num-experiments 1 \
    --steps-after-injection 10
```

#### Medium Experiment
```bash
./shell_scripts/docker_run.sh optimizer \
    --optimizers adam sgd rmsprop \
    --num-experiments 10 \
    --steps-after-injection 100
```

#### Large Experiment with Multiple Optimizers
```bash
./shell_scripts/docker_run.sh optimizer \
    --optimizers adam sgd rmsprop adagrad adamw \
    --num-experiments 15 \
    --steps-after-injection 100
```

#### Large Experiment (Background)
```bash
./shell_scripts/docker_run.sh background fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 50 --steps-after-injection 200 --optimizers adam sgd rmsprop adamw

# Monitor logs
docker logs -f fault_injection_runner
```

#### With GPU Support
```bash
./shell_scripts/docker_run.sh gpu fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 20 --steps-after-injection 100 --optimizers adam sgd rmsprop adamw
```

### Docker-Compose Method
```bash
# Edit docker-compose.yml to set your parameters, then:
docker-compose up --build

# Extract results after completion
./shell_scripts/extract_volumes.sh
```

### Docker Volume Management

#### List Results in Volumes
```bash
./shell_scripts/docker_run.sh list-results
```

#### Extract All Results
```bash
# Method 1: Using extraction script (recommended)
./shell_scripts/extract_volumes.sh

# Method 2: Using docker_run.sh
./shell_scripts/docker_run.sh copy-results

# Method 3: Safe extraction (no mounting)
./shell_scripts/docker_run.sh extract-safe
```

#### Backup Volumes
```bash
./shell_scripts/docker_run.sh backup
```

#### Clean Volumes (Warning: Deletes Data)
```bash
./shell_scripts/docker_run.sh clean-volumes
```

### Why Use Docker?
- **No Permission Issues**: Uses Docker named volumes (Docker-managed storage)
- **Portable**: Runs identically on any system with Docker
- **Clean**: All dependencies contained, no system pollution
- **Remote-Friendly**: Perfect for cluster/cloud deployment
- **Data Persistence**: Results saved in volumes, survive container restarts

## Running Locally (MacOS/Linux)

### Basic Command
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py
```

### Command Line Arguments
- `--optimizers`: List of optimizers to test (default: adam sgd rmsprop adamw)
  - **Note**: Unlike v2, there's no separate `--baseline` and `--test-optimizers`
  - ALL specified optimizers are trained from scratch and compared equally
- `--num-experiments`: Number of experiments to run (default: 10)
- `--steps-after-injection`: Steps to continue training after injection (default: 100)
- `--seed`: Base random seed for reproducibility (default: 42)
- `--learning-rate`: Initial learning rate (default: 0.001)

⚠️ **Important**: This script does NOT use `--baseline` or `--test-optimizers` arguments. Use `--optimizers` to specify all optimizers you want to compare.

### Example Commands

#### Quick Test (Minimal - Good for Testing)
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 1 --steps-after-injection 10 --optimizers adam sgd
```

#### Small Experiment (2 optimizers, 5 runs)
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 5 --steps-after-injection 50 --optimizers adam sgd
```

#### Medium Experiment (3 optimizers, 10 runs)
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 10 --steps-after-injection 100 --optimizers adam sgd rmsprop
```

#### Full Experiment (All optimizers, 20 runs)
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 20 --steps-after-injection 100 --optimizers adam sgd rmsprop adamw
```

#### Custom Learning Rate
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 5 --learning-rate 0.0001 --optimizers adam sgd
```

### Multi-line Commands
If you need to split the command across multiple lines, use backslash:
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 10 \
    --steps-after-injection 100 \
    --optimizers adam sgd rmsprop adamw
```

⚠️ **WARNING**: Do NOT put a line break between the script name and arguments without a backslash!

❌ **WRONG** (will use default values):
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py
    --num-experiments 1 --steps-after-injection 10
```

✅ **CORRECT**:
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py \
    --num-experiments 1 --steps-after-injection 10
```

## Output Structure

### Local Execution
The script creates a timestamped results directory:
```
parallel_optimizer_results_YYYYMMDD_HHMMSS/
├── all_injection_configs.json       # Pre-generated injection parameters
├── experiment_000/                  # Individual experiment results
│   ├── injection_config.json        # Injection parameters for this run
│   ├── injection_batch.npz          # Saved batch data for reproduction
│   ├── results.json                 # Complete results
│   ├── history_adam.csv             # Training history for Adam
│   ├── history_sgd.csv              # Training history for SGD
│   ├── comparison_plots.png         # 4-panel comparison plot
│   └── recovery_zoom.png            # Zoomed recovery period plot
├── intermediate_summary.json        # Summary after every 5 experiments
├── final_report.md                  # Comprehensive analysis report
└── summary_visualizations.png       # Aggregate analysis across all experiments
```

### Docker Execution
When running with Docker, results are saved in Docker volumes and extracted to:
```
extracted_results/
├── optimizer/                       # From fault_injection_optimizer volume
│   └── parallel_run_YYYYMMDD_HHMMSS/
│       ├── all_injection_configs.json
│       ├── experiment_000/
│       ├── intermediate_summary.json
│       ├── final_report.md
│       └── summary_visualizations.png
├── results/                         # From fault_injection_results volume
├── output/                          # From fault_injection_output volume
└── checkpoints/                     # From fault_injection_checkpoints volume
```

## Performance Considerations

### On macOS (CPU-only)
- The script runs sequentially, not in true parallel
- Each optimizer is trained one after another on the same batch
- Approximate runtime:
  - Quick test (1 exp, 2 opts, 10 steps): ~2-5 minutes
  - Small (5 exp, 2 opts, 50 steps): ~15-30 minutes
  - Medium (10 exp, 3 opts, 100 steps): ~45-90 minutes
  - Full (20 exp, 4 opts, 100 steps): ~2-4 hours

### Tips for Faster Execution
1. Start with fewer experiments (`--num-experiments 1`)
2. Use fewer optimizers (`--optimizers adam sgd`)
3. Reduce steps after injection (`--steps-after-injection 10`)
4. Run overnight for larger experiments

## Understanding the Results

### Key Metrics
- **Final Accuracy**: Model accuracy after recovery period
- **Accuracy Change**: Difference between final and post-injection accuracy
- **Degradation Rate**: Slope of accuracy change during recovery (positive = improving)
- **Divergence Rate**: Percentage of experiments where optimizer failed (NaN/Inf)

### Visualizations
1. **comparison_plots.png**: 4-panel plot showing:
   - Accuracy over time for all optimizers
   - Loss over time (log scale)
   - Recovery performance comparison
   - Degradation rates

2. **recovery_zoom.png**: Detailed view of recovery period

3. **summary_visualizations.png**: Aggregate analysis with:
   - Box plots of accuracy changes
   - Average recovery trajectories
   - Head-to-head win rate matrix
   - Degradation rate distributions
   - Recovery vs initial corruption scatter
   - Performance by injection epoch heatmap

### Final Report
The `final_report.md` includes:
- Aggregate statistics for each optimizer
- Best performing optimizer analysis
- Head-to-head comparison win rates
- Breakdown by injection timing (early vs late)
- Key findings and conclusions

## Troubleshooting

### Script uses default values
- Ensure no line break between script name and arguments
- Use backslash (`\`) for multi-line commands

### Out of memory errors
- Reduce batch size in `config.py`
- Run fewer optimizers simultaneously

### NaN/Inf errors
- This is expected behavior when models diverge after injection
- The script handles these cases and marks them as diverged

### Docker Permission Errors
- The script uses Docker named volumes to avoid permission issues
- Never use bind mounts like `-v $(pwd)/results:/app/results`
- Always extract results using the provided scripts

### No Results Found After Docker Run
```bash
# Check if data exists in volumes
./shell_scripts/docker_run.sh list-results

# If data exists, extract it
./shell_scripts/extract_volumes.sh
```

### Docker Container Exits Immediately
```bash
# Run interactively to see errors
./shell_scripts/docker_run.sh interactive

# Then manually run the script
python fault_injection/scripts/test_optimizer_mitigation_v3.py --num-experiments 1
```

## Differences from V2

### Architectural Changes
- **V2**: Trains one baseline optimizer, saves checkpoint, then tests other optimizers from that checkpoint
- **V3**: Trains ALL optimizers from scratch to ensure identical pre-injection context
- **V3 Advantage**: Fairer comparison as all optimizers have same training history
- **V3 Disadvantage**: Takes longer to run (trains multiple models per experiment)

### Command Line Arguments
- **V2 Arguments**:
  - `--baseline`: The optimizer to train initially (e.g., adam)
  - `--test-optimizers`: Other optimizers to test from checkpoint
  
- **V3 Arguments**:
  - `--optimizers`: ALL optimizers to train and compare (no baseline concept)
  
Example migration from V2 to V3:
```bash
# V2 command:
./run.sh --baseline adam --test-optimizers sgd rmsprop --num-experiments 10

# Equivalent V3 command:
./run.sh --optimizers adam sgd rmsprop --num-experiments 10
```