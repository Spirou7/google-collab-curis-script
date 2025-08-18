# Parallel Optimizer Mitigation Experiment (V3)

## Overview
This script (`test_optimizer_mitigation_v3.py`) tests how different optimizers recover from fault injection by training ALL optimizers from scratch for each experiment, ensuring a fair comparison with identical training context.

## Key Features
- **Parallel Training**: All optimizers train from scratch to the injection point using identical data batches
- **Synchronized Injection**: Same fault applied to all models at the exact same point
- **Fair Comparison**: All optimizers have full training context before injection
- **Comprehensive Analysis**: Detailed visualizations and metrics for recovery comparison

## Usage

### Basic Command
```bash
python fault_injection/scripts/test_optimizer_mitigation_v3.py
```

### Command Line Arguments
- `--num-experiments`: Number of experiments to run (default: 10)
- `--steps-after-injection`: Steps to continue training after injection (default: 100)
- `--optimizers`: List of optimizers to test (default: adam sgd rmsprop adamw)
- `--seed`: Base random seed for reproducibility (default: 42)
- `--learning-rate`: Initial learning rate (default: 0.001)

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

## Differences from V2
- **V2**: Trains one optimizer, saves checkpoint, then tests others from that checkpoint
- **V3**: Trains ALL optimizers from scratch to ensure identical pre-injection context
- **V3 Advantage**: Fairer comparison as all optimizers have same training history
- **V3 Disadvantage**: Takes longer to run (trains multiple models per experiment)