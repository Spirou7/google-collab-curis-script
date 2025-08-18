# Parallel Optimizer Mitigation Experiment - Final Report

**Date**: 2025-08-18 14:35:40

**Experiments Completed**: 1/1

## Configuration

- **Optimizers Tested**: adam, sgd
- **Fault Model**: N16_RD
- **Injection Value Range**: [3.60e+02, 1.20e+08]
- **Max Target Epoch**: 3
- **Max Target Step**: 49
- **Steps After Injection**: 10

## Aggregate Results

| Optimizer | Mean Final Acc | Std Final Acc | Mean Acc Change | Positive Recovery % | Mean Degrad. Rate | Divergence % |
|-----------|---------------|---------------|-----------------|-------------------|------------------|-------------|
| adam        |        0.1570 |        0.0000 |          0.0604 |             100.0 |           0.0000 |         0.0 |
| sgd         |        0.1329 |        0.0000 |          0.0509 |             100.0 |           0.0000 |         0.0 |

## Analysis

### Best Performing Optimizer: **adam**

The adam optimizer showed the best average recovery with a mean accuracy change of 0.0604 after fault injection.

### Head-to-Head Win Rates

| Optimizer A | Optimizer B | A Win Rate (%) |
|-------------|-------------|----------------|
| adam        | sgd         |          100.0 |

## Breakdown by Injection Characteristics

### Early Injections (Epoch ≤ 1): 1 experiments

| Optimizer | Mean Acc Change | Best Case | Worst Case |
|-----------|----------------|-----------|------------|
| adam        |         0.0604 |    0.0604 |     0.0604 |
| sgd         |         0.0509 |    0.0509 |     0.0509 |

### Late Injections (Epoch > 1): 0 experiments

No experiments in this group

## Conclusions

This experiment trained all optimizers from scratch for each fault injection, ensuring a fair comparison with identical training context for all optimizers.

Key findings:
- **adam** demonstrated the best average recovery performance
- **adam** was most resilient with 100.0% positive recovery rate
