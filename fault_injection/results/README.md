# Results Directory

This directory contains simulation results from fault injection experiments. 

**Note:** Result files are not tracked in git to save repository space. Only this README is committed.

## Directory Structure
```
results/
└── NaN/
    └── [model]/
        └── [stage]/
            └── [layer]/
                └── [worker]/
                    └── [timestamp]/
                        ├── accuracy_plot.png
                        ├── forward_corruption.png
                        ├── backward_corruption.png
                        └── training_log.txt
```

Results are automatically organized by the injection scripts.