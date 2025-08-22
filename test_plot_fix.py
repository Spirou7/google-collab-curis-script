#!/usr/bin/env python3
"""
Quick test script to verify the optimizer state plotting fixes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fault_injection.scripts.test_optimizer_mitigation_v4 import test_optimizer_mitigation

# Run a minimal test with just 1 experiment and fewer steps
print("Testing optimizer state plotting fixes...")
print("=" * 60)

results = test_optimizer_mitigation(
    model='resnet18',  # Use a specific model for consistency
    stage='fwrd_inject',  # Forward injection
    fmodel='RD',  # Random fault model
    target_epoch=1,  # Early injection
    target_step=10,
    learning_rate=0.001,
    optimizers_to_test=['adam', 'sgd', 'sgd_vanilla'],  # Test different optimizer types
    num_experiments=1,  # Just 1 experiment for testing
    steps_after_injection=50,  # Fewer steps for quick test
    seed=12345  # Fixed seed for reproducibility
)

print("\n" + "=" * 60)
print("Test completed!")
print("Check the results directory for the generated plots.")
print("Look specifically at 'comparison_plots.png' in the experiment folder.")
print("The 'Individual Optimizer State Variables Over Time' plot should now:")
print("  1. Show actual data lines (not empty)")
print("  2. Have proper legends for each optimizer")
print("  3. Show the injection point with a red dashed line")
print("  4. Have appropriate axis scales")