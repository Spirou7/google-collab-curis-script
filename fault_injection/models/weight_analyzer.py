"""
Weight Analysis Utilities for NaN Detection in TensorFlow Models

This module provides comprehensive utilities for detecting and analyzing NaN values
in neural network weights during training. It supports any arbitrary TensorFlow/Keras
model and integrates seamlessly with existing training loops.

Key Features:
- Detect NaN/Inf values in model weights
- Calculate percentage of corrupted weights
- Layer-wise analysis of weight corruption
- Integration hooks for training loops
- Detailed reporting and logging

Author: Research Team
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging


class WeightCorruptionStats:
    """
    Data class to hold weight corruption statistics.
    
    This modular design allows for easy extension and maintains
    separation of concerns between data and analysis logic.
    """
    
    def __init__(self):
        self.total_parameters: int = 0
        self.nan_parameters: int = 0
        self.inf_parameters: int = 0
        self.finite_parameters: int = 0
        self.layer_stats: Dict[str, Dict[str, int]] = {}
        
    @property
    def nan_percentage(self) -> float:
        """Calculate percentage of NaN parameters."""
        if self.total_parameters == 0:
            return 0.0
        return (self.nan_parameters / self.total_parameters) * 100.0
        
    @property
    def inf_percentage(self) -> float:
        """Calculate percentage of Inf parameters."""
        if self.total_parameters == 0:
            return 0.0
        return (self.inf_parameters / self.total_parameters) * 100.0
        
    @property
    def corrupted_percentage(self) -> float:
        """Calculate percentage of corrupted (NaN + Inf) parameters."""
        if self.total_parameters == 0:
            return 0.0
        return ((self.nan_parameters + self.inf_parameters) / self.total_parameters) * 100.0
        
    def __str__(self) -> str:
        return (f"WeightStats(total={self.total_parameters}, "
                f"NaN={self.nan_parameters}({self.nan_percentage:.2f}%), "
                f"Inf={self.inf_parameters}({self.inf_percentage:.2f}%), "
                f"corrupted={self.corrupted_percentage:.2f}%)")


def analyze_weight_corruption(model: tf.keras.Model, 
                            include_layer_details: bool = True) -> WeightCorruptionStats:
    """
    Analyze NaN and Inf corruption in model weights.
    
    This function implements a functional approach to weight analysis,
    maintaining immutability where possible and providing comprehensive
    statistics about weight corruption.
    
    Args:
        model: TensorFlow/Keras model to analyze
        include_layer_details: Whether to include per-layer statistics
        
    Returns:
        WeightCorruptionStats object containing comprehensive analysis
        
    Example:
        >>> stats = analyze_weight_corruption(my_model)
        >>> print(f"Model has {stats.nan_percentage:.2f}% NaN weights")
        >>> if stats.corrupted_percentage > 1.0:
        >>>     print("WARNING: Significant weight corruption detected!")
    """
    
    stats = WeightCorruptionStats()
    
    # Iterate through all layers and their weights
    for layer in model.layers:
        layer_name = layer.name
        layer_stats = {
            'total': 0,
            'nan': 0,
            'inf': 0,
            'finite': 0
        }
        
        # Analyze each weight tensor in the layer
        for weight_idx, weight_tensor in enumerate(layer.weights):
            # Convert to numpy for efficient analysis
            weight_values = weight_tensor.numpy()
            
            # Count different types of values
            nan_mask = np.isnan(weight_values)
            inf_mask = np.isinf(weight_values)
            finite_mask = np.isfinite(weight_values)
            
            total_count = weight_values.size
            nan_count = np.sum(nan_mask)
            inf_count = np.sum(inf_mask)
            finite_count = np.sum(finite_mask)
            
            # Update layer statistics
            layer_stats['total'] += total_count
            layer_stats['nan'] += nan_count
            layer_stats['inf'] += inf_count
            layer_stats['finite'] += finite_count
            
            # Update global statistics
            stats.total_parameters += total_count
            stats.nan_parameters += nan_count
            stats.inf_parameters += inf_count
            stats.finite_parameters += finite_count
        
        # Store layer-specific stats if requested
        if include_layer_details and layer_stats['total'] > 0:
            stats.layer_stats[layer_name] = layer_stats.copy()
    
    return stats


def check_weights_for_corruption(model: tf.keras.Model, 
                                threshold_percentage: float = 0.1) -> Tuple[bool, WeightCorruptionStats]:
    """
    Quick check for weight corruption above a threshold.
    
    This function provides a binary check for corruption detection,
    useful for early termination or alert systems in training loops.
    
    Args:
        model: TensorFlow/Keras model to check
        threshold_percentage: Percentage threshold for corruption alert
        
    Returns:
        Tuple of (is_corrupted, stats) where is_corrupted indicates
        if corruption exceeds threshold
        
    Example:
        >>> is_corrupted, stats = check_weights_for_corruption(model, threshold_percentage=1.0)
        >>> if is_corrupted:
        >>>     print("Training should be terminated due to weight corruption!")
    """
    
    stats = analyze_weight_corruption(model, include_layer_details=False)
    is_corrupted = stats.corrupted_percentage > threshold_percentage
    
    return is_corrupted, stats


def log_weight_corruption_details(stats: WeightCorruptionStats, 
                                logger: Optional[logging.Logger] = None,
                                train_recorder = None) -> None:
    """
    Log detailed weight corruption information.
    
    This function integrates with your existing logging infrastructure,
    supporting both Python logging and your custom train_recorder.
    
    Args:
        stats: WeightCorruptionStats object to log
        logger: Optional Python logger instance
        train_recorder: Your custom train recorder object
        
    Example:
        >>> stats = analyze_weight_corruption(model)
        >>> log_weight_corruption_details(stats, train_recorder=train_recorder)
    """
    
    # Prepare comprehensive log message
    main_message = (f"Weight Analysis: {stats.total_parameters} total parameters, "
                   f"{stats.nan_parameters} NaN ({stats.nan_percentage:.3f}%), "
                   f"{stats.inf_parameters} Inf ({stats.inf_percentage:.3f}%), "
                   f"Total corrupted: {stats.corrupted_percentage:.3f}%")
    
    # Log using provided mechanisms
    if logger:
        logger.info(main_message)
    
    if train_recorder:
        # Integrate with your existing record function pattern
        try:
            from fault_injection.models.inject_utils import record
            record(train_recorder, main_message + "\n")
            
            # Log layer-wise details if available
            if stats.layer_stats:
                record(train_recorder, "Layer-wise corruption details:\n")
                for layer_name, layer_data in stats.layer_stats.items():
                    if layer_data['total'] > 0:
                        layer_corruption_pct = ((layer_data['nan'] + layer_data['inf']) / layer_data['total']) * 100
                        if layer_corruption_pct > 0:
                            layer_msg = (f"  {layer_name}: {layer_data['nan']} NaN, "
                                       f"{layer_data['inf']} Inf out of {layer_data['total']} "
                                       f"({layer_corruption_pct:.3f}% corrupted)\n")
                            record(train_recorder, layer_msg)
        except ImportError:
            # Fallback if record function not available
            print(main_message)


def create_weight_monitoring_hook(model: tf.keras.Model,
                                check_frequency: int = 10,
                                corruption_threshold: float = 0.1,
                                train_recorder = None) -> callable:
    """
    Create a monitoring hook for integration into training loops.
    
    This function demonstrates functional programming principles by returning
    a closure that captures the monitoring configuration. The returned function
    can be easily integrated into existing training loops.
    
    Args:
        model: Model to monitor
        check_frequency: Check weights every N steps
        corruption_threshold: Threshold for corruption warnings
        train_recorder: Your train recorder for logging
        
    Returns:
        Callable hook function that can be called during training
        
    Example:
        >>> weight_monitor = create_weight_monitoring_hook(
        ...     model, check_frequency=10, train_recorder=train_recorder
        ... )
        >>> 
        >>> # In your training loop:
        >>> for step in range(training_steps):
        ...     # ... training code ...
        ...     should_terminate = weight_monitor(step, epoch)
        ...     if should_terminate:
        ...         break
    """
    
    def monitor_hook(step: int, epoch: int = None) -> bool:
        """
        Hook function to monitor weights during training.
        
        Args:
            step: Current training step
            epoch: Current epoch (optional)
            
        Returns:
            Boolean indicating if training should be terminated
        """
        
        # Only check at specified intervals
        if step % check_frequency != 0:
            return False
            
        # Analyze current weight corruption
        is_corrupted, stats = check_weights_for_corruption(
            model, threshold_percentage=corruption_threshold
        )
        
        # Log results
        if train_recorder:
            prefix = f"Step {step}"
            if epoch is not None:
                prefix += f", Epoch {epoch}"
            
            try:
                from fault_injection.models.inject_utils import record
                record(train_recorder, f"{prefix} - Weight Check: ")
                log_weight_corruption_details(stats, train_recorder=train_recorder)
                
                if is_corrupted:
                    record(train_recorder, f"WARNING: Weight corruption above threshold ({corruption_threshold}%)!\n")
                    
            except ImportError:
                print(f"{prefix} - {stats}")
                if is_corrupted:
                    print(f"WARNING: Weight corruption above threshold!")
        
        return is_corrupted
    
    return monitor_hook


# Integration example for your existing training loops
def integrate_with_existing_training_loop_example():
    """
    Example showing how to integrate weight monitoring into existing training.
    
    This demonstrates the modular approach and shows how the utilities
    can be seamlessly integrated into your current fault injection experiments.
    """
    
    # This is pseudocode showing integration patterns
    # Replace with your actual model and training setup
    
    # Example 1: Simple corruption check after each epoch
    """
    for epoch in range(total_epochs):
        # ... your existing training code ...
        
        # Check for corruption at end of epoch
        is_corrupted, stats = check_weights_for_corruption(model, threshold_percentage=1.0)
        log_weight_corruption_details(stats, train_recorder=train_recorder)
        
        if is_corrupted:
            record(train_recorder, "Training terminated due to weight corruption!\n")
            break
    """
    
    # Example 2: Monitoring hook integration  
    """
    # Create monitoring hook
    weight_monitor = create_weight_monitoring_hook(
        model, 
        check_frequency=50,  # Check every 50 steps
        corruption_threshold=0.5,
        train_recorder=train_recorder
    )
    
    # In your training loop
    for epoch in range(total_epochs):
        for step in range(steps_per_epoch):
            # ... your training step ...
            
            # Monitor weights
            should_terminate = weight_monitor(step, epoch)
            if should_terminate:
                early_terminate = True
                break
                
        if early_terminate:
            break
    """
    
    pass


if __name__ == "__main__":
    # Example usage for testing
    print("Weight Analysis Utilities - Testing Mode")
    print("This module provides utilities for detecting NaN values in model weights.")
    print("Import this module and use the functions in your training scripts.") 