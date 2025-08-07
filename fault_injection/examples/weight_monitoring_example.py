#!/usr/bin/env python3
"""
Practical Example: Integrating Weight NaN Detection into Training Loops

This example demonstrates how to integrate the weight_analyzer utilities
into your existing fault injection and training infrastructure.

The examples show both simple integration patterns and advanced monitoring
setups that work with your current codebase structure.
"""

import tensorflow as tf
import numpy as np
import os
import sys

# Add the models directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from ..models.weight_analyzer import (
    analyze_weight_corruption, 
    check_weights_for_corruption,
    log_weight_corruption_details,
    create_weight_monitoring_hook,
    WeightCorruptionStats
)


def example_basic_weight_checking():
    """
    Example 1: Basic weight checking after training steps.
    
    This shows the simplest integration - checking weights periodically
    and logging the results. Perfect for understanding what's happening
    during your fault injection experiments.
    """
    print("=== Example 1: Basic Weight Checking ===")
    
    # Create a simple model for demonstration
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    print("1. Analyzing clean model weights:")
    stats = analyze_weight_corruption(model)
    print(f"   {stats}")
    
    print(f"   Details: {stats.total_parameters} total parameters")
    print(f"   NaN percentage: {stats.nan_percentage:.3f}%")
    print(f"   Corruption percentage: {stats.corrupted_percentage:.3f}%")
    
    # Simulate some NaN injection (like your fault injection)
    print("\n2. Simulating NaN injection into first layer:")
    weights = model.layers[0].get_weights()
    weights[0][0, 0] = np.nan  # Inject a NaN
    weights[0][1, 1] = np.inf  # Inject an Inf
    model.layers[0].set_weights(weights)
    
    print("3. Analyzing corrupted model weights:")
    corrupted_stats = analyze_weight_corruption(model)
    print(f"   {corrupted_stats}")
    
    # Check with threshold
    is_corrupted, _ = check_weights_for_corruption(model, threshold_percentage=0.01)
    print(f"   Is corrupted above 0.01% threshold? {is_corrupted}")
    
    print("4. Layer-wise details:")
    for layer_name, layer_data in corrupted_stats.layer_stats.items():
        if layer_data['nan'] > 0 or layer_data['inf'] > 0:
            corruption_pct = ((layer_data['nan'] + layer_data['inf']) / layer_data['total']) * 100
            print(f"   {layer_name}: {layer_data['nan']} NaN, {layer_data['inf']} Inf "
                  f"({corruption_pct:.3f}% corrupted)")


def example_training_loop_integration():
    """
    Example 2: Integration into a training loop similar to your existing code.
    
    This shows how to monitor weights during training, similar to how you
    currently monitor for NaN in losses.
    """
    print("\n=== Example 2: Training Loop Integration ===")
    
    # Create model and data (simplified)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Dummy training data
    x_train = np.random.random((100, 10))
    y_train = np.random.randint(0, 2, (100, 1))
    
    # Create optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Create weight monitoring hook
    def mock_record(train_recorder, message):
        """Mock version of your record function"""
        print(f"[RECORDER] {message.strip()}")
    
    # Simulate train_recorder object
    class MockTrainRecorder:
        def write(self, message):
            print(f"[RECORDER] {message.strip()}")
    
    train_recorder = MockTrainRecorder()
    
    # Create the monitoring hook
    weight_monitor = create_weight_monitoring_hook(
        model,
        check_frequency=5,  # Check every 5 steps
        corruption_threshold=0.1,
        train_recorder=train_recorder
    )
    
    print("Starting training with weight monitoring...")
    
    # Training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        for step in range(10):  # 10 steps per epoch for demo
            
            # Simulate training step
            batch_x = x_train[step*10:(step+1)*10]
            batch_y = y_train[step*10:(step+1)*10]
            
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = loss_fn(batch_y, predictions)
            
            # Apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Simulate NaN injection at step 7 (like your fault injection)
            if epoch == 1 and step == 7:
                print("   [SIMULATING FAULT INJECTION]")
                weights = model.layers[0].get_weights()
                weights[0][0, 0] = np.nan
                model.layers[0].set_weights(weights)
            
            # Monitor weights using the hook
            should_terminate = weight_monitor(step, epoch)
            
            if should_terminate:
                print("   Training terminated due to weight corruption!")
                return
            
            # Regular loss monitoring (like your existing code)
            if not np.isfinite(loss):
                print(f"   Step {step}: Loss became NaN/Inf! Loss = {loss}")
                break
    
    print("Training completed successfully!")


def example_detailed_analysis():
    """
    Example 3: Detailed analysis for research purposes.
    
    This shows how to get comprehensive statistics about weight corruption,
    useful for understanding the effects of different fault injection strategies.
    """
    print("\n=== Example 3: Detailed Analysis for Research ===")
    
    # Create different models to simulate different corruption patterns
    models = {
        'small_model': tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ]),
        'medium_model': tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    }
    
    # Simulate different corruption scenarios
    corruption_scenarios = {
        'no_corruption': lambda model: None,
        'light_corruption': lambda model: inject_corruption(model, nan_count=1, inf_count=1),
        'heavy_corruption': lambda model: inject_corruption(model, nan_count=10, inf_count=5),
    }
    
    def inject_corruption(model, nan_count=1, inf_count=1):
        """Helper to inject controlled corruption"""
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    flat_kernel = kernel.flatten()
                    
                    # Inject NaNs
                    for _ in range(min(nan_count, len(flat_kernel))):
                        idx = np.random.randint(0, len(flat_kernel))
                        flat_kernel[idx] = np.nan
                    
                    # Inject Infs  
                    for _ in range(min(inf_count, len(flat_kernel))):
                        idx = np.random.randint(0, len(flat_kernel))
                        flat_kernel[idx] = np.inf
                    
                    weights[0] = flat_kernel.reshape(kernel.shape)
                    layer.set_weights(weights)
                    break  # Only corrupt first layer for demo
    
    # Analyze each combination
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        
        for scenario_name, corruption_func in corruption_scenarios.items():
            # Reset model weights
            model.build(input_shape=(None, model.layers[0].input_shape[-1]))
            
            # Apply corruption
            corruption_func(model)
            
            # Analyze
            stats = analyze_weight_corruption(model, include_layer_details=True)
            results[model_name][scenario_name] = stats
            
            print(f"\n{model_name} - {scenario_name}:")
            print(f"  Total parameters: {stats.total_parameters}")
            print(f"  NaN percentage: {stats.nan_percentage:.3f}%")
            print(f"  Inf percentage: {stats.inf_percentage:.3f}%")
            print(f"  Total corruption: {stats.corrupted_percentage:.3f}%")
            
            # Show layer details if there's corruption
            if stats.corrupted_percentage > 0:
                print("  Layer breakdown:")
                for layer_name, layer_data in stats.layer_stats.items():
                    layer_corruption = ((layer_data['nan'] + layer_data['inf']) / layer_data['total']) * 100
                    if layer_corruption > 0:
                        print(f"    {layer_name}: {layer_corruption:.3f}% corrupted")


def example_integration_with_your_db_stats():
    """
    Example 4: Integration with your existing DBStats infrastructure.
    
    This shows how to extend your DBStats class to include weight corruption tracking.
    """
    print("\n=== Example 4: Integration with DBStats ===")
    
    # Example of how you could extend your existing DBStats class
    class ExtendedDBStats:
        """Extended version of your DBStats class with weight monitoring"""
        
        def __init__(self, network, phase, dataset):
            # Your existing DBStats initialization
            self.network = network
            self.phase = phase
            self.dataset = dataset
            
            # Existing fields (from your db_stats.py)
            self.epoch_nan = -1
            self.step_nan = -1
            
            # New weight corruption tracking fields
            self.weight_corruption_history = []
            self.first_weight_corruption_epoch = -1
            self.first_weight_corruption_step = -1
            self.max_weight_corruption_percentage = 0.0
            
        def update_weight_corruption(self, epoch, step, stats: WeightCorruptionStats):
            """Update weight corruption tracking"""
            corruption_pct = stats.corrupted_percentage
            
            # Record corruption percentage
            self.weight_corruption_history.append({
                'epoch': epoch,
                'step': step,
                'corruption_percentage': corruption_pct,
                'nan_count': stats.nan_parameters,
                'inf_count': stats.inf_parameters,
                'total_parameters': stats.total_parameters
            })
            
            # Track first occurrence of weight corruption
            if corruption_pct > 0 and self.first_weight_corruption_epoch == -1:
                self.first_weight_corruption_epoch = epoch
                self.first_weight_corruption_step = step
                
            # Track maximum corruption
            if corruption_pct > self.max_weight_corruption_percentage:
                self.max_weight_corruption_percentage = corruption_pct
                
        def get_corruption_summary(self):
            """Get summary of weight corruption during training"""
            if not self.weight_corruption_history:
                return "No weight corruption detected during training"
                
            total_checks = len(self.weight_corruption_history)
            corrupted_checks = sum(1 for entry in self.weight_corruption_history 
                                 if entry['corruption_percentage'] > 0)
            
            return (f"Weight corruption summary: {corrupted_checks}/{total_checks} checks showed corruption. "
                   f"First corruption at epoch {self.first_weight_corruption_epoch}, "
                   f"step {self.first_weight_corruption_step}. "
                   f"Max corruption: {self.max_weight_corruption_percentage:.3f}%")
    
    # Demo the extended stats
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    
    db_stats = ExtendedDBStats("resnet18", "training", "cifar10")
    
    # Simulate training with weight monitoring
    print("Simulating training with extended DBStats...")
    
    for epoch in range(2):
        for step in range(5):
            # Simulate corruption injection
            if epoch == 1 and step == 2:
                weights = model.layers[0].get_weights()
                weights[0][0, 0] = np.nan
                model.layers[0].set_weights(weights)
                print(f"  Injected NaN at epoch {epoch}, step {step}")
            
            # Analyze weights
            stats = analyze_weight_corruption(model)
            db_stats.update_weight_corruption(epoch, step, stats)
            
            if stats.corrupted_percentage > 0:
                print(f"  Epoch {epoch}, Step {step}: {stats.corrupted_percentage:.3f}% corruption")
    
    print(f"\nFinal summary: {db_stats.get_corruption_summary()}")


if __name__ == "__main__":
    print("Weight Monitoring Integration Examples")
    print("=" * 50)
    
    # Run all examples
    example_basic_weight_checking()
    example_training_loop_integration()
    example_detailed_analysis()
    example_integration_with_your_db_stats()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the output above to see different integration patterns.")
    print("\nTo use in your actual code:")
    print("1. Import the weight_analyzer module")
    print("2. Use check_weights_for_corruption() for simple threshold checking")
    print("3. Use create_weight_monitoring_hook() for automatic monitoring during training")
    print("4. Use analyze_weight_corruption() for detailed analysis") 