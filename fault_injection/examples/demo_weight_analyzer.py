#!/usr/bin/env python3
"""
Simple Demo: Weight NaN Analysis for Any TensorFlow Model

Run this script to see the weight analysis functionality in action.
This demonstrates how to detect NaN/Inf values in model weights and
get percentage statistics.

Usage:
    python demo_weight_analyzer.py
"""

import tensorflow as tf
import numpy as np
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from ..models.weight_analyzer import (
        analyze_weight_corruption,
        check_weights_for_corruption,
        WeightCorruptionStats
    )
    print("✓ Successfully imported weight analyzer utilities")
except ImportError as e:
    print(f"❌ Error importing weight analyzer: {e}")
    print("Make sure weight_analyzer.py is in the models/ directory")
    sys.exit(1)


def create_demo_model():
    """Create a demo model similar to your ResNet architectures"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Build the model
    model.build(input_shape=(None, 32, 32, 3))
    return model


def demo_clean_model_analysis():
    """Demo 1: Analyze a clean model with no corruption"""
    print("\n" + "="*60)
    print("DEMO 1: Clean Model Analysis")
    print("="*60)
    
    model = create_demo_model()
    
    print(f"Model architecture: {len(model.layers)} layers")
    print("Analyzing weight corruption...")
    
    stats = analyze_weight_corruption(model)
    
    print(f"\nResults:")
    print(f"  Total parameters: {stats.total_parameters:,}")
    print(f"  NaN parameters: {stats.nan_parameters}")
    print(f"  Inf parameters: {stats.inf_parameters}")
    print(f"  NaN percentage: {stats.nan_percentage:.6f}%")
    print(f"  Inf percentage: {stats.inf_percentage:.6f}%")
    print(f"  Total corruption: {stats.corrupted_percentage:.6f}%")
    
    # Check with threshold
    is_corrupted, _ = check_weights_for_corruption(model, threshold_percentage=0.001)
    print(f"\nIs corrupted above 0.001% threshold? {is_corrupted}")


def demo_injected_corruption():
    """Demo 2: Inject some NaN/Inf values and analyze"""
    print("\n" + "="*60)
    print("DEMO 2: Injected Corruption Analysis")
    print("="*60)
    
    model = create_demo_model()
    
    print("Injecting corruption into model weights...")
    
    # Inject NaN and Inf values into the first convolutional layer
    layer = model.layers[0]  # First Conv2D layer
    weights = layer.get_weights()
    
    if len(weights) > 0:
        kernel = weights[0]  # The weight tensor
        
        print(f"  Original kernel shape: {kernel.shape}")
        print(f"  Original kernel size: {kernel.size}")
        
        # Inject some NaN values
        kernel_flat = kernel.flatten()
        nan_indices = np.random.choice(len(kernel_flat), size=5, replace=False)
        inf_indices = np.random.choice(len(kernel_flat), size=3, replace=False)
        
        for idx in nan_indices:
            kernel_flat[idx] = np.nan
        for idx in inf_indices:
            kernel_flat[idx] = np.inf
            
        weights[0] = kernel_flat.reshape(kernel.shape)
        layer.set_weights(weights)
        
        print(f"  Injected 5 NaN values and 3 Inf values")
    
    # Analyze the corrupted model
    print("\nAnalyzing corrupted model...")
    stats = analyze_weight_corruption(model)
    
    print(f"\nResults after corruption:")
    print(f"  Total parameters: {stats.total_parameters:,}")
    print(f"  NaN parameters: {stats.nan_parameters}")
    print(f"  Inf parameters: {stats.inf_parameters}")
    print(f"  NaN percentage: {stats.nan_percentage:.6f}%")
    print(f"  Inf percentage: {stats.inf_percentage:.6f}%")
    print(f"  Total corruption: {stats.corrupted_percentage:.6f}%")
    
    # Show layer-wise breakdown
    print(f"\nLayer-wise corruption breakdown:")
    for layer_name, layer_data in stats.layer_stats.items():
        if layer_data['nan'] > 0 or layer_data['inf'] > 0:
            corruption_pct = ((layer_data['nan'] + layer_data['inf']) / layer_data['total']) * 100
            print(f"  {layer_name}: {layer_data['nan']} NaN, {layer_data['inf']} Inf "
                  f"out of {layer_data['total']} total ({corruption_pct:.6f}% corrupted)")
    
    # Check various thresholds
    print(f"\nThreshold checks:")
    thresholds = [0.001, 0.01, 0.1, 1.0]
    for threshold in thresholds:
        is_corrupted, _ = check_weights_for_corruption(model, threshold_percentage=threshold)
        print(f"  Above {threshold}% threshold? {is_corrupted}")


def demo_layer_wise_analysis():
    """Demo 3: Layer-wise corruption analysis"""
    print("\n" + "="*60)
    print("DEMO 3: Layer-wise Corruption Analysis")
    print("="*60)
    
    model = create_demo_model()
    
    print("Injecting different corruption patterns in different layers...")
    
    # Inject different patterns in different layers
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            kernel = weights[0]
            kernel_flat = kernel.flatten()
            
            if i == 0:  # First layer: heavy NaN corruption
                nan_count = min(10, len(kernel_flat))
                indices = np.random.choice(len(kernel_flat), size=nan_count, replace=False)
                for idx in indices:
                    kernel_flat[idx] = np.nan
                print(f"  Layer {i} ({layer.name}): Injected {nan_count} NaN values")
                
            elif i == 2:  # Third layer: Inf corruption  
                inf_count = min(5, len(kernel_flat))
                indices = np.random.choice(len(kernel_flat), size=inf_count, replace=False)
                for idx in indices:
                    kernel_flat[idx] = np.inf
                print(f"  Layer {i} ({layer.name}): Injected {inf_count} Inf values")
                
            elif i == 6:  # Dense layer: mixed corruption
                nan_count = min(3, len(kernel_flat))
                inf_count = min(2, len(kernel_flat))
                nan_indices = np.random.choice(len(kernel_flat), size=nan_count, replace=False)
                inf_indices = np.random.choice(len(kernel_flat), size=inf_count, replace=False)
                
                for idx in nan_indices:
                    kernel_flat[idx] = np.nan
                for idx in inf_indices:
                    kernel_flat[idx] = np.inf
                print(f"  Layer {i} ({layer.name}): Injected {nan_count} NaN + {inf_count} Inf values")
            
            # Update weights
            weights[0] = kernel_flat.reshape(kernel.shape)
            layer.set_weights(weights)
    
    # Analyze with layer details
    print(f"\nAnalyzing layer-wise corruption...")
    stats = analyze_weight_corruption(model, include_layer_details=True)
    
    print(f"\nOverall Results:")
    print(f"  Total parameters: {stats.total_parameters:,}")
    print(f"  Total corruption: {stats.corrupted_percentage:.4f}%")
    
    print(f"\nDetailed Layer Breakdown:")
    for layer_name, layer_data in stats.layer_stats.items():
        total = layer_data['total']
        nan_count = layer_data['nan']
        inf_count = layer_data['inf']
        
        if total > 0:
            nan_pct = (nan_count / total) * 100
            inf_pct = (inf_count / total) * 100
            total_corruption_pct = ((nan_count + inf_count) / total) * 100
            
            print(f"  {layer_name}:")
            print(f"    Total parameters: {total:,}")
            print(f"    NaN: {nan_count} ({nan_pct:.4f}%)")
            print(f"    Inf: {inf_count} ({inf_pct:.4f}%)")
            print(f"    Total corruption: {total_corruption_pct:.4f}%")


def demo_practical_usage():
    """Demo 4: Practical usage patterns for training loops"""
    print("\n" + "="*60)
    print("DEMO 4: Practical Usage Patterns")
    print("="*60)
    
    model = create_demo_model()
    
    print("Simulating a training loop with weight monitoring...")
    
    # Simulate training steps with periodic weight checking
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        for step in range(5):
            
            # Simulate fault injection at specific points (like your experiments)
            if epoch == 1 and step == 2:
                print(f"  Step {step}: INJECTING FAULT (simulating your injection experiments)")
                layer = model.layers[0]
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]
                    kernel_flat = kernel.flatten()
                    # Inject a few NaN values
                    for i in range(3):
                        idx = np.random.randint(0, len(kernel_flat))
                        kernel_flat[idx] = np.nan
                    weights[0] = kernel_flat.reshape(kernel.shape)
                    layer.set_weights(weights)
            
            # Check weights every few steps
            if step % 2 == 0:  # Check every 2 steps
                is_corrupted, stats = check_weights_for_corruption(
                    model, threshold_percentage=0.01
                )
                
                print(f"  Step {step}: Weight check - "
                      f"{stats.corrupted_percentage:.4f}% corrupted")
                
                if is_corrupted:
                    print(f"    ⚠️  WARNING: Corruption above threshold!")
                    print(f"    Details: {stats.nan_parameters} NaN, "
                          f"{stats.inf_parameters} Inf out of {stats.total_parameters} total")
                    
                    # In real training, you might want to terminate here
                    print(f"    (In real training, you might terminate here)")
    
    print(f"\nFinal analysis:")
    final_stats = analyze_weight_corruption(model)
    print(f"  Final corruption: {final_stats.corrupted_percentage:.4f}%")


def main():
    """Run all demos"""
    print("Weight NaN Analysis Demo for TensorFlow Models")
    print("This demo shows how to detect and analyze NaN/Inf values in model weights")
    print("Perfect for fault injection research and training monitoring!")
    
    try:
        demo_clean_model_analysis()
        demo_injected_corruption()
        demo_layer_wise_analysis()
        demo_practical_usage()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. Use analyze_weight_corruption(model) for detailed analysis")
        print("2. Use check_weights_for_corruption(model, threshold) for quick checks")
        print("3. Monitor weights during training to detect corruption early")
        print("4. Get layer-wise breakdown to understand corruption patterns")
        print("\nThis can help you understand:")
        print("- How fault injection affects different layers")
        print("- When corruption starts during training")
        print("- What percentage of weights become corrupted")
        print("- Which layers are most vulnerable to your injection strategies")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure TensorFlow is installed and weight_analyzer.py is available")


if __name__ == "__main__":
    main() 