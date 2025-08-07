"""
Injection Visualization Module for Layer-wise NaN/Inf Analysis

This module provides specialized visualization tools for analyzing the propagation
of NaN/Inf values through neural network layers during fault injection experiments.

Key Features:
- Forward pass: Track NaN/Inf in layer outputs (neuron corruption)
- Backward pass: Track NaN/Inf in layer weights (weight corruption)  
- Generates separate plots for forward and backward corruption analysis
- Integrates with the existing fault injection framework

Author: Research Team
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

# Optional seaborn import for enhanced plotting
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


class LayerCorruptionStats:
    """
    Data class to hold layer-wise corruption statistics.
    
    Modular design for tracking corruption at individual layer level.
    """
    
    def __init__(self):
        self.layer_output_corruption: Dict[str, float] = {}  # Forward pass - output corruption
        self.layer_weight_corruption: Dict[str, float] = {}  # Backward pass - weight corruption
        self.layer_names: List[str] = []
        self.timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")


def analyze_layer_outputs_corruption(layer_outputs: Dict[str, tf.Tensor]) -> Dict[str, float]:
    """
    Analyze NaN/Inf corruption in layer outputs (forward propagation analysis).
    
    Args:
        layer_outputs: Dictionary mapping layer names to their output tensors
        
    Returns:
        Dictionary mapping layer names to corruption percentages
    """
    corruption_percentages = {}
    
    for layer_name, output_tensor in layer_outputs.items():
        if output_tensor is None:
            continue
            
        # Convert to numpy for analysis
        if hasattr(output_tensor, 'numpy'):
            output_array = output_tensor.numpy()
        else:
            output_array = np.array(output_tensor)
        
        # Count total elements
        total_elements = output_array.size
        
        if total_elements == 0:
            corruption_percentages[layer_name] = 0.0
            continue
        
        # Count NaN and Inf values
        nan_count = np.sum(np.isnan(output_array))
        inf_count = np.sum(np.isinf(output_array))
        corrupted_count = nan_count + inf_count
        
        # Calculate corruption percentage
        corruption_percentage = (corrupted_count / total_elements) * 100.0
        corruption_percentages[layer_name] = corruption_percentage
        
        print(f"Layer {layer_name}: {corruption_percentage:.4f}% output corruption ({nan_count} NaN, {inf_count} Inf)")
    
    return corruption_percentages


def analyze_layer_weights_corruption(model: tf.keras.Model) -> Dict[str, float]:
    """
    Analyze NaN/Inf corruption in layer weights (backward propagation analysis).
    
    Args:
        model: TensorFlow/Keras model to analyze
        
    Returns:
        Dictionary mapping layer names to weight corruption percentages
    """
    corruption_percentages = {}
    
    for layer in model.layers:
        if not layer.trainable_weights:
            continue
            
        layer_name = layer.name
        total_corrupted = 0
        total_elements = 0
        
        # Analyze all trainable weights in this layer
        for weight_tensor in layer.trainable_weights:
            weight_array = weight_tensor.numpy()
            
            elements_in_weight = weight_array.size
            nan_count = np.sum(np.isnan(weight_array))
            inf_count = np.sum(np.isinf(weight_array))
            corrupted_in_weight = nan_count + inf_count
            
            total_corrupted += corrupted_in_weight
            total_elements += elements_in_weight
        
        if total_elements == 0:
            corruption_percentages[layer_name] = 0.0
            continue
            
        # Calculate corruption percentage for this layer
        corruption_percentage = (total_corrupted / total_elements) * 100.0
        corruption_percentages[layer_name] = corruption_percentage
        
        print(f"Layer {layer_name}: {corruption_percentage:.4f}% weight corruption")
    
    return corruption_percentages


def create_forward_corruption_plot(corruption_data: Dict[str, float], 
                                 injection_params: Dict, 
                                 save_path: str) -> None:
    """
    Create visualization for forward pass corruption (layer outputs).
    
    Modular plotting function focused on forward propagation analysis.
    """
    if not corruption_data:
        print("No forward corruption data to plot")
        return
    
    # Prepare data for plotting
    layer_names = list(corruption_data.keys())
    corruption_percentages = list(corruption_data.values())
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    bars = plt.bar(range(len(layer_names)), corruption_percentages, 
                   color='red', alpha=0.7, edgecolor='darkred', linewidth=1)
    
    # Customize plot
    plt.xlabel('Layer Name', fontsize=12, fontweight='bold')
    plt.ylabel('Output Corruption Percentage (%)', fontsize=12, fontweight='bold')
    plt.title(f'Forward Pass: Layer Output NaN/Inf Corruption\n'
              f'Model: {injection_params.get("model", "Unknown")} | '
              f'Injection: {injection_params.get("fmodel", "Unknown")} | '
              f'Target: {injection_params.get("target_layer", "Unknown")}', 
              fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars, corruption_percentages)):
        if percentage > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{percentage:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Forward corruption plot saved to: {save_path}")


def create_backward_corruption_plot(corruption_data: Dict[str, float], 
                                  injection_params: Dict, 
                                  save_path: str) -> None:
    """
    Create visualization for backward pass corruption (layer weights).
    
    Modular plotting function focused on backward propagation analysis.
    """
    if not corruption_data:
        print("No backward corruption data to plot")
        return
    
    # Prepare data for plotting
    layer_names = list(corruption_data.keys())
    corruption_percentages = list(corruption_data.values())
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    bars = plt.bar(range(len(layer_names)), corruption_percentages, 
                   color='blue', alpha=0.7, edgecolor='darkblue', linewidth=1)
    
    # Customize plot
    plt.xlabel('Layer Name', fontsize=12, fontweight='bold')
    plt.ylabel('Weight Corruption Percentage (%)', fontsize=12, fontweight='bold')
    plt.title(f'Backward Pass: Layer Weight NaN/Inf Corruption\n'
              f'Model: {injection_params.get("model", "Unknown")} | '
              f'Injection: {injection_params.get("fmodel", "Unknown")} | '
              f'Target: {injection_params.get("target_layer", "Unknown")}', 
              fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, percentage) in enumerate(zip(bars, corruption_percentages)):
        if percentage > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{percentage:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Backward corruption plot saved to: {save_path}")


def generate_injection_corruption_analysis(model: tf.keras.Model,
                                         layer_outputs: Dict[str, tf.Tensor],
                                         injection_params: Dict,
                                         output_dir: str = None) -> Tuple[str, str]:
    """
    Complete modular analysis workflow for injection corruption visualization.
    
    Generates both forward and backward corruption plots during injection.
    
    Args:
        model: The neural network model being analyzed
        layer_outputs: Dictionary of layer outputs from forward pass
        injection_params: Parameters of the current injection
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (forward_plot_path, backward_plot_path)
    """
    # Use default output directory if not specified
    if output_dir is None:
        # Use the new results directory structure
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
            "NaN"
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Analyze forward pass corruption (layer outputs)
    print("\n" + "="*60)
    print("ANALYZING FORWARD PASS CORRUPTION (Layer Outputs)")
    print("="*60)
    forward_corruption = analyze_layer_outputs_corruption(layer_outputs)
    
    # Analyze backward pass corruption (layer weights)
    print("\n" + "="*60)
    print("ANALYZING BACKWARD PASS CORRUPTION (Layer Weights)")
    print("="*60)
    backward_corruption = analyze_layer_weights_corruption(model)
    
    # Generate plot filenames - use simple names if using organized directory structure
    # Check if output_dir has organized structure (contains multiple path components)
    path_components = output_dir.split(os.sep)
    if len(path_components) > 3 and ('simulation_results' in path_components or 'results' in path_components):
        # Using organized directory structure - use simple filenames
        forward_filename = "forward_corruption.png"
        backward_filename = "backward_corruption.png"
    else:
        # Using legacy flat structure - use detailed filenames
        model_name = injection_params.get('model', 'unknown')
        stage = injection_params.get('stage', 'unknown')
        fmodel = injection_params.get('fmodel', 'unknown')
        target_layer = injection_params.get('target_layer', 'unknown_layer')
        
        # Clean layer name for filename (replace problematic characters)
        clean_layer_name = target_layer.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        forward_filename = f"fwrd_{model_name}_{stage}_{fmodel}_{clean_layer_name}_corruption_{timestamp}.png"
        backward_filename = f"bkwd_{model_name}_{stage}_{fmodel}_{clean_layer_name}_corruption_{timestamp}.png"
    
    forward_path = os.path.join(output_dir, forward_filename)
    backward_path = os.path.join(output_dir, backward_filename)
    
    # Create plots
    print("\n" + "="*60)
    print("GENERATING CORRUPTION VISUALIZATION PLOTS")
    print("="*60)
    
    create_forward_corruption_plot(forward_corruption, injection_params, forward_path)
    create_backward_corruption_plot(backward_corruption, injection_params, backward_path)
    
    print(f"\n✅ Injection corruption analysis complete!")
    print(f"📊 Forward plot: {forward_path}")
    print(f"📊 Backward plot: {backward_path}")
    
    return forward_path, backward_path


def capture_pre_injection_state(model: tf.keras.Model, 
                               layer_outputs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Capture the corruption state before injection for comparison.
    
    Args:
        model: The neural network model
        layer_outputs: Dictionary of layer outputs
        
    Returns:
        Tuple of (forward_corruption_baseline, backward_corruption_baseline)
    """
    forward_baseline = analyze_layer_outputs_corruption(layer_outputs)
    backward_baseline = analyze_layer_weights_corruption(model)
    
    return forward_baseline, backward_baseline


def create_comparison_plots(pre_injection_forward: Dict[str, float],
                           post_injection_forward: Dict[str, float],
                           pre_injection_backward: Dict[str, float],
                           post_injection_backward: Dict[str, float],
                           injection_params: Dict,
                           output_dir: str = None) -> Tuple[str, str]:
    """
    Create comparison plots showing before/after injection corruption.
    
    Modular comparison visualization for injection impact analysis.
    """
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
            "NaN"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Forward comparison plot
    if pre_injection_forward and post_injection_forward:
        layer_names = list(set(pre_injection_forward.keys()) | set(post_injection_forward.keys()))
        
        pre_values = [pre_injection_forward.get(name, 0.0) for name in layer_names]
        post_values = [post_injection_forward.get(name, 0.0) for name in layer_names]
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(layer_names))
        width = 0.35
        
        plt.bar(x - width/2, pre_values, width, label='Pre-Injection', color='lightblue', alpha=0.7)
        plt.bar(x + width/2, post_values, width, label='Post-Injection', color='red', alpha=0.7)
        
        plt.xlabel('Layer Name', fontsize=12, fontweight='bold')
        plt.ylabel('Output Corruption Percentage (%)', fontsize=12, fontweight='bold')
        plt.title(f'Forward Pass Corruption Comparison\n'
                  f'Model: {injection_params.get("model", "Unknown")} | '
                  f'Injection: {injection_params.get("fmodel", "Unknown")}', 
                  fontsize=14, fontweight='bold')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Include layer name in comparison plot filename too
        target_layer = injection_params.get('target_layer', 'unknown_layer')
        clean_layer_name = target_layer.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        forward_comparison_path = os.path.join(output_dir, f"fwrd_comparison_{clean_layer_name}_{timestamp}.png")
        plt.savefig(forward_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Backward comparison plot  
    if pre_injection_backward and post_injection_backward:
        layer_names = list(set(pre_injection_backward.keys()) | set(post_injection_backward.keys()))
        
        pre_values = [pre_injection_backward.get(name, 0.0) for name in layer_names]
        post_values = [post_injection_backward.get(name, 0.0) for name in layer_names]
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(layer_names))
        width = 0.35
        
        plt.bar(x - width/2, pre_values, width, label='Pre-Injection', color='lightgreen', alpha=0.7)
        plt.bar(x + width/2, post_values, width, label='Post-Injection', color='blue', alpha=0.7)
        
        plt.xlabel('Layer Name', fontsize=12, fontweight='bold')
        plt.ylabel('Weight Corruption Percentage (%)', fontsize=12, fontweight='bold')
        plt.title(f'Backward Pass Corruption Comparison\n'
                  f'Model: {injection_params.get("model", "Unknown")} | '
                  f'Injection: {injection_params.get("fmodel", "Unknown")}', 
                  fontsize=14, fontweight='bold')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Include layer name in comparison plot filename too
        target_layer = injection_params.get('target_layer', 'unknown_layer')
        clean_layer_name = target_layer.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        backward_comparison_path = os.path.join(output_dir, f"bkwd_comparison_{clean_layer_name}_{timestamp}.png")
        plt.savefig(backward_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return forward_comparison_path, backward_comparison_path
    
    return None, None 