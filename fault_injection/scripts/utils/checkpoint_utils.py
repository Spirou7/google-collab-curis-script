import os
import json
import tensorflow as tf
from typing import Dict


def save_post_injection_checkpoint(model: tf.keras.Model, 
                                  experiment_dir: str,
                                  injection_step: int,
                                  corrupted_state_info: Dict) -> str:
    """
    Save model state immediately after injection.
    This is the corrupted state that all optimizers will start from.
    
    Args:
        model: The model with corrupted weights
        experiment_dir: Directory for this experiment
        injection_step: Global step when injection occurred
        corrupted_state_info: Information about the corruption
    
    Returns:
        Path to checkpoint directory
    """
    print(f"\n💾 SAVING POST-INJECTION CHECKPOINT")
    print(f"   • Injection step: {injection_step}")
    print(f"   • Corruption stats:")
    print(f"     - NaN weights: {corrupted_state_info.get('nan_weights', 0)}")
    print(f"     - Inf weights: {corrupted_state_info.get('inf_weights', 0)}")
    print(f"     - Post-injection accuracy: {corrupted_state_info.get('post_injection_accuracy', 0):.4f}")
    
    checkpoint_dir = os.path.join(experiment_dir, "post_injection_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"   • Checkpoint directory: {checkpoint_dir}")
    
    # Save model weights (now corrupted)
    weights_path = os.path.join(checkpoint_dir, 'corrupted_model.weights.h5')
    print(f"   • Saving model weights to: {weights_path}")
    model.save_weights(weights_path)
    
    # Verify weights were saved
    file_size = os.path.getsize(weights_path) / (1024 * 1024)  # Convert to MB
    print(f"   • Weights file size: {file_size:.2f} MB")
    
    # Save corruption info
    info_path = os.path.join(checkpoint_dir, 'corruption_info.json')
    print(f"   • Saving corruption info to: {info_path}")
    with open(info_path, 'w') as f:
        json.dump(corrupted_state_info, f, indent=2, default=str)
    
    print(f"   ✓ Post-injection checkpoint saved successfully")
    return checkpoint_dir


def load_corrupted_checkpoint(model: tf.keras.Model, 
                             checkpoint_dir: str) -> Dict:
    """
    Load corrupted weights from checkpoint.
    
    Args:
        model: Fresh model to load weights into
        checkpoint_dir: Directory containing checkpoint
    
    Returns:
        Dictionary with corruption information
    """
    weights_path = os.path.join(checkpoint_dir, 'corrupted_model.weights.h5')
    print(f"\n💾 Loading corrupted weights from checkpoint...")
    print(f"   • Path: {weights_path}")
    model.load_weights(weights_path)
    print(f"   ✓ Corrupted weights loaded")
    
    # Load corruption info
    info_path = os.path.join(checkpoint_dir, 'corruption_info.json')
    with open(info_path, 'r') as f:
        corruption_info = json.load(f)
    
    # Verify corruption is present
    print(f"\n🔍 Verifying corruption in loaded model...")
    nan_count = sum([tf.reduce_sum(tf.cast(tf.math.is_nan(v), tf.int32)).numpy() 
                     for v in model.trainable_variables])
    inf_count = sum([tf.reduce_sum(tf.cast(tf.math.is_inf(v), tf.int32)).numpy() 
                     for v in model.trainable_variables])
    print(f"   • NaN weights: {nan_count}")
    print(f"   • Inf weights: {inf_count}")
    print(f"   ✓ Corruption confirmed in loaded model")
    
    return corruption_info


def analyze_weight_corruption(model: tf.keras.Model) -> Dict:
    """
    Analyze model weights for corruption (NaN/Inf values).
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with corruption statistics
    """
    print(f"\n🔍 ANALYZING WEIGHT CORRUPTION...")
    
    nan_count = 0
    inf_count = 0
    total_weights = 0
    corrupted_layers = []
    
    for i, var in enumerate(model.trainable_variables):
        var_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)).numpy()
        var_inf = tf.reduce_sum(tf.cast(tf.math.is_inf(var), tf.int32)).numpy()
        var_size = tf.size(var).numpy()
        
        nan_count += var_nan
        inf_count += var_inf
        total_weights += var_size
        
        if var_nan > 0 or var_inf > 0:
            print(f"   ⚠️ Layer {i}: {var_nan} NaN, {var_inf} Inf out of {var_size}")
            corrupted_layers.append({
                'layer_index': i,
                'nan_count': int(var_nan),
                'inf_count': int(var_inf),
                'total_weights': int(var_size)
            })
    
    corruption_info = {
        'nan_weights': int(nan_count),
        'inf_weights': int(inf_count),
        'total_weights': int(total_weights),
        'corruption_percentage': (nan_count + inf_count) / total_weights * 100 if total_weights > 0 else 0,
        'corrupted_layers': corrupted_layers
    }
    
    print(f"\n📊 CORRUPTION SUMMARY:")
    print(f"   • Total weights: {total_weights:,}")
    print(f"   • NaN weights: {nan_count:,} ({nan_count/total_weights*100:.2f}%)")
    print(f"   • Inf weights: {inf_count:,} ({inf_count/total_weights*100:.2f}%)")
    print(f"   • Total corruption: {corruption_info['corruption_percentage']:.2f}%")
    
    return corruption_info