import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional
from fault_injection.core import config


def create_train_step_function(model, train_loss, train_accuracy):
    """Create a training step function."""
    @tf.function
    def train_step(iterator):
        images, labels = next(iterator)
        with tf.GradientTape() as tape:
            outputs, _, _, _ = model(images, training=True, inject=False)
            predictions = outputs['logits']
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
        
        tvars = model.trainable_variables
        gradients = tape.gradient(avg_loss, tvars)
        model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))
        
        train_loss.update_state(avg_loss)
        train_accuracy.update_state(labels, predictions)
        return avg_loss, images, labels  # Return batch data for injection
    
    return train_step


def create_recovery_train_step(model, train_loss, train_accuracy):
    """Create a training step function for recovery phase."""
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            outputs, _, _, _ = model(images, training=True, inject=False)
            predictions = outputs['logits']
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
        
        tvars = model.trainable_variables
        gradients = tape.gradient(avg_loss, tvars)
        model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))
        
        train_loss.update_state(avg_loss)
        train_accuracy.update_state(labels, predictions)
        return avg_loss
    
    return train_step


def create_get_layer_outputs_function(model):
    """Create function to get layer outputs for injection."""
    @tf.function
    def get_layer_outputs(images, inj_layer):
        outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
        return l_inputs[inj_layer], l_kernels[inj_layer], l_outputs[inj_layer]
    
    return get_layer_outputs


def perform_injection(model, back_model, images, labels, injection_config: Dict,
                     train_loss, train_accuracy, get_layer_outputs):
    """
    Perform deterministic injection with pre-generated values.
    
    Args:
        model: Forward model
        back_model: Backward model for gradient computation
        images: Input images batch
        labels: Labels batch
        injection_config: Injection configuration
        train_loss: Loss metric
        train_accuracy: Accuracy metric
        get_layer_outputs: Function to get layer outputs
    
    Returns:
        Tuple of (loss, corrupted_outputs)
    """
    inj_layer = injection_config['target_layer']
    inj_position = injection_config['injection_position']
    inj_value = injection_config['injection_value']
    
    print(f"\n⚡ PERFORMING FAULT INJECTION")
    print(f"   • Layer: {inj_layer}")
    print(f"   • Position: {inj_position}")
    print(f"   • Value: {inj_value:.2e}")
    
    with tf.GradientTape() as tape:
        # Get layer outputs for injection
        print(f"   • Getting layer outputs...")
        l_inputs, l_kernels, l_outputs = get_layer_outputs(images, inj_layer)
        
        # Log tensor shapes
        if l_outputs is not None:
            print(f"   • Output tensor shape: {l_outputs.shape}")
        
        # Import necessary injection utilities
        from fault_injection.models.inject_utils import InjType, get_inj_args_with_random_range
        
        # Create a dummy recorder for the injection function
        class DummyRecorder:
            def write(self, text):
                pass
            def flush(self):
                pass
        
        dummy_recorder = DummyRecorder()
        
        # Create a config object with our pre-generated values
        class InjectionConfig:
            def __init__(self):
                self.inj_pos = [inj_position]  # Pre-generated position
                self.inj_values = [inj_value]  # Pre-generated value
        
        inj_config = InjectionConfig()
        
        # Use the proper injection function to create InjArgs object
        print(f"   • Creating injection arguments...")
        inj_args, inj_flag = get_inj_args_with_random_range(
            InjType[injection_config['fmodel']], 
            None,  # inj_replica
            inj_layer,
            l_inputs, 
            l_kernels, 
            l_outputs, 
            dummy_recorder,
            inj_config,  # Pass our config with pre-generated values
            injection_config['injection_value'],  # min_val (use exact value)
            injection_config['injection_value']   # max_val (use exact value)
        )
        
        print(f"   • Injection type: {injection_config['fmodel']}")
        print(f"   • Injection flag: {inj_flag}")
        
        # Perform forward pass with injection
        print(f"   • Executing forward pass with injection...")
        outputs, l_inputs_inj, l_kernels_inj, l_outputs_inj = model(
            images, training=True, inject=inj_flag, inj_args=inj_args
        )
        predictions = outputs['logits']
        grad_start = outputs['grad_start']
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
        
        print(f"   • Loss after injection: {avg_loss:.6f}")
    
    # Backward pass with manual gradient computation
    print(f"   • Computing gradients...")
    man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
    manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs_inj, l_kernels_inj, 
                                                inject=False, inj_args=None)
    
    gradients = manual_gradients + golden_gradients[-2:]
    print(f"   • Applying gradients to optimizer...")
    model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
    
    train_loss.update_state(avg_loss)
    train_accuracy.update_state(labels, predictions)
    
    print(f"   ✓ Injection completed")
    return avg_loss, l_outputs_inj


def setup_training_metrics():
    """Setup training metrics."""
    print(f"\n📈 Setting up training metrics...")
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    print(f"   ✓ Metrics initialized")
    return train_loss, train_accuracy


def log_training_progress(step: int, total_steps: int, train_loss, train_accuracy, 
                         start_time: float, current_epoch: int = None):
    """Log training progress."""
    import time
    
    elapsed = time.time() - start_time
    steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
    
    if current_epoch is not None:
        print(f"   Step {step:4d}/{total_steps} | Epoch {current_epoch:2d} | "
              f"Loss: {train_loss.result():.4f} | Acc: {train_accuracy.result():.4f} | "
              f"Speed: {steps_per_sec:.1f} steps/s")
    else:
        print(f"   Step {step:3d}/{total_steps} | "
              f"Acc: {train_accuracy.result():.4f} | "
              f"Loss: {train_loss.result():.4f} | "
              f"Speed: {steps_per_sec:.1f} steps/s")