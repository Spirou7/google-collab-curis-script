import tensorflow as tf
import random
import numpy as np
import os
import csv
import json
import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
import sys
import pickle
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fault_injection.models.inject_utils import (
    choose_random_layer, InjType, get_inj_args_with_random_range,
    get_inj_args, get_replay_args, is_input_target, is_weight_target, 
    is_output_target
)
from fault_injection.models.resnet import resnet_18
from fault_injection.models.backward_resnet import backward_resnet_18
from fault_injection.models.resnet_nobn import resnet_18_nobn
from fault_injection.models.backward_resnet_nobn import backward_resnet_18_nobn
from fault_injection.models import efficientnet
from fault_injection.models import backward_efficientnet
from fault_injection.models import densenet
from fault_injection.models import backward_densenet
from fault_injection.models import nf_resnet
from fault_injection.models import backward_nf_resnet
from fault_injection.core import config
from fault_injection.data.prepare_data import generate_datasets

# Configure TensorFlow for CPU on MacOS
tf.config.set_visible_devices([], 'GPU')
tf.config.set_soft_device_placement(True)

# Golden gradient indices for different models (from random_injection.py)
golden_grad_idx = {
    'resnet18': -2,
    'resnet18_nobn': -2,
    'resnet18_sgd': -2,
    'effnet': -4,
    'densenet': -2,
    'nfnet': -2
}


class SequentialOptimizerExperiment:
    """
    Refactored optimizer mitigation experiment that follows random_injection.py pattern.
    Trains optimizers sequentially with identical injection parameters.
    """
    
    def __init__(self,
                 # Injection parameters (can be fixed or random)
                 model=None, stage=None, fmodel=None,
                 target_layer=None, target_epoch=None, target_step=None,
                 learning_rate=None, inj_pos=None, inj_values=None,
                 min_val=None, max_val=None, seed=None, max_global_steps=None,
                 min_epoch=None, max_epoch=None,
                 # Optimizer comparison parameters
                 optimizers_to_test=None, num_experiments=10,
                 steps_after_injection=100,
                 clear_optimizer_state_on_nan=False):
        """
        Initialize experiment with flexible parameters matching random_injection.py
        
        Args:
            model: Model name or None for random selection
            stage: 'fwrd_inject' or 'bkwd_inject' or None for random
            fmodel: Fault model type or None for random
            target_layer: Specific layer or None for random
            target_epoch: Epoch to inject or None for random
            target_step: Step to inject or None for random
            learning_rate: Learning rate or None for random
            inj_pos: Injection position or None for random
            inj_values: Injection values or None for random
            min_val: Min injection value for range
            max_val: Max injection value for range
            seed: Random seed
            max_global_steps: Early stopping criterion
            min_epoch: Minimum epoch for random selection (default: 0)
            max_epoch: Maximum epoch for random selection (default: 10)
            optimizers_to_test: List of optimizer names to compare
            num_experiments: Number of experiments to run
            steps_after_injection: Maximum steps to continue after injection (stops early if 1.25x recovery achieved)
            clear_optimizer_state_on_nan: Clear optimizer internal states when NaN/Inf detected (default: False)
        """
        # Store injection parameters
        self.model = model
        self.stage = stage
        self.fmodel = fmodel
        self.target_layer = target_layer
        self.target_epoch = target_epoch
        self.target_step = target_step
        self.learning_rate = learning_rate
        self.inj_pos = inj_pos
        self.inj_values = inj_values
        self.min_val = min_val
        self.max_val = max_val
        self.seed = seed if seed is not None else 42
        self.max_global_steps = max_global_steps
        self.min_epoch = min_epoch if min_epoch is not None else 0
        self.max_epoch = max_epoch if max_epoch is not None else 10
        
        # Optimizer comparison parameters
        self.optimizers_to_test = optimizers_to_test or ['adam', 'sgd', 'sgd_vanilla', 'rmsprop', 'adamw']
        self.num_experiments = num_experiments
        self.steps_after_injection = steps_after_injection
        self.clear_optimizer_state_on_nan = clear_optimizer_state_on_nan
        
        # Available options (from random_injection.py)
        self.available_models = ['resnet18', 'resnet18_sgd', 'resnet18_nobn', 'effnet', 'densenet', 'nfnet']
        self.available_stages = ['fwrd_inject', 'bkwd_inject']
        self.available_fmodels = ['INPUT', 'INPUT_16', 'WT', 'WT_16', 'RBFLIP', 'RD', 'RD_CORRECT', 
                                  'ZERO', 'N16_RD', 'N16_RD_CORRECT', 'RD_GLB', 'RD_CORRECT_GLB',
                                  'N64_INPUT', 'N64_WT', 'N64_INPUT_16', 'N64_WT_16', 
                                  'N64_INPUT_GLB', 'N64_WT_GLB']
        self.learning_rate_range = [0.0001, 0.001, 0.01, 0.1]
        
        # Setup results directory with proper Docker volume support
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if we're in Docker by looking for the /app directory
        if os.path.exists("/app"):
            # Running in Docker container - use the mounted volume path
            # The volume is mounted at /app/fault_injection/results
            volume_path = "/app/fault_injection/results"
            os.makedirs(volume_path, exist_ok=True)
            self.results_base_dir = os.path.join(volume_path, f"optimizer_comparison_{timestamp}")
        else:
            # Running locally
            self.results_base_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "results",
                f"optimizer_comparison_{timestamp}"
            )
        
        os.makedirs(self.results_base_dir, exist_ok=True)
        print(f"Results will be saved to: {self.results_base_dir}")
    
    def get_random_injection_params(self, experiment_seed):
        """
        Generate injection parameters for an experiment.
        Uses provided values or generates random ones.
        """
        # Set seed for this experiment
        random.seed(experiment_seed)
        np.random.seed(experiment_seed)
        
        # Generate parameters (use provided or random)
        params = {
            'model': self.model if self.model else random.choice(self.available_models),
            'stage': self.stage if self.stage else random.choice(self.available_stages),
            'fmodel': self.fmodel if self.fmodel else random.choice(self.available_fmodels),
            'target_layer': self.target_layer,  # Will be set later if None
            'target_epoch': self.target_epoch if self.target_epoch is not None else random.randint(self.min_epoch, self.max_epoch),
            'target_step': self.target_step if self.target_step is not None else random.randint(0, 49),
            'learning_rate': self.learning_rate if self.learning_rate else random.choice(self.learning_rate_range),
            'inj_pos': self.inj_pos,
            'inj_values': self.inj_values,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'seed': experiment_seed,
            'max_global_steps': self.max_global_steps
        }
        
        # Choose random layer if not specified
        if params['target_layer'] is None:
            params['target_layer'] = choose_random_layer(params['model'], params['stage'])
        
        return params
    
    def get_model(self, m_name, seed):
        """Get model and backward model (from random_injection.py)"""
        if m_name == 'resnet18' or m_name == 'resnet18_sgd':
            model = resnet_18(seed, m_name)
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            back_model = backward_resnet_18(m_name)
        elif m_name == 'resnet18_nobn':
            model = resnet_18_nobn(seed)
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            back_model = backward_resnet_18_nobn()
        elif m_name == 'effnet':
            model = efficientnet.efficient_net_b0(seed)
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            back_model = backward_efficientnet.backward_efficient_net_b0()
        elif m_name == 'densenet':
            model = densenet.densenet_121(seed)
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            back_model = backward_densenet.backward_densenet_121()
        elif m_name == 'nfnet':
            model = nf_resnet.NF_ResNet(num_classes=10, seed=seed, alpha=1, stochdepth_rate=0)
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            back_model = backward_nf_resnet.BackwardNF_ResNet(num_classes=10, alpha=1, stochdepth_rate=0)
        else:
            raise ValueError(f"Unknown model: {m_name}")
        
        return model, back_model
    
    def create_optimizer(self, optimizer_name: str, model_name: str, learning_rate: float):
        """Create optimizer with appropriate learning rate schedule"""
        # Setup learning rate schedule based on model type (from random_injection.py)
        if 'sgd' in model_name:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=2000,
                end_learning_rate=0.001
            )
        elif 'effnet' in model_name:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=2000,
                end_learning_rate=0.0005
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=5000,
                end_learning_rate=0.0001
            )
        
        # SLOWER DECAY FACTORS FOR SLOWDEGRADE
        # Paper suggests 0.999 or 0.9999 for sustained fault effects
        ADAM_BETA1 = 0.999   # Default is 0.9 - this is 10x slower decay
        ADAM_BETA2 = 0.9999  # Default is 0.999 - this is 10x slower decay
        
        # Create optimizer based on name
        optimizer_map = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'sgd_vanilla': tf.keras.optimizers.SGD,  # Same class, different config
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamw': tf.keras.optimizers.Adam,  # Will try to use AdamW if available
            'adagrad': tf.keras.optimizers.Adagrad,
            'adadelta': tf.keras.optimizers.Adadelta,
            'nadam': tf.keras.optimizers.Nadam,
        }
        
        # Special handling for AdamW
        if optimizer_name.lower() == 'adamw':
            try:
                if hasattr(tf.keras.optimizers, 'AdamW'):
                    return tf.keras.optimizers.AdamW(
                        learning_rate=lr_schedule,
                        beta_1=ADAM_BETA1,
                        beta_2=ADAM_BETA2
                    )
                elif hasattr(tf.keras.optimizers, 'experimental') and hasattr(tf.keras.optimizers.experimental, 'AdamW'):
                    return tf.keras.optimizers.experimental.AdamW(
                        learning_rate=lr_schedule,
                        beta_1=ADAM_BETA1,
                        beta_2=ADAM_BETA2
                    )
            except:
                pass
            # Fallback to Adam with slow decay
            return tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=ADAM_BETA1,
                beta_2=ADAM_BETA2
            )
        
        optimizer_class = optimizer_map.get(optimizer_name.lower(), tf.keras.optimizers.Adam)
        
        # Create optimizer with appropriate parameters
        if optimizer_name.lower() == 'adam':  # EXPLICIT ADAM HANDLING
            return optimizer_class(
                learning_rate=lr_schedule,
                beta_1=ADAM_BETA1,   # Slow decay for SlowDegrade (default: 0.9)
                beta_2=ADAM_BETA2    # Very slow decay (default: 0.999)
            )
        elif optimizer_name.lower() == 'nadam':  # Nadam also uses beta values
            return optimizer_class(
                learning_rate=lr_schedule,
                beta_1=ADAM_BETA1,
                beta_2=ADAM_BETA2
            )
        elif optimizer_name.lower() == 'sgd':
            # For SlowDegrade experiments, use higher momentum to sustain faults
            return optimizer_class(learning_rate=lr_schedule, momentum=0.999)
        elif optimizer_name.lower() == 'sgd_vanilla':
            # Vanilla SGD with NO momentum (stateless)
            return optimizer_class(learning_rate=lr_schedule, momentum=0.0)
        elif optimizer_name.lower() == 'rmsprop':
            # RMSprop uses 'rho' as its decay factor - increase for slower decay
            return optimizer_class(learning_rate=lr_schedule, rho=0.999)
        elif optimizer_name.lower() == 'adadelta':
            # Increase rho for much slower decay
            return optimizer_class(learning_rate=lr_schedule, rho=0.9999)
        else:
            # Default fallback - shouldn't reach here for Adam anymore
            return optimizer_class(learning_rate=lr_schedule)
    
    def clear_optimizer_states(self, optimizer):
        """Clear/reset all internal state variables of the optimizer."""
        # For optimizers with slots (Adam, RMSprop, SGD with momentum)
        if hasattr(optimizer, '_slots'):
            # Clear all slot variables
            optimizer._slots = {}
        
        # Reset iteration counter if exists
        if hasattr(optimizer, 'iterations'):
            optimizer.iterations.assign(0)
        
        # For TensorFlow 2.x optimizers
        if hasattr(optimizer, 'variables'):
            # Get all optimizer variables (excluding iteration counter)
            for var in optimizer.variables():
                if var.dtype in [tf.float32, tf.float16, tf.float64]:
                    # Reset to zeros
                    var.assign(tf.zeros_like(var))
        
        return optimizer
    
    def train_with_injection(self, optimizer_name: str, injection_params: Dict, 
                            experiment_dir: str, stored_injection: Dict = None) -> Dict:
        """
        Train a single optimizer with injection.
        Adapted from random_injection.py's run_training_simulation.
        """
        print(f"\n{'='*60}")
        print(f"Training {optimizer_name} optimizer")
        print(f"{'='*60}")
        
        # Set seeds
        seed = injection_params['seed']
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Get datasets
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        
        # Get model and backward model
        model, back_model = self.get_model(injection_params['model'], seed)
        
        # Setup optimizer
        model.optimizer = self.create_optimizer(
            optimizer_name, 
            injection_params['model'],
            injection_params['learning_rate']
        )
        
        # Setup metrics
        train_loss = tf.keras.metrics.Mean(name=f'train_loss_{optimizer_name}')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name=f'train_accuracy_{optimizer_name}')
        
        # Create log file for this optimizer
        log_path = os.path.join(experiment_dir, f"{optimizer_name}_training.log")
        train_recorder = open(log_path, 'w')
        
        def record(text):
            train_recorder.write(text)
            train_recorder.flush()
            print(text.strip())
        
        # Training functions (directly from random_injection.py)
        @tf.function
        def train_step(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                outputs, _, _, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            
            tvars = model.trainable_variables
            gradients = tape.gradient(avg_loss, tvars)
            model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))
            
            # Calculate single-step accuracy
            correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.cast(labels, tf.int64))
            step_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss, step_accuracy
        
        @tf.function
        def fwrd_inj_train_step1(iter_inputs, inj_layer):
            images, labels = iter_inputs
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
            predictions = outputs['logits']
            return l_inputs[inj_layer], l_kernels[inj_layer], l_outputs[inj_layer]
        
        def fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag):
            with tf.GradientTape() as tape:
                images, labels = iter_inputs
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, 
                                                               inject=inj_flag, inj_args=inj_args)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            
            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs, l_kernels, 
                                                        inject=False, inj_args=None)
            
            gradients = manual_gradients + golden_gradients[golden_grad_idx[injection_params['model']]:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            # Calculate single-step accuracy
            correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.cast(labels, tf.int64))
            step_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)).numpy()
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss, l_outputs, step_accuracy
        
        def bkwd_inj_train_step1(iter_inputs, inj_layer):
            images, labels = iter_inputs
            with tf.GradientTape() as tape:
                outputs, l_inputs, l_kernels, _ = model(images, training=True, inject=False)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            man_grad_start = tape.gradient(avg_loss, grad_start)
            _, bkwd_inputs, bkwd_kernels, bkwd_outputs = back_model.call(
                man_grad_start, l_inputs, l_kernels, inject=False, inj_args=None
            )
            
            # Debug: Check what keys are available
            if inj_layer not in bkwd_inputs:
                print(f"Warning: Layer '{inj_layer}' not found in backward model outputs")
                print(f"Available layers: {list(bkwd_inputs.keys())[:10]}...")  # Show first 10 keys
                # Try to find a matching layer or use a default
                if len(bkwd_inputs) > 0:
                    # Use the first available layer as fallback
                    fallback_layer = list(bkwd_inputs.keys())[0]
                    print(f"Using fallback layer: {fallback_layer}")
                    return bkwd_inputs[fallback_layer], bkwd_kernels[fallback_layer], bkwd_outputs[fallback_layer]
                else:
                    raise KeyError(f"No layers available in backward model outputs")
            
            return bkwd_inputs[inj_layer], bkwd_kernels[inj_layer], bkwd_outputs[inj_layer]
        
        def bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag):
            print("bkwd_inj_train_step2")
            images, labels = iter_inputs
            with tf.GradientTape() as tape:
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs, l_kernels, 
                                                        inject=inj_flag, inj_args=inj_args)
            
            gradients = manual_gradients + golden_gradients[golden_grad_idx[injection_params['model']]:]
            
            # IMPORTANT: Backward injection behavior
            # Apply corrupted gradients to BOTH optimizer state AND model weights
            # This allows the full SlowDegrade cascade effect to occur
            
            # Apply gradients - this updates BOTH optimizer state AND model weights
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            # NOTE: Weight restoration has been removed to allow corrupted gradients
            # to directly affect weights, enabling the SlowDegrade phenomenon
            
            # Calculate single-step accuracy
            correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.cast(labels, tf.int64))
            step_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)).numpy()
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss, l_outputs, step_accuracy
        
        # Training loop parameters
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        target_epoch = injection_params['target_epoch']
        target_step = injection_params['target_step']
        injection_global_step = target_epoch * steps_per_epoch + target_step
        
        record(f"Target injection: epoch {target_epoch}, step {target_step}\n")
        record(f"Global injection step: {injection_global_step}\n")
        record(f"Steps per epoch: {steps_per_epoch}\n")
        
        # History tracking - simplified to match gold standard
        history = {
            'global_steps': [],  # Track global step number
            'epochs': [],        # Track epoch number
            'steps': [],         # Track step within epoch
            'accuracy': [],      # Per-step accuracy (matches gold standard)
            'loss': [],          # Per-step loss (matches gold standard)
            'optimizer_state_magnitudes': [],  # Track optimizer state magnitudes
            'detailed_optimizer_states': [],  # Track detailed per-variable, per-slot states
            'momentum_values': []  # Track actual m,v values over time for Adam
        }
        
        def get_optimizer_state_magnitude():
            """Calculate total magnitude of optimizer state variables."""
            total_magnitude = 0.0
            var_count = 0
            
            # Get optimizer variables (momentum, variance, etc.)
            for var in model.optimizer.variables():
                # Skip integer variables (like step counters)
                if var.dtype in [tf.int32, tf.int64]:
                    continue
                    
                # Convert to float32 if needed and calculate norm
                if var.dtype != tf.float32:
                    var_float = tf.cast(var, tf.float32)
                else:
                    var_float = var
                    
                magnitude = tf.norm(var_float).numpy()
                
                # Check for inf/nan and skip them
                if np.isfinite(magnitude):
                    total_magnitude += magnitude
                    var_count += 1
                else:
                    print(f"    Warning: Found inf/nan in optimizer variable at epoch {epoch}, step {step}")
            
            # Debug logging
            if global_step % 50 == 0 and var_count > 0:
                print(f"  Optimizer state magnitude at epoch {epoch}, step {step}: {total_magnitude:.6f} from {var_count} variables")
            
            return total_magnitude if np.isfinite(total_magnitude) else 0.0
        
        def get_detailed_optimizer_state():
            """Get detailed optimizer state information per variable and slot type."""
            detailed_state = {}
            slots_found = 0
            
            # Dynamically discover slot names by trying common ones
            possible_slot_names = [
                'momentum', 'accumulator', 'rms', 'm', 'v', 
                'velocity', 'vhat', 'delta_accumulator',
                'linear', 'gradient_accumulator'
            ]
            
            # For standard Keras optimizers, we need to access slots
            if hasattr(model.optimizer, 'get_slot'):
                # Get all trainable variables
                for var_idx, var in enumerate(model.trainable_variables[:10]):  # Limit to first 10 vars for efficiency
                    var_name = var.name.replace(':0', '')  # Clean variable name
                    # Shorten long variable names for readability
                    if len(var_name) > 50:
                        var_name = f"var_{var_idx}_{var_name[:20]}...{var_name[-20:]}"
                    detailed_state[var_name] = {}
                    
                    # Try each possible slot name
                    for slot_name in possible_slot_names:
                        try:
                            slot = model.optimizer.get_slot(var, slot_name)
                            if slot is not None:
                                # Calculate norm of the slot variable
                                if slot.dtype != tf.float32:
                                    slot_float = tf.cast(slot, tf.float32)
                                else:
                                    slot_float = slot
                                magnitude = tf.norm(slot_float).numpy()
                                # Only store finite values
                                if np.isfinite(magnitude):
                                    detailed_state[var_name][slot_name] = float(magnitude)
                                    slots_found += 1
                                else:
                                    print(f"      Warning: Found inf/nan in {slot_name} for {var_name[:30]}")
                        except Exception as e:
                            # Slot doesn't exist for this optimizer
                            pass
            
            # Also track iteration count
            for var in model.optimizer.variables():
                if var.dtype in [tf.int32, tf.int64]:
                    if 'iteration' in var.name.lower() or 'step' in var.name.lower():
                        detailed_state['_iteration'] = int(var.numpy())
                        break
            
            # Debug logging periodically
            if global_step % 100 == 0:
                print(f"  Detailed state at step {global_step}: {slots_found} slots found for {optimizer_name}")
                if slots_found == 0:
                    # Try to debug why no slots were found
                    print(f"    Debug: optimizer type = {type(model.optimizer).__name__}")
                    print(f"    Debug: has get_slot method = {hasattr(model.optimizer, 'get_slot')}")
                    # Check if it's vanilla SGD with no momentum
                    if 'sgd' in optimizer_name.lower() and hasattr(model.optimizer, 'momentum'):
                        print(f"    Debug: SGD momentum = {model.optimizer.momentum}")
                if detailed_state and slots_found > 0:
                    # Show first variable's slots as example
                    for var_name, slots in detailed_state.items():
                        if var_name != '_iteration' and slots:
                            print(f"    Example - {var_name[:30]}...: {list(slots.keys())}")
                            break
            
            return detailed_state
        
        def log_adam_momentum_values(show_all=False):
            """Log actual m and v values for Adam optimizer."""
            if 'adam' not in optimizer_name.lower():
                return {}
            
            # Check if optimizer has been initialized (has variables)
            if not model.optimizer.variables():
                return {}  # Optimizer not yet initialized
            
            momentum_values = {'m': {}, 'v': {}}
            
            # TensorFlow 2.x stores optimizer variables differently
            # They are stored in optimizer.variables() with specific naming patterns
            opt_vars = model.optimizer.variables()
            
            # Debug: log optimizer variable names once to understand structure
            if global_step == injection_global_step:
                print(f"\n  DEBUG: Adam optimizer has {len(opt_vars)} variables")
                for i, opt_var in enumerate(opt_vars[:10]):  # Show first 10
                    print(f"    Var {i}: {opt_var.name}, shape: {opt_var.shape}, dtype: {opt_var.dtype}")
            
            # Track max values across ALL variables
            max_m_overall = 0
            max_v_overall = 0
            max_m_var_name = None
            max_v_var_name = None
            
            # Direct approach: parse optimizer variable names
            for opt_var in opt_vars:
                opt_var_name = opt_var.name
                
                # Skip iteration counter
                if 'iteration' in opt_var_name.lower():
                    continue
                    
                # Extract variable type from name
                if '/m/' in opt_var_name:
                    # This is a momentum variable
                    # Extract the base variable name
                    parts = opt_var_name.split('/')
                    # Find the part after 'm/'
                    var_key = '/'.join(parts[2:]).replace(':0', '')  # Skip 'Adam/m/' prefix
                    if len(var_key) > 30:
                        var_key = var_key[:15] + '...' + var_key[-10:]
                    
                    m_values = opt_var.numpy().flatten()
                    
                    # Check for inf/nan
                    has_inf = np.any(np.isinf(m_values))
                    has_nan = np.any(np.isnan(m_values))
                    
                    if has_inf or has_nan:
                        max_abs_m = float('inf') if has_inf else float('nan')
                    else:
                        max_abs_m = float(np.max(np.abs(m_values))) if len(m_values) > 0 else 0
                    
                    # Track overall maximum (including inf/nan)
                    if not np.isfinite(max_m_overall) or max_abs_m > max_m_overall:
                        max_m_overall = max_abs_m
                        max_m_var_name = var_key
                    
                    # Compute statistics, handling inf/nan gracefully
                    mean_val = float(np.mean(np.abs(m_values[np.isfinite(m_values)]))) if np.any(np.isfinite(m_values)) else float('nan')
                    min_val = float(np.min(m_values[np.isfinite(m_values)])) if np.any(np.isfinite(m_values)) else float('nan')
                    max_signed = float(np.max(m_values[np.isfinite(m_values)])) if np.any(np.isfinite(m_values)) else float('nan')
                    
                    momentum_values['m'][var_key] = {
                        'mean': mean_val,
                        'max': max_abs_m,
                        'min': min_val,
                        'max_signed': max_signed,
                        'first_5': [f'{v:.2e}' if np.isfinite(v) else 'inf' if np.isinf(v) else 'nan' 
                                   for v in m_values[:5]] if len(m_values) >= 5 else 
                                   [f'{v:.2e}' if np.isfinite(v) else 'inf' if np.isinf(v) else 'nan' 
                                   for v in m_values],
                        'has_inf': has_inf,
                        'has_nan': has_nan
                    }
                    
                elif '/v/' in opt_var_name:
                    # This is a variance variable
                    parts = opt_var_name.split('/')
                    var_key = '/'.join(parts[2:]).replace(':0', '')  # Skip 'Adam/v/' prefix
                    if len(var_key) > 30:
                        var_key = var_key[:15] + '...' + var_key[-10:]
                    
                    v_values = opt_var.numpy().flatten()
                    
                    # Check for inf/nan
                    has_inf = np.any(np.isinf(v_values))
                    has_nan = np.any(np.isnan(v_values))
                    
                    if has_inf or has_nan:
                        max_v = float('inf') if has_inf else float('nan')
                    else:
                        max_v = float(np.max(v_values)) if len(v_values) > 0 else 0
                    
                    # Track overall maximum (including inf/nan)
                    if not np.isfinite(max_v_overall) or max_v > max_v_overall:
                        max_v_overall = max_v
                        max_v_var_name = var_key
                    
                    # Compute statistics, handling inf/nan gracefully
                    mean_val = float(np.mean(v_values[np.isfinite(v_values)])) if np.any(np.isfinite(v_values)) else float('nan')
                    
                    momentum_values['v'][var_key] = {
                        'mean': mean_val,
                        'max': max_v,
                        'first_5': [f'{v:.2e}' if np.isfinite(v) else 'inf' if np.isinf(v) else 'nan' 
                                   for v in v_values[:5]] if len(v_values) >= 5 else 
                                   [f'{v:.2e}' if np.isfinite(v) else 'inf' if np.isinf(v) else 'nan' 
                                   for v in v_values],
                        'has_inf': has_inf,
                        'has_nan': has_nan
                    }
            
            # Add overall max info
            momentum_values['_max_info'] = {
                'max_m_overall': max_m_overall,
                'max_m_var': max_m_var_name,
                'max_v_overall': max_v_overall,
                'max_v_var': max_v_var_name
            }
            
            # If not showing all, filter to important variables + the ones with max values
            if not show_all:
                # For example, look for kernel, bias, gamma, beta
                important_keys = ['kernel', 'bias', 'gamma', 'beta']
                filtered_m = {}
                filtered_v = {}
                
                # Always include the variable with the maximum value
                if max_m_var_name and max_m_var_name in momentum_values['m']:
                    filtered_m['MAX: ' + max_m_var_name] = momentum_values['m'][max_m_var_name]
                if max_v_var_name and max_v_var_name in momentum_values['v']:
                    filtered_v['MAX: ' + max_v_var_name] = momentum_values['v'][max_v_var_name]
                
                for key in momentum_values['m']:
                    for important in important_keys:
                        if important in key.lower() and key != max_m_var_name:
                            filtered_m[important] = momentum_values['m'][key]
                            break
                
                for key in momentum_values['v']:
                    for important in important_keys:
                        if important in key.lower() and key != max_v_var_name:
                            filtered_v[important] = momentum_values['v'][key]
                            break
                
                # If we found filtered results, use them for cleaner output
                if filtered_m:
                    momentum_values['m'] = filtered_m
                if filtered_v:
                    momentum_values['v'] = filtered_v
            
            return momentum_values
        
        # Initialize training state
        injection_performed = False
        post_injection_accuracy = None
        post_injection_loss = None
        nan_weights = 0
        inf_weights = 0
        
        # Recovery tracking variables
        steps_to_accuracy_increase = None  # Steps until accuracy starts increasing
        steps_to_full_recovery = None  # Steps until accuracy fully recovers
        pre_injection_accuracy = None  # Store pre-injection accuracy for recovery comparison
        accuracy_started_increasing = False
        lowest_post_injection_accuracy = None
        recovery_search_limit = 1000  # Give up after 1000 steps
        actual_inj_pos = None
        actual_inj_values = None
        
        # Calculate total epochs needed
        start_epoch = 0
        steps_after_injection_global = injection_global_step + self.steps_after_injection
        total_epochs_needed = math.ceil(steps_after_injection_global / steps_per_epoch) + 1
        
        # Initialize global step counter
        global_step = 0
        early_terminate = False
        
        # Main training loop - structured by epochs and steps (matching gold standard)
        for epoch in range(start_epoch, total_epochs_needed):
            if early_terminate:
                break
                
            # Create iterator for this epoch
            train_iterator = iter(train_dataset)
            
            # Loop through steps in this epoch
            for step in range(steps_per_epoch):
                # Check if we've trained enough steps after injection
                if injection_performed and global_step > injection_global_step + self.steps_after_injection:
                    early_terminate = True
                    break
                
                # Check for max global steps termination
                if injection_params['max_global_steps'] and global_step >= injection_params['max_global_steps']:
                    record(f"Early termination at step {global_step}\n")
                    early_terminate = True
                    break
                
                # CRITICAL: Reset metrics at each step (matching gold standard lines 270-271)
                train_loss.reset_states()
                train_accuracy.reset_states()
            
                if global_step == injection_global_step:
                    # Store pre-injection accuracy
                    if history['accuracy']:
                        pre_injection_accuracy = history['accuracy'][-1]
                    else:
                        pre_injection_accuracy = 0.1  # Default if no history
                
                    # Log momentum values BEFORE injection
                    pre_injection_momentum = log_adam_momentum_values()
                    if pre_injection_momentum:
                        record(f"\n📊 Pre-injection Adam momentum values:\n")
                        for var_name, m_data in pre_injection_momentum.get('m', {}).items():
                            record(f"  {var_name}: mean(|m|)={m_data['mean']:.2e}, max(|m|)={m_data['max']:.2e}, max_signed={m_data['max_signed']:.2e}\n")
                            if abs(m_data['max']) > 1e-6:  # Only show values if non-trivial
                                record(f"    First 5 values: {m_data['first_5']}\n")
                    
                    # Perform injection
                    record(f"\n🎯 Performing injection at epoch {epoch}, step {step} (global step {global_step})\n")
                    record(f"Pre-injection accuracy: {pre_injection_accuracy:.4f}\n")
                    
                    try:
                        iter_inputs = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_dataset)
                        iter_inputs = next(train_iterator)
                    
                    inj_layer = injection_params['target_layer']
                    
                    # Create proper injection config object for inject_utils
                    class InjectionConfig:
                        def __init__(self, params):
                            self.inj_pos = params.get('inj_pos', [])
                            self.inj_values = params.get('inj_values', [])
                            self.target_worker = -1
                    
                    inj_config = InjectionConfig(injection_params)
                    
                    # Get layer outputs based on stage
                    if 'fwrd' in injection_params['stage']:
                        record("Forward injection\n")
                        l_inputs, l_kernels, l_outputs = fwrd_inj_train_step1(iter_inputs, inj_layer)
                    else:
                        record("Backward injection\n")
                        l_inputs, l_kernels, l_outputs = bkwd_inj_train_step1(iter_inputs, inj_layer)
                
                    # Generate or reuse injection arguments
                    if stored_injection and stored_injection['inj_pos'] is not None:
                        # Reuse the exact same injection from the first optimizer
                        record("Using stored injection from first optimizer\n")
                        record(f"Stored positions: {stored_injection['inj_pos']}\n")
                        record(f"Stored values: {stored_injection['inj_values']}\n")
                        
                        inj_args, inj_flag = get_replay_args(
                            InjType[injection_params['fmodel']], inj_config, None, inj_layer,
                            l_inputs, l_kernels, l_outputs, train_recorder,
                            inj_pos=stored_injection['inj_pos'], 
                            inj_values=stored_injection['inj_values']
                        )
                        
                        # Store for return (redundant but for consistency)
                        actual_inj_pos = stored_injection['inj_pos']
                        actual_inj_values = stored_injection['inj_values']
                        
                    elif injection_params['min_val'] is not None and injection_params['max_val'] is not None:
                        record("Generating NEW random injection with min/max range\n")
                        inj_args, inj_flag = get_inj_args_with_random_range(
                            InjType[injection_params['fmodel']], None, inj_layer,
                            l_inputs, l_kernels, l_outputs, train_recorder,
                            inj_config, injection_params['min_val'], injection_params['max_val']
                        )
                        
                        # Extract the actual positions and values for storage
                        actual_inj_pos = inj_config.inj_pos
                        actual_inj_values = inj_config.inj_values
                        
                    elif injection_params['inj_pos'] and injection_params['inj_values']:
                        record("Using specific injection position and value\n")
                        inj_args, inj_flag = get_replay_args(
                            InjType[injection_params['fmodel']], inj_config, None, inj_layer,
                            l_inputs, l_kernels, l_outputs, train_recorder,
                            inj_pos=injection_params['inj_pos'], 
                            inj_values=injection_params['inj_values']
                        )
                        
                        actual_inj_pos = injection_params['inj_pos']
                        actual_inj_values = injection_params['inj_values']
                        
                    else:
                        record("Using random injection\n")
                        inj_args, inj_flag = get_inj_args(
                            InjType[injection_params['fmodel']], None, inj_layer,
                            l_inputs, l_kernels, l_outputs, train_recorder, inj_config
                        )
                        
                        actual_inj_pos = inj_config.inj_pos
                        actual_inj_values = inj_config.inj_values
                    
                    # Perform injection
                    if 'fwrd' in injection_params['stage']:
                        losses, injected_layer_outputs, injection_step_accuracy = fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                    else:
                        losses, injected_layer_outputs, injection_step_accuracy = bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                    
                    # Check for NaN/Inf after injection
                    for var in model.trainable_variables:
                        nan_weights += tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)).numpy()
                        inf_weights += tf.reduce_sum(tf.cast(tf.math.is_inf(var), tf.int32)).numpy()
                    
                    # Clear optimizer states if NaN/Inf detected and clearing is enabled
                    if self.clear_optimizer_state_on_nan and (nan_weights > 0 or inf_weights > 0):
                        record(f"🔧 Clearing optimizer states due to NaN/Inf detection\n")
                        self.clear_optimizer_states(model.optimizer)
                        record(f"   Optimizer states cleared for {optimizer_name}\n")
                    
                    # Record metrics (per-step, matching gold standard)
                    post_injection_accuracy = float(train_accuracy.result())  # This is single-step since we reset
                    post_injection_loss = float(train_loss.result())
                    lowest_post_injection_accuracy = post_injection_accuracy
                    
                    record(f"Post-injection: accuracy={post_injection_accuracy:.4f}, "
                          f"loss={post_injection_loss:.4f}, NaN={nan_weights}, Inf={inf_weights}\n")
                    
                    # Log momentum values AFTER injection
                    post_injection_momentum = log_adam_momentum_values()
                    if post_injection_momentum:
                        record(f"\n📊 Post-injection Adam momentum values:\n")
                        
                        # First show the OVERALL maximum across ALL variables
                        max_info = post_injection_momentum.get('_max_info', {})
                        if max_info:
                            max_m_val = max_info.get('max_m_overall', 0)
                            max_m_var = max_info.get('max_m_var', 'unknown')
                            max_v_val = max_info.get('max_v_overall', 0)
                            max_v_var = max_info.get('max_v_var', 'unknown')
                            
                            record(f"\n🔥 MAXIMUM VALUES ACROSS ALL VARIABLES:\n")
                            record(f"  Max |m|: {max_m_val:.2e} in '{max_m_var}'\n")
                            record(f"  Max v: {max_v_val:.2e} in '{max_v_var}'\n")
                            
                            # Check if max momentum is in SlowDegrade range
                            if 2.6e8 <= max_m_val <= 1.1e19:
                                record(f"  ✅ MAX MOMENTUM IN SLOWDEGRADE RANGE! ({max_m_val:.2e})\n")
                            elif max_m_val > 1.1e19:
                                record(f"  ⚠️ MAX MOMENTUM TOO LARGE (>{1.1e19}, actual: {max_m_val:.2e})\n")
                            elif not np.isfinite(max_m_val):
                                record(f"  💥 MAX MOMENTUM IS INF/NAN!\n")
                            else:
                                record(f"  ❌ MAX MOMENTUM BELOW RANGE (<2.6e8, actual: {max_m_val:.2e})\n")
                        
                        # Then show filtered important variables
                        record(f"\nFiltered important variables:\n")
                        for var_name, m_data in post_injection_momentum.get('m', {}).items():
                            if var_name.startswith('MAX: '):
                                record(f"  {var_name}: mean(|m|)={m_data['mean']:.2e}, max(|m|)={m_data['max']:.2e}\n")
                            else:
                                record(f"  {var_name}: mean(|m|)={m_data['mean']:.2e}, max(|m|)={m_data['max']:.2e}, max_signed={m_data.get('max_signed', 0):.2e}\n")
                            
                            # Show first 5 values only for non-MAX entries to keep it concise
                            if not var_name.startswith('MAX: ') and abs(m_data['max']) > 1e-6:
                                record(f"    First 5 values: {m_data.get('first_5', [])}\n")
                    
                    injection_performed = True
                    
                else:
                    # Normal training step
                    try:
                        batch_data = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_dataset)
                        batch_data = next(train_iterator)
                    
                    losses, step_acc = train_step(batch_data)
            
                # Record metrics - all metrics are per-step now (matching gold standard)
                current_accuracy = float(train_accuracy.result())  # Single-step accuracy since we reset
                current_loss = float(train_loss.result())  # Single-step loss since we reset
                
                history['global_steps'].append(global_step)
                history['epochs'].append(epoch)
                history['steps'].append(step)
                history['accuracy'].append(current_accuracy)
                history['loss'].append(current_loss)
            
                # Track recovery after injection
                if injection_performed and global_step > injection_global_step:
                    steps_since_injection = global_step - injection_global_step
                    
                    # Log momentum decay every 10 steps to track the 0.9^k pattern
                    if 'adam' in optimizer_name.lower() and steps_since_injection % 10 == 0 and steps_since_injection <= 100:
                        current_momentum = log_adam_momentum_values()
                        if current_momentum and 'post_injection_momentum' in locals():
                            record(f"\n📉 Momentum decay at t+{steps_since_injection}:\n")
                            for var_name, m_data in current_momentum.get('m', {}).items():
                                if var_name in post_injection_momentum.get('m', {}):
                                    initial_max = post_injection_momentum['m'][var_name]['max']
                                    current_max = m_data['max']
                                    
                                    # Calculate expected decay (0.999^steps for our SlowDegrade beta1)
                                    expected = initial_max * (0.999 ** steps_since_injection)
                                    
                                    record(f"  {var_name}:\n")
                                    record(f"    Current max(|m|): {current_max:.2e}\n")
                                    record(f"    Expected (0.999^{steps_since_injection}): {expected:.2e}\n")
                                    
                                    # Check if actual decay matches theoretical
                                    if initial_max > 1e6:  # Only check if initial was substantial
                                        ratio = current_max / expected if expected > 0 else float('inf')
                                        if 0.5 <= ratio <= 2.0:
                                            record(f"    ✓ Decay following ~0.999^k pattern (ratio: {ratio:.2f})\n")
                                        else:
                                            record(f"    ✗ Decay NOT following 0.999^k pattern (ratio: {ratio:.2f})\n")
                    
                    # Check if we're within the recovery search limit
                    if steps_since_injection <= recovery_search_limit:
                        # Track when accuracy starts increasing
                        if steps_to_accuracy_increase is None:
                            if current_accuracy > lowest_post_injection_accuracy:
                                steps_to_accuracy_increase = steps_since_injection
                                accuracy_started_increasing = True
                                record(f"📈 Accuracy started increasing after {steps_to_accuracy_increase} steps\n")
                            else:
                                # Update lowest post-injection accuracy
                                lowest_post_injection_accuracy = min(lowest_post_injection_accuracy, current_accuracy)
                        
                        # Track when accuracy fully recovers (needs to reach 1.25x pre-injection accuracy)
                        if steps_to_full_recovery is None and pre_injection_accuracy is not None:
                            recovery_threshold = pre_injection_accuracy * 1.25
                            if current_accuracy >= recovery_threshold:
                                steps_to_full_recovery = steps_since_injection
                                record(f"✅ Full recovery achieved after {steps_to_full_recovery} steps\n")
                                record(f"   Pre-injection: {pre_injection_accuracy:.4f}, Current: {current_accuracy:.4f}\n")
                                record(f"   Recovery threshold (1.25x): {recovery_threshold:.4f}\n")
                                record(f"📊 Stopping early due to full recovery (saved {self.steps_after_injection - steps_since_injection} steps)\n")
                                early_terminate = True  # Set flag to exit loops properly
                    
                    # Give up after 1000 steps
                    elif steps_to_full_recovery is None:
                        record(f"⚠️ Recovery tracking stopped after {recovery_search_limit} steps\n")
                        # Set to a sentinel value to indicate we gave up
                        if steps_to_full_recovery is None:
                            steps_to_full_recovery = -1  # -1 indicates recovery not achieved within limit
                
                # Track optimizer state magnitude
                if len(model.optimizer.variables()) > 0:
                    state_mag = get_optimizer_state_magnitude()
                    history['optimizer_state_magnitudes'].append(state_mag)
                    # Also track detailed state
                    detailed = get_detailed_optimizer_state()
                    history['detailed_optimizer_states'].append(detailed)
                    
                    # Store actual momentum values for Adam
                    if 'adam' in optimizer_name.lower():
                        momentum_snapshot = log_adam_momentum_values()
                        history['momentum_values'].append(momentum_snapshot)
                    else:
                        history['momentum_values'].append({})
                else:
                    # For optimizers without state (like vanilla SGD)
                    history['optimizer_state_magnitudes'].append(0.0)
                    history['detailed_optimizer_states'].append({})
                    history['momentum_values'].append({})
                
                # Progress update - log EVERY step (matching gold standard lines 292-297)
                record(f"Epoch: {epoch}/{total_epochs_needed}, step: {step}/{steps_per_epoch}, "
                      f"loss: {current_loss:.5f}, accuracy: {current_accuracy:.5f}\n")
                
                # Check for NaN termination (matching gold standard)
                if not np.isfinite(current_loss):
                    record(f"Encounter NaN! Terminate training!\n")
                    early_terminate = True
                    break
                
                # Increment global step counter
                global_step += 1
        
        # Close log file
        train_recorder.close()
        
        # Calculate recovery metrics using per-step accuracy (now the only type)
        # Find the index in history corresponding to injection
        injection_idx = None
        for idx, gs in enumerate(history['global_steps']):
            if gs == injection_global_step:
                injection_idx = idx
                break
        
        if injection_idx is not None and injection_idx > 0:
            pre_injection_acc = history['accuracy'][injection_idx - 1]
        else:
            pre_injection_acc = 0.1  # Default
        
        final_acc = history['accuracy'][-1] if history['accuracy'] else 0
        
        # Calculate degradation rate
        if injection_idx is not None and len(history['accuracy']) > injection_idx + 10:
            recovery_steps = history['global_steps'][injection_idx + 1:]
            recovery_acc = history['accuracy'][injection_idx + 1:]
            if len(recovery_steps) > 1:
                z = np.polyfit(recovery_steps, recovery_acc, 1)
                degradation_rate = float(z[0])
            else:
                degradation_rate = 0
        else:
            degradation_rate = 0
        
        return {
            'optimizer_name': optimizer_name,
            'history': history,
            'pre_injection_accuracy': pre_injection_acc,
            'post_injection_accuracy': post_injection_accuracy,
            'post_injection_loss': post_injection_loss,
            'final_accuracy': final_acc,
            'accuracy_change': final_acc - (post_injection_accuracy or 0),
            'total_recovery': final_acc - pre_injection_acc,
            'degradation_rate': degradation_rate,
            'nan_weights': nan_weights,
            'inf_weights': inf_weights,
            'injection_performed': injection_performed,
            'actual_injection_step': injection_global_step,  # Pass the actual injection step
            'injection_pos': actual_inj_pos if injection_performed else None,
            'injection_values': actual_inj_values if injection_performed else None,
            # New recovery metrics
            'steps_to_accuracy_increase': steps_to_accuracy_increase,
            'steps_to_full_recovery': steps_to_full_recovery,
            'lowest_post_injection_accuracy': lowest_post_injection_accuracy,
            'recovery_achieved': steps_to_full_recovery is not None and steps_to_full_recovery > 0
        }
    
    def run_single_experiment(self, experiment_id: int) -> Dict:
        """Run a single experiment with all optimizers sequentially."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id + 1}/{self.num_experiments}")
        print(f"{'='*80}")
        
        # Generate injection parameters for this experiment
        experiment_seed = self.seed + experiment_id
        injection_params = self.get_random_injection_params(experiment_seed)
        
        # Create experiment directory
        experiment_dir = os.path.join(self.results_base_dir, f"experiment_{experiment_id:03d}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save injection configuration
        config_path = os.path.join(experiment_dir, "injection_config.json")
        with open(config_path, 'w') as f:
            json.dump(injection_params, f, indent=2, default=str)
        
        print(f"Injection configuration:")
        for key, value in injection_params.items():
            print(f"  {key}: {value}")
        
        # Storage for the exact injection to reuse across optimizers
        stored_injection = {'inj_pos': None, 'inj_values': None}
        
        # Run each optimizer sequentially with the same injection
        optimizer_results = {}
        
        for i, optimizer_name in enumerate(self.optimizers_to_test):
            print(f"Training with {optimizer_name} optimizer")
            try:
                # Pass the stored injection for reuse (None for first optimizer)
                result = self.train_with_injection(
                    optimizer_name,
                    injection_params,
                    experiment_dir,
                    stored_injection if i > 0 else None
                )
                
                # Store the injection from the first optimizer
                if i == 0 and result.get('injection_pos') is not None:
                    stored_injection['inj_pos'] = result['injection_pos']
                    stored_injection['inj_values'] = result['injection_values']
                
                optimizer_results[optimizer_name] = result
                
            except Exception as e:
                print(f"❌ Error training {optimizer_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                optimizer_results[optimizer_name] = {
                    'error': str(e),
                    'final_accuracy': 0,
                    'accuracy_change': 0
                }
        
        # Create visualizations
        self.create_experiment_visualization(optimizer_results, injection_params, experiment_dir)
        
        # Save results
        results = {
            'experiment_id': experiment_id,
            'injection_params': injection_params,
            'optimizer_results': optimizer_results
        }
        
        results_path = os.path.join(experiment_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def create_experiment_visualization(self, optimizer_results: Dict, injection_params: Dict, 
                                       experiment_dir: str):
        """Create visualization comparing optimizer performance."""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizers_to_test)))
        
        # Get the actual injection step from the first optimizer's results
        injection_step = None
        for result in optimizer_results.values():
            if 'actual_injection_step' in result:
                injection_step = result['actual_injection_step']
                break
        
        # Fallback calculation if not found (shouldn't happen with updated code)
        if injection_step is None:
            # CIFAR-10 has 50,000 training samples
            steps_per_epoch = math.ceil(50000 / config.BATCH_SIZE)
            injection_step = injection_params['target_epoch'] * steps_per_epoch + injection_params['target_step']
        
        # Plot 1: Accuracy over time (per-step accuracy, matching gold standard)
        for i, (opt_name, result) in enumerate(optimizer_results.items()):
            if 'history' in result:
                history = result['history']
                # Use global_steps for x-axis and accuracy for y-axis
                if 'global_steps' in history and 'accuracy' in history:
                    final_acc = history['accuracy'][-1] if history['accuracy'] else 0
                    ax1.plot(history['global_steps'], history['accuracy'],
                            color=colors[i], label=f'{opt_name} (final: {final_acc:.3f})',
                            linewidth=2, alpha=0.8)
        
        ax1.axvline(x=injection_step, color='red', linestyle='--', label='Injection', alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Per-Step Accuracy')
        ax1.set_title('Per-Step Accuracy During Training and Recovery')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss over time
        for i, (opt_name, result) in enumerate(optimizer_results.items()):
            if 'history' in result:
                history = result['history']
                if 'global_steps' in history and 'loss' in history:
                    ax2.plot(history['global_steps'], history['loss'],
                            color=colors[i], label=opt_name,
                            linewidth=2, alpha=0.8)
        
        ax2.axvline(x=injection_step, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Training Loss')
        ax2.set_title('Loss During Training and Recovery')
        ax2.set_yscale('log')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Recovery comparison bar chart
        optimizer_names = list(optimizer_results.keys())
        final_accuracies = [r.get('final_accuracy', 0) for r in optimizer_results.values()]
        accuracy_changes = [r.get('accuracy_change', 0) for r in optimizer_results.values()]
        
        x = np.arange(len(optimizer_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, final_accuracies, width, label='Final Accuracy', alpha=0.8)
        bars2 = ax3.bar(x + width/2, accuracy_changes, width, label='Accuracy Change', alpha=0.8)
        
        # Color bars based on performance
        for bar, change in zip(bars2, accuracy_changes):
            if change > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax3.set_xlabel('Optimizer')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Recovery Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(optimizer_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Plot 4: Adam Momentum Values (m and v) over time
        # Plot the actual momentum values from the beginning to end
        has_momentum_data = False
        
        # Check if we have momentum data to plot
        for opt_name, result in optimizer_results.items():
            if 'adam' in opt_name.lower() and 'history' in result and 'momentum_values' in result['history']:
                momentum_history = result['history']['momentum_values']
                if momentum_history and any(mv for mv in momentum_history if mv):  # Check for non-empty entries
                    has_momentum_data = True
                    break
        
        if has_momentum_data:
            # Plot momentum values for Adam optimizers
            for opt_idx, (opt_name, result) in enumerate(optimizer_results.items()):
                if 'adam' in opt_name.lower() and 'history' in result and 'momentum_values' in result['history']:
                    momentum_history = result['history']['momentum_values']
                    global_steps = result['history']['global_steps']
                    
                    # Extract max momentum values over time for first variable
                    max_m_values = []
                    max_v_values = []
                    
                    for step_momentum in momentum_history:
                        if step_momentum and 'm' in step_momentum:
                            # Get max |m| across all variables at this step
                            max_m = 0
                            for var_data in step_momentum['m'].values():
                                if isinstance(var_data, dict) and 'max' in var_data:
                                    max_m = max(max_m, abs(var_data['max']))
                            max_m_values.append(max_m if max_m > 0 else np.nan)
                        else:
                            max_m_values.append(np.nan)
                        
                        if step_momentum and 'v' in step_momentum:
                            # Get max v across all variables at this step
                            max_v = 0
                            for var_data in step_momentum['v'].values():
                                if isinstance(var_data, dict) and 'max' in var_data:
                                    max_v = max(max_v, var_data['max'])
                            max_v_values.append(max_v if max_v > 0 else np.nan)
                        else:
                            max_v_values.append(np.nan)
                    
                    # Plot max |m| values
                    valid_steps_m = [(s, m) for s, m in zip(global_steps[:len(max_m_values)], max_m_values) if np.isfinite(m)]
                    if valid_steps_m:
                        steps_m, values_m = zip(*valid_steps_m)
                        ax4.plot(steps_m, values_m,
                                color=colors[opt_idx], label=f'{opt_name} max|m|',
                                linewidth=2, alpha=0.8, linestyle='-')
                    
                    # Plot max v values with dashed line
                    valid_steps_v = [(s, v) for s, v in zip(global_steps[:len(max_v_values)], max_v_values) if np.isfinite(v)]
                    if valid_steps_v:
                        steps_v, values_v = zip(*valid_steps_v)
                        ax4.plot(steps_v, values_v,
                                color=colors[opt_idx], label=f'{opt_name} max(v)',
                                linewidth=1.5, alpha=0.6, linestyle='--')
            
            ax4.axvline(x=injection_step, color='red', linestyle='--', label='Injection', alpha=0.7)
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Momentum Value')
            ax4.set_title('Adam Momentum Values (m and v) Throughout Training')
            ax4.set_yscale('log')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
            
            # Add horizontal lines for SlowDegrade range
            ax4.axhline(y=2.6e8, color='green', linestyle=':', alpha=0.5, label='SlowDegrade min')
            ax4.axhline(y=1.1e19, color='orange', linestyle=':', alpha=0.5, label='SlowDegrade max')
            
        else:
            # Fallback to optimizer state magnitudes if no momentum data
            print("\n=== No momentum data found, using optimizer state magnitudes ===")
            for i, (opt_name, result) in enumerate(optimizer_results.items()):
                if 'history' in result and 'optimizer_state_magnitudes' in result['history']:
                    history = result['history']
                    if history['optimizer_state_magnitudes']:
                        steps = history['global_steps'][:len(history['optimizer_state_magnitudes'])]
                        magnitudes = history['optimizer_state_magnitudes']
                        # Filter out inf/nan values for display
                        filtered_magnitudes = [m if np.isfinite(m) else np.nan for m in magnitudes]
                        ax4.plot(steps, filtered_magnitudes,
                                color=colors[i], label=opt_name,
                                linewidth=2, alpha=0.8)
            
            ax4.axvline(x=injection_step, color='red', linestyle='--', label='Injection', alpha=0.7)
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Optimizer State Magnitude')
            ax4.set_title('Optimizer State Magnitudes Over Time')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Injection details
        ax5.axis('off')
        injection_text = f"""Injection Configuration:
        
Model: {injection_params['model']}
Stage: {injection_params['stage']}
Fault Model: {injection_params['fmodel']}
Target Layer: {injection_params['target_layer']}
Target Epoch: {injection_params['target_epoch']}
Target Step: {injection_params['target_step']}
Learning Rate: {injection_params['learning_rate']}

Post-Injection Results:
"""
        for opt_name, result in optimizer_results.items():
            if 'post_injection_accuracy' in result:
                injection_text += f"\n{opt_name}:"
                injection_text += f"\n  Accuracy: {result.get('post_injection_accuracy', 0):.4f}"
                injection_text += f"\n  NaN weights: {result.get('nan_weights', 0)}"
                injection_text += f"\n  Inf weights: {result.get('inf_weights', 0)}"
                
                # Add recovery metrics
                steps_to_increase = result.get('steps_to_accuracy_increase')
                steps_to_recovery = result.get('steps_to_full_recovery')
                
                if steps_to_increase is not None:
                    injection_text += f"\n  Steps to increase: {steps_to_increase}"
                else:
                    injection_text += f"\n  Steps to increase: Not observed"
                
                if steps_to_recovery is not None:
                    if steps_to_recovery == -1:
                        injection_text += f"\n  Full recovery (1.25x): >1000 steps"
                    else:
                        injection_text += f"\n  Full recovery (1.25x): {steps_to_recovery} steps"
                else:
                    injection_text += f"\n  Full recovery (1.25x): Not achieved"
        
        ax5.text(0.1, 0.9, injection_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Plot 6: Momentum decay visualization (showing 0.9^k decay pattern)
        # Focus on Adam momentum decay after injection
        has_adam_decay_data = False
        
        for opt_name, result in optimizer_results.items():
            if 'adam' in opt_name.lower() and 'history' in result and 'momentum_values' in result['history']:
                momentum_history = result['history']['momentum_values']
                global_steps = result['history']['global_steps']
                
                # Find injection index
                inj_idx = None
                for idx, step in enumerate(global_steps):
                    if step == injection_step:
                        inj_idx = idx
                        break
                
                if inj_idx is not None and inj_idx < len(momentum_history) - 10:  # Need some post-injection data
                    has_adam_decay_data = True
                    
                    # Extract post-injection momentum values
                    post_inj_steps = global_steps[inj_idx:]
                    post_inj_momentum = momentum_history[inj_idx:]
                    
                    # Get initial (at injection) max momentum
                    initial_momentum = {}
                    if post_inj_momentum[0] and 'm' in post_inj_momentum[0]:
                        for var_name, var_data in post_inj_momentum[0]['m'].items():
                            if isinstance(var_data, dict) and 'max' in var_data:
                                initial_momentum[var_name] = abs(var_data['max'])
                    
                    # Track decay for variables with significant initial momentum
                    decay_data = []
                    for step_idx, step_momentum in enumerate(post_inj_momentum[:50]):  # First 50 steps after injection
                        if step_momentum and 'm' in step_momentum:
                            max_m = 0
                            for var_name, var_data in step_momentum['m'].items():
                                if isinstance(var_data, dict) and 'max' in var_data:
                                    if var_name in initial_momentum and initial_momentum[var_name] > 1e6:
                                        max_m = max(max_m, abs(var_data['max']))
                            if max_m > 0:
                                decay_data.append((post_inj_steps[step_idx], max_m))
                    
                    if decay_data:
                        steps_decay, values_decay = zip(*decay_data)
                        
                        # Plot actual decay
                        ax6.plot([s - injection_step for s in steps_decay], values_decay,
                                'o-', color=colors[0], label=f'{opt_name} actual decay',
                                markersize=4, linewidth=2, alpha=0.8)
                        
                        # Plot theoretical 0.999^k decay (with our SlowDegrade beta1)
                        if initial_momentum:
                            max_initial = max(initial_momentum.values())
                            theoretical_steps = range(0, min(50, len(post_inj_steps)))
                            theoretical_values = [max_initial * (0.999 ** k) for k in theoretical_steps]
                            ax6.plot(theoretical_steps, theoretical_values,
                                    '--', color='gray', label='Theoretical 0.999^k decay',
                                    linewidth=2, alpha=0.6)
                        
                        ax6.set_xlabel('Steps After Injection')
                        ax6.set_ylabel('Max |Momentum| Value')
                        ax6.set_title('Adam Momentum Decay After Injection (0.999^k pattern)')
                        ax6.set_yscale('log')
                        ax6.legend(loc='best')
                        ax6.grid(True, alpha=0.3)
                        
                        # Add horizontal lines for SlowDegrade range
                        ax6.axhline(y=2.6e8, color='green', linestyle=':', alpha=0.5, label='SlowDegrade min')
                        ax6.axhline(y=1.1e19, color='orange', linestyle=':', alpha=0.5)
                        break  # Only plot for first Adam optimizer
        
        if not has_adam_decay_data:
            ax6.axis('off')
            ax6.text(0.5, 0.5, 'No Adam momentum decay data available',
                    transform=ax6.transAxes, ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plot_path = os.path.join(experiment_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self) -> List[Dict]:
        """Run all experiments."""
        print(f"\n{'='*80}")
        print(f"Starting Sequential Optimizer Mitigation Experiments")
        print(f"{'='*80}")
        print(f"Optimizers to test: {self.optimizers_to_test}")
        print(f"Number of experiments: {self.num_experiments}")
        print(f"Max steps after injection: {self.steps_after_injection} (stops early on 1.25x recovery)")
        print(f"Results directory: {self.results_base_dir}")
        
        all_results = []
        
        for exp_id in range(self.num_experiments):
            try:
                results = self.run_single_experiment(exp_id)
                all_results.append(results)
                
            except Exception as e:
                print(f"\n❌ Error in experiment {exp_id}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate final report
        self.generate_final_report(all_results)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENTS COMPLETE")
        print(f"Successful: {len(all_results)}/{self.num_experiments}")
        print(f"Results saved to: {self.results_base_dir}")
        print(f"{'='*80}")
        
        return all_results
    
    def generate_final_report(self, results: List[Dict]):
        """Generate final report summarizing all experiments."""
        if not results:
            return
        
        report_path = os.path.join(self.results_base_dir, 'final_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Sequential Optimizer Mitigation Experiment - Final Report\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiments Completed**: {len(results)}/{self.num_experiments}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Optimizers Tested**: {', '.join(self.optimizers_to_test)}\n")
            f.write(f"- **Steps After Injection**: {self.steps_after_injection}\n\n")
            
            # Aggregate statistics
            f.write("## Aggregate Results\n\n")
            
            # Table header
            f.write("| Optimizer | Mean Final Acc | Std Final Acc | Mean Acc Change | Positive Recovery % |\n")
            f.write("|-----------|---------------|---------------|-----------------|-------------------|\n")
            
            best_mean_change = -float('inf')
            best_optimizer = None
            
            for optimizer in self.optimizers_to_test:
                final_accs = []
                acc_changes = []
                
                for result in results:
                    opt_result = result['optimizer_results'].get(optimizer, {})
                    if 'final_accuracy' in opt_result:
                        final_accs.append(opt_result['final_accuracy'])
                        acc_changes.append(opt_result.get('accuracy_change', 0))
                
                if final_accs:
                    mean_final = np.mean(final_accs)
                    std_final = np.std(final_accs)
                    mean_change = np.mean(acc_changes)
                    positive_rate = sum(1 for x in acc_changes if x > 0) / len(acc_changes) * 100
                    
                    f.write(f"| {optimizer:11} | {mean_final:13.4f} | {std_final:13.4f} | "
                           f"{mean_change:15.4f} | {positive_rate:17.1f} |\n")
                    
                    if mean_change > best_mean_change:
                        best_mean_change = mean_change
                        best_optimizer = optimizer
            
            if best_optimizer:
                f.write(f"\n### Best Performing Optimizer: **{best_optimizer}**\n\n")
                f.write(f"The {best_optimizer} optimizer showed the best average recovery with a mean accuracy "
                       f"change of {best_mean_change:.4f} after fault injection.\n\n")
        
        print(f"\nFinal report saved to: {report_path}")


def test_optimizer_mitigation(model=None, stage=None, fmodel=None,
                             target_layer=None, target_epoch=None,
                             target_step=None, learning_rate=None,
                             inj_pos=None, inj_values=None,
                             min_val=None, max_val=None,
                             seed=None, max_global_steps=None,
                             min_epoch=None, max_epoch=None,
                             optimizers_to_test=None, num_experiments=10,
                             steps_after_injection=100,
                             clear_optimizer_state_on_nan=False):
    """
    Main function to run optimizer mitigation experiments.
    Parameters match random_fault_injection() for consistency.
    """
    experiment = SequentialOptimizerExperiment(
        model=model, stage=stage, fmodel=fmodel,
        target_layer=target_layer, target_epoch=target_epoch,
        target_step=target_step, learning_rate=learning_rate,
        inj_pos=inj_pos, inj_values=inj_values,
        min_val=min_val, max_val=max_val,
        seed=seed, max_global_steps=max_global_steps,
        min_epoch=min_epoch, max_epoch=max_epoch,
        optimizers_to_test=optimizers_to_test,
        num_experiments=num_experiments,
        steps_after_injection=steps_after_injection,
        clear_optimizer_state_on_nan=clear_optimizer_state_on_nan
    )
    
    return experiment.run()


def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test optimizer mitigation with sequential training (Refactored Version)'
    )
    
    # Injection parameters
    parser.add_argument('--model', type=str, default=None,
                       choices=['resnet18', 'resnet18_sgd', 'resnet18_nobn', 'effnet', 'densenet', 'nfnet'],
                       help='Model to use (default: random)')
    parser.add_argument('--stage', type=str, default=None,
                       choices=['fwrd_inject', 'bkwd_inject'],
                       help='Injection stage (default: random)')
    parser.add_argument('--fmodel', type=str, default=None,
                       help='Fault model type (default: random)')
    parser.add_argument('--target-layer', type=str, default=None,
                       help='Target layer for injection (default: random)')
    parser.add_argument('--target-epoch', type=int, default=None,
                       help='Target epoch for injection (default: random between min-epoch and max-epoch)')
    parser.add_argument('--target-step', type=int, default=None,
                       help='Target step for injection (default: random 0-49)')
    parser.add_argument('--min-epoch', type=int, default=None,
                       help='Minimum epoch for random injection selection (default: 0)')
    parser.add_argument('--max-epoch', type=int, default=None,
                       help='Maximum epoch for random injection selection (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (default: random from [0.0001, 0.001, 0.01, 0.1])')
    parser.add_argument('--min-val', type=float, default=None,
                       help='Minimum injection value')
    parser.add_argument('--max-val', type=float, default=None,
                       help='Maximum injection value')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max-global-steps', type=int, default=None,
                       help='Maximum global steps for early stopping')
    
    # Optimizer comparison parameters
    parser.add_argument('--optimizers', type=str, nargs='+',
                       default=['adam', 'sgd', 'rmsprop', 'adamw'],
                       help='Optimizers to test')
    parser.add_argument('--num-experiments', type=int, default=10,
                       help='Number of experiments to run (default: 10)')
    parser.add_argument('--steps-after-injection', type=int, default=100,
                       help='Maximum steps to train after injection, stops early if accuracy reaches 1.25x pre-injection level (default: 100)')
    parser.add_argument('--clear-optimizer-state-on-nan', action='store_true',
                       help='Clear optimizer internal states when NaN/Inf detected after injection')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SEQUENTIAL OPTIMIZER MITIGATION EXPERIMENT - REFACTORED VERSION")
    print("="*80)
    print("\nKey features:")
    print("✓ Flexible parameter specification like random_injection.py")
    print("✓ Sequential training for fair comparison")
    print("✓ Proper tensor bounds for injection positions")
    print("✓ Support for both forward and backward injections")
    print("✓ Docker volume support for results")
    print("="*80 + "\n")
    
    # Run experiment
    results = test_optimizer_mitigation(
        model=args.model,
        stage=args.stage,
        fmodel=args.fmodel,
        target_layer=args.target_layer,
        target_epoch=args.target_epoch,
        target_step=args.target_step,
        learning_rate=args.learning_rate,
        min_val=args.min_val,
        max_val=args.max_val,
        seed=args.seed,
        max_global_steps=args.max_global_steps,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
        optimizers_to_test=args.optimizers,
        num_experiments=args.num_experiments,
        steps_after_injection=args.steps_after_injection,
        clear_optimizer_state_on_nan=args.clear_optimizer_state_on_nan
    )
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total experiments run: {len(results)}")
    print(f"{'='*80}")
    
    # Print recovery summary statistics
    if results:
        print(f"\n📊 RECOVERY METRICS SUMMARY:")
        print(f"{'-'*60}")
        
        for exp_results in results:
            if exp_results:
                print(f"\nExperiment {exp_results[0].get('experiment_id', 'N/A')}:")
                for opt_result in exp_results:
                    opt_name = opt_result.get('optimizer_name', 'Unknown')
                    steps_to_increase = opt_result.get('steps_to_accuracy_increase')
                    steps_to_recovery = opt_result.get('steps_to_full_recovery')
                    lowest_acc = opt_result.get('lowest_post_injection_accuracy')
                    
                    print(f"  {opt_name}:")
                    
                    if steps_to_increase is not None:
                        print(f"    • Steps to accuracy increase: {steps_to_increase}")
                    else:
                        print(f"    • Steps to accuracy increase: Not observed")
                    
                    if steps_to_recovery is not None:
                        if steps_to_recovery == -1:
                            print(f"    • Steps to full recovery (1.25x): >1000 (limit reached)")
                        else:
                            print(f"    • Steps to full recovery (1.25x): {steps_to_recovery}")
                    else:
                        print(f"    • Steps to full recovery (1.25x): Not achieved")
                    
                    if lowest_acc is not None:
                        print(f"    • Lowest post-injection accuracy: {lowest_acc:.4f}")
        
        # Calculate aggregate statistics
        all_increase_steps = []
        all_recovery_steps = []
        
        for exp_results in results:
            for opt_result in exp_results:
                if opt_result.get('steps_to_accuracy_increase') is not None:
                    all_increase_steps.append(opt_result['steps_to_accuracy_increase'])
                if opt_result.get('steps_to_full_recovery') is not None and opt_result['steps_to_full_recovery'] > 0:
                    all_recovery_steps.append(opt_result['steps_to_full_recovery'])
        
        print(f"\n{'-'*60}")
        print(f"AGGREGATE STATISTICS:")
        if all_increase_steps:
            print(f"  • Avg steps to accuracy increase: {np.mean(all_increase_steps):.1f}")
            print(f"  • Min steps to accuracy increase: {min(all_increase_steps)}")
            print(f"  • Max steps to accuracy increase: {max(all_increase_steps)}")
        
        if all_recovery_steps:
            print(f"  • Avg steps to full recovery (1.25x): {np.mean(all_recovery_steps):.1f}")
            print(f"  • Min steps to full recovery (1.25x): {min(all_recovery_steps)}")
            print(f"  • Max steps to full recovery (1.25x): {max(all_recovery_steps)}")
            print(f"  • Recovery rate (1.25x): {len(all_recovery_steps)}/{sum(len(exp) for exp in results)} ({100*len(all_recovery_steps)/sum(len(exp) for exp in results):.1f}%)")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
