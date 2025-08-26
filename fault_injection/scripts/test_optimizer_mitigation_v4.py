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
                    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
                elif hasattr(tf.keras.optimizers, 'experimental') and hasattr(tf.keras.optimizers.experimental, 'AdamW'):
                    return tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule)
            except:
                pass
            # Fallback to Adam
            return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        optimizer_class = optimizer_map.get(optimizer_name.lower(), tf.keras.optimizers.Adam)
        
        # Create optimizer with appropriate parameters
        if optimizer_name.lower() == 'sgd':
            return optimizer_class(learning_rate=lr_schedule, momentum=0.9)
        elif optimizer_name.lower() == 'sgd_vanilla':
            # Vanilla SGD with NO momentum (stateless)
            return optimizer_class(learning_rate=lr_schedule, momentum=0.0)
        elif optimizer_name.lower() == 'rmsprop':
            return optimizer_class(learning_rate=lr_schedule, rho=0.9)
        elif optimizer_name.lower() == 'adadelta':
            return optimizer_class(learning_rate=lr_schedule, rho=0.95)
        else:
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
            
            # Save current model weights before applying gradients
            saved_weights = [tf.Variable(w) for w in model.trainable_variables]
            
            # Apply gradients - this updates BOTH optimizer state AND model weights
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            # Restore original weights (but optimizer state remains corrupted)
            for var, saved_weight in zip(model.trainable_variables, saved_weights):
                var.assign(saved_weight)
            
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
        
        def log_adam_momentum_values():
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
            
            # Create a mapping from trainable variables to their momentum slots
            for var_idx, var in enumerate(model.trainable_variables[:5]):  # Limit to first 5 for readability
                var_name = var.name.split('/')[-1].replace(':0', '')
                if len(var_name) > 30:
                    var_name = f"var_{var_idx}_{var_name[:15]}...{var_name[-10:]}"
                
                # Find corresponding momentum variables in optimizer.variables()
                # Adam stores them with names like 'Adam/m/var_name:0' and 'Adam/v/var_name:0'
                for opt_var in opt_vars:
                    opt_var_name = opt_var.name
                    
                    # Check if this is the m (momentum) slot for our variable
                    if '/m/' in opt_var_name or '_m:' in opt_var_name:
                        # Try to match by shape and position
                        if opt_var.shape == var.shape:
                            # This is likely the momentum for this variable
                            m_values = opt_var.numpy().flatten()
                            if var_name not in momentum_values['m']:  # Only set if not already found
                                momentum_values['m'][var_name] = {
                                    'mean': float(np.mean(np.abs(m_values))),
                                    'max': float(np.max(np.abs(m_values))),
                                    'min': float(np.min(m_values)),
                                    'max_signed': float(np.max(m_values)),
                                    'first_5': m_values[:5].tolist() if len(m_values) >= 5 else m_values.tolist()
                                }
                    
                    # Check if this is the v (variance) slot for our variable
                    elif '/v/' in opt_var_name or '_v:' in opt_var_name:
                        if opt_var.shape == var.shape:
                            v_values = opt_var.numpy().flatten()
                            if var_name not in momentum_values['v']:  # Only set if not already found
                                momentum_values['v'][var_name] = {
                                    'mean': float(np.mean(v_values)),
                                    'max': float(np.max(v_values)),
                                    'first_5': v_values[:5].tolist() if len(v_values) >= 5 else v_values.tolist()
                                }
            
            # If we couldn't find momentum variables by name matching, try by index
            # Adam typically stores variables in order: [iterations, m_vars..., v_vars...]
            if not momentum_values['m'] and len(opt_vars) > 1:
                # Skip first variable (iteration counter)
                num_trainable = min(5, len(model.trainable_variables))
                
                # Assume m variables come first after iteration counter
                for i in range(num_trainable):
                    if i + 1 < len(opt_vars):
                        var = model.trainable_variables[i]
                        var_name = f"var_{i}"
                        m_var = opt_vars[i + 1]  # +1 to skip iteration counter
                        
                        if m_var.shape == var.shape and m_var.dtype in [tf.float32, tf.float16]:
                            m_values = m_var.numpy().flatten()
                            momentum_values['m'][var_name] = {
                                'mean': float(np.mean(np.abs(m_values))),
                                'max': float(np.max(np.abs(m_values))),
                                'min': float(np.min(m_values)),
                                'max_signed': float(np.max(m_values)),
                                'first_5': m_values[:5].tolist() if len(m_values) >= 5 else m_values.tolist()
                            }
                    
                    # v variables typically come after all m variables
                    v_idx = 1 + num_trainable + i  # iteration + all m vars + current v index
                    if v_idx < len(opt_vars):
                        var = model.trainable_variables[i]
                        var_name = f"var_{i}"
                        v_var = opt_vars[v_idx]
                        
                        if v_var.shape == var.shape and v_var.dtype in [tf.float32, tf.float16]:
                            v_values = v_var.numpy().flatten()
                            momentum_values['v'][var_name] = {
                                'mean': float(np.mean(v_values)),
                                'max': float(np.max(v_values)),
                                'first_5': v_values[:5].tolist() if len(v_values) >= 5 else v_values.tolist()
                            }
            
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
                                record(f"    First 5 values: {[f'{v:.2e}' for v in m_data['first_5']]}\n")
                    
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
                        for var_name, m_data in post_injection_momentum.get('m', {}).items():
                            record(f"  {var_name}: mean(|m|)={m_data['mean']:.2e}, max(|m|)={m_data['max']:.2e}, max_signed={m_data['max_signed']:.2e}\n")
                            if abs(m_data['max']) > 1e-6:  # Only show values if non-trivial
                                record(f"    First 5 values: {[f'{v:.2e}' for v in m_data['first_5']]}\n")
                            
                            # Check if momentum got corrupted to the right range for SlowDegrade
                            max_m = m_data['max']
                            if 2.6e8 <= max_m <= 1.1e19:
                                record(f"  ✅ {var_name} momentum in SlowDegrade range! ({max_m:.2e})\n")
                            elif max_m > 1.1e19:
                                record(f"  ⚠️ {var_name} momentum too large for SlowDegrade (>{1.1e19}, actual: {max_m:.2e})\n")
                            elif max_m > 1e6:  # Only warn if there's substantial momentum
                                record(f"  ❌ {var_name} momentum below SlowDegrade range (<2.6e8, actual: {max_m:.2e})\n")
                    
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
                                    
                                    # Calculate expected decay (0.9^steps for Adam default beta1)
                                    expected = initial_max * (0.9 ** steps_since_injection)
                                    
                                    record(f"  {var_name}:\n")
                                    record(f"    Current max(|m|): {current_max:.2e}\n")
                                    record(f"    Expected (0.9^{steps_since_injection}): {expected:.2e}\n")
                                    
                                    # Check if actual decay matches theoretical
                                    if initial_max > 1e6:  # Only check if initial was substantial
                                        ratio = current_max / expected if expected > 0 else float('inf')
                                        if 0.5 <= ratio <= 2.0:
                                            record(f"    ✓ Decay following ~0.9^k pattern (ratio: {ratio:.2f})\n")
                                        else:
                                            record(f"    ✗ Decay NOT following 0.9^k pattern (ratio: {ratio:.2f})\n")
                    
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
        
        # Plot 4: Individual Optimizer State Variables over time
        # NEW: Track each variable separately instead of averaging
        per_variable_data = {}  # {optimizer: {variable_name: {slot_type: [magnitudes_over_time]}}}
        all_slot_types = set()
        all_variables = set()
        
        # Debug: Print what data we have
        print("\n=== Debugging Optimizer State Data ===")
        for opt_name, result in optimizer_results.items():
            if 'history' in result:
                print(f"\n{opt_name}:")
                if 'detailed_optimizer_states' in result['history']:
                    print(f"  - Has detailed_optimizer_states: {len(result['history']['detailed_optimizer_states'])} states")
                    if result['history']['detailed_optimizer_states']:
                        first_state = result['history']['detailed_optimizer_states'][0]
                        print(f"  - First state keys: {list(first_state.keys())[:5]}...")  # Show first 5 keys
                else:
                    print(f"  - No detailed_optimizer_states")
                if 'optimizer_state_magnitudes' in result['history']:
                    print(f"  - Has optimizer_state_magnitudes: {len(result['history']['optimizer_state_magnitudes'])} values")
                    if result['history']['optimizer_state_magnitudes']:
                        first_5 = result['history']['optimizer_state_magnitudes'][:5]
                        print(f"  - First 5 magnitudes: {first_5}")
        
        # Collect per-variable data for each optimizer
        for opt_name, result in optimizer_results.items():
            if 'history' in result and 'detailed_optimizer_states' in result['history']:
                detailed_states = result['history']['detailed_optimizer_states']
                if detailed_states and len(detailed_states) > 0:
                    per_variable_data[opt_name] = {}
                    
                    # First pass: identify all variables and slot types
                    for state in detailed_states:
                        for var_name, slots in state.items():
                            if var_name != '_iteration' and isinstance(slots, dict):
                                all_variables.add(var_name)
                                all_slot_types.update(slots.keys())
                                if var_name not in per_variable_data[opt_name]:
                                    per_variable_data[opt_name][var_name] = {}
                    
                    # Second pass: collect magnitude time series for each variable-slot combination
                    for var_name in per_variable_data[opt_name]:
                        for slot_type in all_slot_types:
                            per_variable_data[opt_name][var_name][slot_type] = []
                            for state in detailed_states:
                                if var_name in state and isinstance(state[var_name], dict):
                                    if slot_type in state[var_name]:
                                        per_variable_data[opt_name][var_name][slot_type].append(state[var_name][slot_type])
                                    else:
                                        per_variable_data[opt_name][var_name][slot_type].append(0.0)
                                else:
                                    per_variable_data[opt_name][var_name][slot_type].append(0.0)
        
        # Plot individual variable-slot combinations
        line_styles = ['-', '--', '-.', ':']
        marker_styles = ['o', 's', '^', 'v', 'D', 'p', '*', 'x']
        
        # Limit number of variables to display (to avoid overcrowding)
        MAX_VARIABLES_TO_PLOT = 5  # Show only first 5 variables
        
        plot_index = 0
        has_plotted_data = False
        
        print("\n=== Plotting Individual Variable State Data ===")
        print(f"Slot types found: {all_slot_types}")
        print(f"Total variables found: {len(all_variables)}")
        print(f"Plotting first {MAX_VARIABLES_TO_PLOT} variables per optimizer")
        
        # Create a color map for better distinction
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')  # Use tab20 for more distinct colors
        
        for opt_idx, (opt_name, variables) in enumerate(per_variable_data.items()):
            result = optimizer_results[opt_name]
            if 'history' in result:
                # Sort variables by name for consistent ordering
                sorted_vars = sorted(list(variables.keys()))
                vars_to_plot = sorted_vars[:MAX_VARIABLES_TO_PLOT]  # Limit number of variables
                
                for var_idx, var_name in enumerate(vars_to_plot):
                    for slot_idx, slot_type in enumerate(sorted(all_slot_types)):
                        if slot_type in variables[var_name] and variables[var_name][slot_type]:
                            magnitudes = variables[var_name][slot_type]
                            steps = result['history']['global_steps'][:len(magnitudes)]
                            
                            # Create a shorter variable name for the label
                            short_var_name = var_name.split('/')[-1] if '/' in var_name else var_name
                            if len(short_var_name) > 20:
                                short_var_name = short_var_name[:8] + '...' + short_var_name[-8:]
                            
                            # Create label
                            label = f'{opt_name}-{short_var_name}-{slot_type}'
                            
                            # Check if there's any non-zero data
                            has_nonzero = any(m > 0 for m in magnitudes)
                            if has_nonzero:  # Only plot if there's actual data
                                # Use different colors for different optimizers and variables
                                color_idx = (opt_idx * MAX_VARIABLES_TO_PLOT + var_idx) % 20
                                line_color = cmap(color_idx / 20.0)
                                
                                # Use different line styles for different slot types
                                line_style = line_styles[slot_idx % len(line_styles)]
                                
                                print(f"  Plotting: {label[:50]}... (max={max(magnitudes):.6f})")
                                
                                ax4.plot(steps, magnitudes,
                                        color=line_color,
                                        label=label if plot_index < 10 else None,  # Limit legend entries
                                        linewidth=1.5,
                                        alpha=0.7,
                                        linestyle=line_style)
                                plot_index += 1
                                has_plotted_data = True
        
        # If no detailed states, fall back to total magnitude plot
        if not has_plotted_data:
            print("\n=== Falling back to total magnitude plot ===")
            for i, (opt_name, result) in enumerate(optimizer_results.items()):
                if 'history' in result and 'optimizer_state_magnitudes' in result['history']:
                    history = result['history']
                    if history['optimizer_state_magnitudes']:
                        steps = history['global_steps'][:len(history['optimizer_state_magnitudes'])]
                        magnitudes = history['optimizer_state_magnitudes']
                        # Filter out inf/nan values for display
                        filtered_magnitudes = [m if np.isfinite(m) else np.nan for m in magnitudes]
                        finite_mags = [m for m in magnitudes if np.isfinite(m)]
                        max_val = max(finite_mags) if finite_mags else 0
                        print(f"{opt_name}: {len(magnitudes)} magnitude values, max={max_val:.6f}, inf_count={sum(1 for m in magnitudes if np.isinf(m))}")
                        ax4.plot(steps, filtered_magnitudes,
                                color=colors[i], label=opt_name,
                                linewidth=2, alpha=0.8)
                        has_plotted_data = True
        
        # Only add injection line if we have data to plot
        if has_plotted_data:
            ax4.axvline(x=injection_step, color='red', linestyle='--', label='Injection', alpha=0.7)
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Average State Variable Magnitude')
            ax4.set_title('Individual Optimizer State Variables Over Time')
            
            # Set X-axis limits to focus on the injection region
            if injection_step is not None:
                # Show from 10 steps before injection to steps_after_injection after
                ax4.set_xlim(max(0, injection_step - 10), injection_step + self.steps_after_injection + 10)
            
            # Try to set appropriate Y-axis scale
            all_y_values = []
            for line in ax4.get_lines():
                y_data = line.get_ydata()
                if len(y_data) > 0:
                    all_y_values.extend(y_data)
            
            if all_y_values:
                # Filter out zeros, NaNs, and Infs for scale calculation
                valid_values = [v for v in all_y_values if v > 0 and np.isfinite(v)]
                if valid_values:
                    y_min = min(valid_values) * 0.5
                    y_max = max(valid_values) * 2.0
                    
                    # Ensure y_min and y_max are finite
                    if np.isfinite(y_min) and np.isfinite(y_max):
                        ax4.set_ylim(y_min, y_max)
                        # Use log scale if range is large
                        if y_max / y_min > 100:
                            try:
                                ax4.set_yscale('log')
                            except:
                                pass  # Keep linear scale if log scale fails
                    else:
                        # Fallback to auto scaling
                        print("Warning: Could not set y-axis limits due to non-finite values")
            
            ax4.legend(loc='best', fontsize=8, ncol=2)  # Multi-column legend for many lines
            ax4.grid(True, alpha=0.3)
        else:
            # If still no data, show a message
            ax4.text(0.5, 0.5, 'No optimizer state data available\n(Check if optimizers have internal states)',
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Individual Optimizer State Variables Over Time')
        
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
        
        # Plot 6: Detailed state tracking (per-variable, per-slot)
        # Analyze detailed states to find which slots and variables to plot
        slot_types_found = set()
        variable_groups = {}  # Group variables by layer type
        
        for opt_name, result in optimizer_results.items():
            if 'history' in result and 'detailed_optimizer_states' in result['history']:
                detailed_states = result['history']['detailed_optimizer_states']
                if detailed_states and len(detailed_states) > 0:
                    # Analyze first state to understand structure
                    first_state = detailed_states[0]
                    for var_name, slots in first_state.items():
                        if var_name != '_iteration' and isinstance(slots, dict):
                            # Group variables by layer type (conv, dense, etc)
                            if 'conv' in var_name.lower():
                                group = 'conv'
                            elif 'dense' in var_name.lower() or 'fc' in var_name.lower():
                                group = 'dense'
                            elif 'batch' in var_name.lower() or 'bn' in var_name.lower():
                                group = 'batchnorm'
                            else:
                                group = 'other'
                            
                            if group not in variable_groups:
                                variable_groups[group] = set()
                            variable_groups[group].add(var_name)
                            
                            # Track slot types
                            slot_types_found.update(slots.keys())
        
        # Plot aggregated state magnitudes by layer type and slot type
        if slot_types_found and variable_groups:
            # Prepare data for plotting
            plot_data = {}  # {optimizer: {slot_type: {layer_group: [magnitudes]}}}
            
            for opt_name, result in optimizer_results.items():
                if 'history' in result and 'detailed_optimizer_states' in result['history']:
                    detailed_states = result['history']['detailed_optimizer_states']
                    steps = result['history']['global_steps'][:len(detailed_states)]
                    
                    plot_data[opt_name] = {}
                    for slot_type in slot_types_found:
                        plot_data[opt_name][slot_type] = {group: [] for group in variable_groups}
                        
                        for state in detailed_states:
                            # Aggregate magnitudes by group
                            group_mags = {group: 0.0 for group in variable_groups}
                            group_counts = {group: 0 for group in variable_groups}
                            
                            for var_name, slots in state.items():
                                if var_name != '_iteration' and isinstance(slots, dict):
                                    # Determine group
                                    if 'conv' in var_name.lower():
                                        group = 'conv'
                                    elif 'dense' in var_name.lower() or 'fc' in var_name.lower():
                                        group = 'dense'
                                    elif 'batch' in var_name.lower() or 'bn' in var_name.lower():
                                        group = 'batchnorm'
                                    else:
                                        group = 'other'
                                    
                                    if slot_type in slots:
                                        group_mags[group] += slots[slot_type]
                                        group_counts[group] += 1
                            
                            # Store averaged magnitudes
                            for group in variable_groups:
                                if group_counts[group] > 0:
                                    plot_data[opt_name][slot_type][group].append(
                                        group_mags[group] / group_counts[group]
                                    )
                                else:
                                    plot_data[opt_name][slot_type][group].append(0.0)
            
            # Create stacked plot showing slot types
            slot_list = sorted(list(slot_types_found))
            if len(slot_list) > 0:
                # Plot first slot type (usually momentum or m)
                primary_slot = slot_list[0] if slot_list else None
                if primary_slot:
                    for i, opt_name in enumerate(optimizer_results.keys()):
                        if opt_name in plot_data and primary_slot in plot_data[opt_name]:
                            # Plot conv layer magnitudes
                            if 'conv' in plot_data[opt_name][primary_slot]:
                                mags = plot_data[opt_name][primary_slot]['conv']
                                # Filter out non-finite values
                                if mags:
                                    filtered_mags = [m if np.isfinite(m) else np.nan for m in mags]
                                    if any(np.isfinite(m) and m > 0 for m in filtered_mags):
                                        steps = result['history']['global_steps'][:len(filtered_mags)]
                                        ax6.plot(steps, filtered_mags,
                                                color=colors[i], label=f'{opt_name} ({primary_slot})',
                                                linewidth=2, alpha=0.8)
                    
                    ax6.axvline(x=injection_step, color='red', linestyle='--', alpha=0.7)
                    ax6.set_xlabel('Training Step')
                    ax6.set_ylabel(f'Average {primary_slot} Magnitude')
                    ax6.set_title(f'Detailed State: {primary_slot} in Conv Layers')
                    ax6.legend(loc='best')
                    ax6.grid(True, alpha=0.3)
                    ax6.set_yscale('log')
                else:
                    ax6.axis('off')
                    ax6.text(0.5, 0.5, 'No optimizer state slots found\n(SGD_vanilla has no internal state)',
                            transform=ax6.transAxes, ha='center', va='center', fontsize=12)
            else:
                ax6.axis('off')
                ax6.text(0.5, 0.5, 'No detailed state data available', 
                        transform=ax6.transAxes, ha='center', va='center')
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, 'No detailed optimizer state tracking available',
                    transform=ax6.transAxes, ha='center', va='center')
        
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
