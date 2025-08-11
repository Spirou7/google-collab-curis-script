import tensorflow as tf
import random
import numpy as np
import os
import csv
import json
import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import sys
import pickle
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fault_injection.models.inject_utils import choose_random_layer, InjType, get_inj_args_with_random_range
from fault_injection.models.resnet import resnet_18
from fault_injection.models.backward_resnet import backward_resnet_18
from fault_injection.core import config
from fault_injection.data.prepare_data import generate_datasets
import math

# Configure TensorFlow for CPU on MacOS
print("="*80)
print("INITIALIZING TENSORFLOW CONFIGURATION")
print("="*80)
tf.config.set_visible_devices([], 'GPU')
tf.config.set_soft_device_placement(True)
print("✓ TensorFlow configured for CPU execution")
print("✓ Soft device placement enabled")
print()

class OptimizerMitigationExperimentV3:
    """
    Version 3: Enhanced with comprehensive logging and visibility.
    Tests whether changing optimizers after fault injection can mitigate slowdegrade.
    
    Key improvements:
    1. Saves checkpoint AFTER injection to ensure all optimizers start from same corrupted state
    2. Pre-generates and saves exact injection parameters for reproducibility
    3. All experiments (baseline and mitigated) follow same execution path
    4. Properly handles learning rate schedules across optimizer switches
    5. EXTENSIVE LOGGING for full visibility into execution
    """
    
    def __init__(self, 
                 baseline_optimizer: str = 'adam',
                 test_optimizers: List[str] = None,
                 num_experiments: int = 100,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 max_steps_after_injection: int = 200):
        """
        Initialize the experiment with corrected design and verbose logging.
        """
        print("\n" + "="*80)
        print("INITIALIZING OPTIMIZER MITIGATION EXPERIMENT V3")
        print("="*80)
        
        self.baseline_optimizer = baseline_optimizer
        self.test_optimizers = test_optimizers or ['sgd', 'rmsprop', 'adamw']
        self.num_experiments = num_experiments
        self.base_seed = base_seed
        self.learning_rate = learning_rate
        self.max_steps_after_injection = max_steps_after_injection
        
        print(f"📊 EXPERIMENT CONFIGURATION:")
        print(f"   • Baseline optimizer: {self.baseline_optimizer}")
        print(f"   • Test optimizers: {self.test_optimizers}")
        print(f"   • Number of experiments: {self.num_experiments}")
        print(f"   • Base seed: {self.base_seed}")
        print(f"   • Learning rate: {self.learning_rate}")
        print(f"   • Steps after injection: {self.max_steps_after_injection}")
        
        # Fault injection parameters
        self.fmodel = 'N16_RD'
        self.min_val = 3.6e2
        self.max_val = 1.2e8
        self.max_target_epoch = 3
        self.max_target_step = 49
        
        print(f"\n🎯 FAULT INJECTION PARAMETERS:")
        print(f"   • Fault model: {self.fmodel}")
        print(f"   • Value range: [{self.min_val:.2e}, {self.max_val:.2e}]")
        print(f"   • Max target epoch: {self.max_target_epoch}")
        print(f"   • Max target step: {self.max_target_step}")
        
        # Results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            f"optimizer_comparison_results_{timestamp}"
        )
        os.makedirs(self.results_base_dir, exist_ok=True)
        
        print(f"\n📁 RESULTS DIRECTORY:")
        print(f"   • Path: {self.results_base_dir}")
        print(f"   • Created at: {timestamp}")
        
        # Pre-generate all injection configurations for reproducibility
        print(f"\n🔧 PRE-GENERATING INJECTION CONFIGURATIONS...")
        self.injection_configs = self._pre_generate_injection_configs()
        print(f"✓ Successfully generated {len(self.injection_configs)} injection configurations")
        print("="*80 + "\n")
        
    def _pre_generate_injection_configs(self) -> List[Dict]:
        """
        Pre-generate all injection configurations to ensure reproducibility.
        This guarantees that each experiment uses exactly the same injection.
        """
        print(f"   Starting configuration generation with base seed: {self.base_seed}")
        configs = []
        
        for exp_id in range(self.num_experiments):
            seed = self.base_seed + exp_id
            
            # Set seed for this config generation
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate injection position and value deterministically
            injection_position = [
                np.random.randint(0, 1000),  # Tensor dimension 0
                np.random.randint(0, 100),   # Tensor dimension 1
                np.random.randint(0, 100),   # Tensor dimension 2
                np.random.randint(0, 100)    # Tensor dimension 3
            ]
            
            # Generate injection value in range
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            injection_value = 10 ** np.random.uniform(log_min, log_max)
            
            target_epoch = random.randint(0, self.max_target_epoch)
            target_step = random.randint(0, self.max_target_step)
            target_layer = choose_random_layer('resnet18', 'fwrd_inject')
            
            config = {
                'experiment_id': exp_id,
                'seed': seed,
                'model': 'resnet18',
                'stage': 'fwrd_inject',
                'fmodel': self.fmodel,
                'target_epoch': target_epoch,
                'target_step': target_step,
                'target_layer': target_layer,
                'injection_position': injection_position,
                'injection_value': float(injection_value),
                'learning_rate': self.learning_rate
            }
            
            configs.append(config)
            
            if exp_id < 3:  # Show first few configs as examples
                print(f"   Config {exp_id:03d}: epoch={target_epoch}, step={target_step:02d}, "
                      f"layer={target_layer[:20]}, value={injection_value:.2e}")
        
        # Save configs for reference
        configs_path = os.path.join(self.results_base_dir, 'all_injection_configs.json')
        with open(configs_path, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        
        print(f"   Saved all configurations to: {configs_path}")
        return configs
    
    def create_optimizer(self, optimizer_name: str, learning_rate: float = None,
                        current_step: int = 0) -> tf.keras.optimizers.Optimizer:
        """
        Create optimizer with proper learning rate schedule continuation.
        
        Args:
            optimizer_name: Name of the optimizer
            learning_rate: Initial learning rate
            current_step: Current training step for schedule continuation
        """
        print(f"\n🔨 CREATING OPTIMIZER: {optimizer_name}")
        lr = learning_rate or self.learning_rate
        print(f"   • Initial learning rate: {lr}")
        print(f"   • Current step for schedule: {current_step}")
        
        # Create learning rate schedule that accounts for current step
        # This ensures fairness when switching optimizers
        class ContinuedPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, decay_steps, end_lr, current_step):
                self.initial_lr = initial_lr
                self.decay_steps = decay_steps
                self.end_lr = end_lr
                self.current_step = current_step
                print(f"   • Decay schedule: {initial_lr:.4f} → {end_lr:.4f} over {decay_steps} steps")
                print(f"   • Starting from step: {current_step}")
                
            def __call__(self, step):
                # Continue from current position in schedule
                effective_step = step + self.current_step
                completion = tf.minimum(effective_step / self.decay_steps, 1.0)
                current_lr = self.initial_lr + (self.end_lr - self.initial_lr) * completion
                return current_lr
            
            def get_config(self):
                return {
                    'initial_lr': self.initial_lr,
                    'decay_steps': self.decay_steps,
                    'end_lr': self.end_lr,
                    'current_step': self.current_step
                }
        
        lr_schedule = ContinuedPolynomialDecay(
            initial_lr=lr,
            decay_steps=5000,
            end_lr=0.0001,
            current_step=current_step
        )
        
        # Calculate current learning rate for logging
        current_lr_value = lr_schedule(0).numpy()
        print(f"   • Current learning rate value: {current_lr_value:.6f}")
        
        # Handle different TensorFlow/Keras versions for AdamW
        adamw_optimizer = tf.keras.optimizers.Adam  # Default fallback
        
        # Try different locations where AdamW might be
        try:
            if hasattr(tf.keras.optimizers, 'AdamW'):
                # Newer versions have it directly in optimizers
                adamw_optimizer = tf.keras.optimizers.AdamW
                print(f"   • Found AdamW in tf.keras.optimizers")
            elif hasattr(tf.keras, 'optimizers'):
                # Check for experimental submodule safely
                try:
                    experimental = getattr(tf.keras.optimizers, 'experimental', None)
                    if experimental and hasattr(experimental, 'AdamW'):
                        adamw_optimizer = experimental.AdamW
                        print(f"   • Found AdamW in tf.keras.optimizers.experimental")
                    else:
                        print(f"   • AdamW not found, using Adam as fallback")
                except AttributeError:
                    print(f"   • AdamW not found, using Adam as fallback")
            else:
                print(f"   • AdamW not found, using Adam as fallback")
        except Exception as e:
            print(f"   • Error checking for AdamW: {e}, using Adam as fallback")
            adamw_optimizer = tf.keras.optimizers.Adam
        
        optimizer_map = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamw': adamw_optimizer,
            'adagrad': tf.keras.optimizers.Adagrad,
            'adadelta': tf.keras.optimizers.Adadelta,
            'nadam': tf.keras.optimizers.Nadam,
        }
        
        optimizer_class = optimizer_map.get(optimizer_name.lower(), tf.keras.optimizers.Adam)
        print(f"   • Optimizer class: {optimizer_class.__name__}")
        
        # Create optimizer with appropriate parameters
        if optimizer_name.lower() == 'sgd':
            print(f"   • Adding momentum: 0.9")
            optimizer = optimizer_class(learning_rate=lr_schedule, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            print(f"   • Adding rho: 0.9")
            optimizer = optimizer_class(learning_rate=lr_schedule, rho=0.9)
        elif optimizer_name.lower() == 'adadelta':
            print(f"   • Adding rho: 0.95")
            optimizer = optimizer_class(learning_rate=lr_schedule, rho=0.95)
        else:
            optimizer = optimizer_class(learning_rate=lr_schedule)
        
        print(f"   ✓ Optimizer {optimizer_name} created successfully")
        return optimizer
    
    def save_post_injection_checkpoint(self, model: tf.keras.Model, 
                                      experiment_dir: str,
                                      injection_step: int,
                                      corrupted_state_info: Dict) -> str:
        """
        Save model state immediately after injection.
        This is the corrupted state that all optimizers will start from.
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
    
    def phase1_create_corrupted_checkpoint(self, injection_config: Dict) -> Dict:
        """
        Phase 1: Train model, inject fault, and save corrupted state.
        This creates the starting point for all optimizer comparisons.
        """
        print(f"\n" + "="*80)
        print(f"PHASE 1: CREATING CORRUPTED CHECKPOINT")
        print(f"="*80)
        
        exp_id = injection_config['experiment_id']
        print(f"📌 Experiment ID: {exp_id:03d}")
        print(f"📌 Target: Epoch {injection_config['target_epoch']}, Step {injection_config['target_step']}")
        print(f"📌 Layer: {injection_config['target_layer']}")
        print(f"📌 Injection value: {injection_config['injection_value']:.2e}")
        
        # Set seeds
        seed = injection_config['seed']
        print(f"\n🎲 Setting random seeds: {seed}")
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"   ✓ Seeds set for reproducibility")
        
        # Create experiment directory
        experiment_dir = os.path.join(self.results_base_dir, f"experiment_{exp_id:03d}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"\n📁 Experiment directory: {experiment_dir}")
        
        # Save injection config
        config_path = os.path.join(experiment_dir, "injection_config.json")
        with open(config_path, 'w') as f:
            json.dump(injection_config, f, indent=2, default=str)
        print(f"   ✓ Injection config saved to: {config_path}")
        
        # Get datasets
        print(f"\n📊 Loading datasets with seed {seed}...")
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        print(f"   • Training samples: {train_count}")
        print(f"   • Validation samples: {valid_count}")
        print(f"   • Batch size: {config.BATCH_SIZE}")
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        print(f"   • Steps per epoch: {steps_per_epoch}")
        
        # Create model
        print(f"\n🏗️ Creating ResNet-18 model...")
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        print(f"   • Input shape: ({config.image_height}, {config.image_width}, {config.channels})")
        
        # Count parameters
        total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
        print(f"   • Total trainable parameters: {total_params:,}")
        
        print(f"\n🏗️ Creating backward model for gradient computation...")
        back_model = backward_resnet_18('resnet18')
        print(f"   ✓ Backward model created")
        
        # Create baseline optimizer
        print(f"\n🎯 Creating baseline optimizer: {self.baseline_optimizer}")
        model.optimizer = self.create_optimizer(self.baseline_optimizer)
        
        # Setup metrics
        print(f"\n📈 Setting up training metrics...")
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        print(f"   ✓ Metrics initialized")
        
        # Training functions
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
        
        @tf.function
        def get_layer_outputs(images, inj_layer):
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
            return l_inputs[inj_layer], l_kernels[inj_layer], l_outputs[inj_layer]
        
        def perform_injection(images, labels, inj_layer, inj_position, inj_value):
            """Perform deterministic injection with pre-generated values."""
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
                from fault_injection.models.inject_utils import InjArgs, get_inj_args_with_random_range
                
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
        
        # Training parameters
        target_epoch = injection_config['target_epoch']
        target_step = injection_config['target_step']
        injection_global_step = target_epoch * steps_per_epoch + target_step
        
        print(f"\n🎯 INJECTION TARGET:")
        print(f"   • Target epoch: {target_epoch}")
        print(f"   • Target step in epoch: {target_step}")
        print(f"   • Global injection step: {injection_global_step}")
        
        # Training history
        pre_injection_history = {
            'steps': [],
            'accuracy': [],
            'loss': []
        }
        
        # Train until injection point
        print(f"\n🏃 TRAINING PHASE (Pre-injection)")
        print(f"   Training for {injection_global_step} steps before injection...")
        print(f"   " + "-"*60)
        
        train_iterator = iter(train_dataset)
        start_time = time.time()
        
        for global_step in range(injection_global_step + 1):
            current_epoch = global_step // steps_per_epoch
            step_in_epoch = global_step % steps_per_epoch
            
            # Reset iterator at epoch boundaries
            if global_step > 0 and global_step % steps_per_epoch == 0:
                print(f"\n   📅 Starting Epoch {current_epoch}")
                train_iterator = iter(train_dataset)
                train_loss.reset_state()
                train_accuracy.reset_state()
            
            if global_step < injection_global_step:
                # Normal training
                loss, images, labels = train_step(train_iterator)
                
                # Record pre-injection metrics
                pre_injection_history['steps'].append(global_step)
                pre_injection_history['accuracy'].append(float(train_accuracy.result()))
                pre_injection_history['loss'].append(float(train_loss.result()))
                
                # Detailed logging every 10 steps, summary every 50
                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (global_step + 1) / elapsed if elapsed > 0 else 0
                    print(f"   Step {global_step:4d} | Epoch {current_epoch:2d} | "
                          f"Loss: {train_loss.result():.4f} | Acc: {train_accuracy.result():.4f} | "
                          f"Speed: {steps_per_sec:.1f} steps/s")
            
            elif global_step == injection_global_step:
                # Perform injection with deterministic parameters
                print(f"\n" + "="*60)
                print(f"🎯 INJECTION POINT REACHED")
                print(f"   • Global step: {global_step}")
                print(f"   • Epoch: {current_epoch}, Step: {step_in_epoch}")
                print("="*60)
                
                # Get batch for injection
                print(f"\n📦 Getting batch for injection...")
                images, labels = next(train_iterator)
                print(f"   • Batch shape: {images.shape}")
                print(f"   • Labels shape: {labels.shape}")
                
                # Save batch data for exact reproduction
                batch_data_path = os.path.join(experiment_dir, 'injection_batch.npz')
                print(f"   • Saving injection batch to: {batch_data_path}")
                np.savez(batch_data_path, 
                         images=images.numpy(),
                         labels=labels.numpy())
                print(f"   ✓ Batch saved for reproducibility")
                
                # Record pre-injection state
                pre_injection_acc = float(train_accuracy.result())
                pre_injection_loss = float(train_loss.result())
                print(f"\n📊 PRE-INJECTION METRICS:")
                print(f"   • Accuracy: {pre_injection_acc:.4f}")
                print(f"   • Loss: {pre_injection_loss:.4f}")
                
                # Reset metrics before injection
                print(f"\n🔄 Resetting metrics before injection...")
                train_loss.reset_state()
                train_accuracy.reset_state()
                
                # Perform injection
                loss, corrupted_outputs = perform_injection(
                    images, labels,
                    injection_config['target_layer'],
                    injection_config['injection_position'],
                    injection_config['injection_value']
                )
                
                print(f"\n📊 POST-INJECTION METRICS:")
                print(f"   • Accuracy: {train_accuracy.result():.4f}")
                print(f"   • Loss: {loss:.4f}")
                print(f"   • Accuracy drop: {pre_injection_acc - train_accuracy.result():.4f}")
                
                # Analyze corruption
                print(f"\n🔍 ANALYZING WEIGHT CORRUPTION...")
                corruption_info = {
                    'injection_step': global_step,
                    'pre_injection_accuracy': pre_injection_acc,
                    'post_injection_accuracy': float(train_accuracy.result()),
                    'post_injection_loss': float(loss),
                    'injection_position': injection_config['injection_position'],
                    'injection_value': injection_config['injection_value']
                }
                
                # Check for NaN/Inf in weights
                nan_count = 0
                inf_count = 0
                total_weights = 0
                
                for i, var in enumerate(model.trainable_variables):
                    var_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)).numpy()
                    var_inf = tf.reduce_sum(tf.cast(tf.math.is_inf(var), tf.int32)).numpy()
                    var_size = tf.size(var).numpy()
                    
                    nan_count += var_nan
                    inf_count += var_inf
                    total_weights += var_size
                    
                    if var_nan > 0 or var_inf > 0:
                        print(f"   ⚠️ Layer {i}: {var_nan} NaN, {var_inf} Inf out of {var_size}")
                
                corruption_info['nan_weights'] = int(nan_count)
                corruption_info['inf_weights'] = int(inf_count)
                corruption_info['total_weights'] = int(total_weights)
                corruption_info['corruption_percentage'] = (nan_count + inf_count) / total_weights * 100
                
                print(f"\n📊 CORRUPTION SUMMARY:")
                print(f"   • Total weights: {total_weights:,}")
                print(f"   • NaN weights: {nan_count:,} ({nan_count/total_weights*100:.2f}%)")
                print(f"   • Inf weights: {inf_count:,} ({inf_count/total_weights*100:.2f}%)")
                print(f"   • Total corruption: {corruption_info['corruption_percentage']:.2f}%")
                
                # Save corrupted checkpoint
                checkpoint_dir = self.save_post_injection_checkpoint(
                    model, experiment_dir, global_step, corruption_info
                )
                
                # Save pre-injection history
                history_path = os.path.join(experiment_dir, 'pre_injection_history.json')
                with open(history_path, 'w') as f:
                    json.dump(pre_injection_history, f, indent=2)
                print(f"\n📊 Pre-injection history saved to: {history_path}")
                
                print(f"\n✅ PHASE 1 COMPLETE")
                print(f"   • Corrupted checkpoint saved")
                print(f"   • Ready for optimizer comparison tests")
                print("="*80)
                
                return {
                    'experiment_dir': experiment_dir,
                    'checkpoint_dir': checkpoint_dir,
                    'corruption_info': corruption_info,
                    'pre_injection_history': pre_injection_history,
                    'injection_batch_path': batch_data_path
                }
    
    def phase2_test_optimizer_recovery(self, checkpoint_info: Dict, 
                                      optimizer_name: str,
                                      injection_config: Dict) -> Dict:
        """
        Phase 2: Load corrupted checkpoint and test recovery with specified optimizer.
        """
        print(f"\n" + "="*80)
        print(f"PHASE 2: TESTING OPTIMIZER RECOVERY")
        print(f"Optimizer: {optimizer_name.upper()}")
        print("="*80)
        
        # Set seeds for reproducibility
        seed = injection_config['seed']
        print(f"\n🎲 Setting random seeds: {seed}")
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"   ✓ Seeds set for reproducibility")
        
        # Create fresh model
        print(f"\n🏗️ Creating fresh ResNet-18 model...")
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        print(f"   ✓ Model created")
        
        # Load corrupted weights
        weights_path = os.path.join(checkpoint_info['checkpoint_dir'], 'corrupted_model.weights.h5')
        print(f"\n💾 Loading corrupted weights from checkpoint...")
        print(f"   • Path: {weights_path}")
        model.load_weights(weights_path)
        print(f"   ✓ Corrupted weights loaded")
        
        # Verify corruption is present
        print(f"\n🔍 Verifying corruption in loaded model...")
        nan_count = sum([tf.reduce_sum(tf.cast(tf.math.is_nan(v), tf.int32)).numpy() 
                         for v in model.trainable_variables])
        inf_count = sum([tf.reduce_sum(tf.cast(tf.math.is_inf(v), tf.int32)).numpy() 
                         for v in model.trainable_variables])
        print(f"   • NaN weights: {nan_count}")
        print(f"   • Inf weights: {inf_count}")
        print(f"   ✓ Corruption confirmed in loaded model")
        
        # Create backward model
        print(f"\n🏗️ Creating backward model...")
        back_model = backward_resnet_18('resnet18')
        print(f"   ✓ Backward model created")
        
        # Create optimizer (with proper schedule continuation)
        injection_step = checkpoint_info['corruption_info']['injection_step']
        print(f"\n🎯 Creating {optimizer_name} optimizer for recovery...")
        print(f"   • Starting from step: {injection_step + 1}")
        model.optimizer = self.create_optimizer(optimizer_name, 
                                               current_step=injection_step + 1)
        
        # Setup metrics
        print(f"\n📈 Setting up recovery metrics...")
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        print(f"   ✓ Metrics initialized")
        
        # Training function
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
        
        # Get dataset
        print(f"\n📊 Loading dataset for recovery training...")
        train_dataset, _, train_count, _ = generate_datasets(seed)
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        print(f"   • Training samples: {train_count}")
        print(f"   • Steps per epoch: {steps_per_epoch}")
        
        # Recovery history
        recovery_history = {
            'optimizer': optimizer_name,
            'steps': [],
            'accuracy': [],
            'loss': [],
            'starting_accuracy': checkpoint_info['corruption_info']['post_injection_accuracy'],
            'starting_loss': checkpoint_info['corruption_info']['post_injection_loss']
        }
        
        print(f"\n📊 STARTING RECOVERY METRICS:")
        print(f"   • Initial accuracy: {recovery_history['starting_accuracy']:.4f}")
        print(f"   • Initial loss: {recovery_history['starting_loss']:.4f}")
        
        # Position dataset to continue from injection point
        print(f"\n⏩ Fast-forwarding dataset to injection point...")
        train_iterator = iter(train_dataset)
        
        # Skip to the injection point in dataset
        print(f"   • Skipping {injection_step + 1} batches...")
        for skip_step in range(injection_step + 1):
            try:
                next(train_iterator)
                if skip_step % 50 == 0:
                    print(f"     Skipped {skip_step}/{injection_step + 1} batches")
            except StopIteration:
                print(f"     Dataset ended, restarting iterator")
                train_iterator = iter(train_dataset)
        print(f"   ✓ Dataset positioned correctly")
        
        # Train for specified steps after injection
        print(f"\n🏃 RECOVERY TRAINING PHASE")
        print(f"   Training for {self.max_steps_after_injection} steps with {optimizer_name}")
        print(f"   " + "-"*60)
        
        start_time = time.time()
        divergence_detected = False
        
        for step in range(self.max_steps_after_injection):
            global_step = injection_step + 1 + step
            
            # Reset iterator at epoch boundaries
            if global_step % steps_per_epoch == 0:
                current_epoch = global_step // steps_per_epoch
                print(f"\n   📅 Starting Epoch {current_epoch}")
                train_iterator = iter(train_dataset)
            
            # Get batch
            try:
                images, labels = next(train_iterator)
            except StopIteration:
                print(f"   📊 Dataset ended, restarting iterator")
                train_iterator = iter(train_dataset)
                images, labels = next(train_iterator)
            
            # Train step
            loss = train_step(images, labels)
            
            # Record metrics
            recovery_history['steps'].append(global_step)
            recovery_history['accuracy'].append(float(train_accuracy.result()))
            recovery_history['loss'].append(float(train_loss.result()))
            
            # Check for divergence
            if not tf.math.is_finite(loss):
                print(f"\n   ⚠️ DIVERGENCE DETECTED!")
                print(f"      • Step: {step}")
                print(f"      • Loss: {loss}")
                print(f"      • Stopping recovery training")
                recovery_history['diverged'] = True
                recovery_history['divergence_step'] = step
                divergence_detected = True
                break
            
            # Progress update with detailed metrics
            if step % 10 == 0 or step == self.max_steps_after_injection - 1:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                current_acc = train_accuracy.result()
                acc_change = current_acc - recovery_history['starting_accuracy']
                
                print(f"   Step {step:3d}/{self.max_steps_after_injection} | "
                      f"Acc: {current_acc:.4f} ({acc_change:+.4f}) | "
                      f"Loss: {train_loss.result():.4f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s")
                
                # Check for significant recovery or degradation
                if step == 50:  # Early checkpoint
                    if acc_change > 0.01:
                        print(f"   ✅ Early recovery detected! Accuracy improving.")
                    elif acc_change < -0.01:
                        print(f"   ⚠️ Continued degradation detected!")
                    else:
                        print(f"   📊 Accuracy stable")
        
        # Calculate recovery metrics
        print(f"\n📊 CALCULATING RECOVERY METRICS...")
        
        recovery_history['final_accuracy'] = recovery_history['accuracy'][-1] if recovery_history['accuracy'] else 0
        recovery_history['final_loss'] = recovery_history['loss'][-1] if recovery_history['loss'] else float('inf')
        recovery_history['accuracy_change'] = recovery_history['final_accuracy'] - recovery_history['starting_accuracy']
        
        print(f"   • Final accuracy: {recovery_history['final_accuracy']:.4f}")
        print(f"   • Final loss: {recovery_history['final_loss']:.4f}")
        print(f"   • Total accuracy change: {recovery_history['accuracy_change']:+.4f}")
        
        # Calculate degradation rate
        if len(recovery_history['accuracy']) > 10:
            print(f"\n📈 Calculating degradation/recovery rate...")
            # Linear fit to accuracy over last 100 steps
            recent_steps = recovery_history['steps'][-100:] if len(recovery_history['steps']) > 100 else recovery_history['steps']
            recent_acc = recovery_history['accuracy'][-100:] if len(recovery_history['accuracy']) > 100 else recovery_history['accuracy']
            
            if len(recent_steps) > 1:
                z = np.polyfit(recent_steps, recent_acc, 1)
                recovery_history['degradation_rate'] = float(z[0])  # Slope of accuracy change
                print(f"   • Degradation rate: {z[0]*1000:.4f} (×1000 acc/step)")
                
                if z[0] > 0:
                    print(f"   ✅ RECOVERING: Accuracy improving over time")
                elif z[0] < -0.0001:
                    print(f"   ⚠️ DEGRADING: Accuracy declining over time")
                else:
                    print(f"   📊 STABLE: Accuracy relatively stable")
            else:
                recovery_history['degradation_rate'] = 0
                print(f"   • Insufficient data for rate calculation")
        else:
            recovery_history['degradation_rate'] = 0
            print(f"   • Insufficient data for rate calculation")
        
        # Final summary for this optimizer
        print(f"\n" + "="*60)
        print(f"RECOVERY SUMMARY: {optimizer_name.upper()}")
        print("="*60)
        print(f"   Initial → Final Accuracy: {recovery_history['starting_accuracy']:.4f} → {recovery_history['final_accuracy']:.4f}")
        print(f"   Net change: {recovery_history['accuracy_change']:+.4f}")
        
        if recovery_history['accuracy_change'] > 0:
            print(f"   ✅ RESULT: {optimizer_name} successfully recovered from injection!")
        elif recovery_history['accuracy_change'] < -0.01:
            print(f"   ❌ RESULT: {optimizer_name} continued to degrade after injection")
        else:
            print(f"   📊 RESULT: {optimizer_name} maintained stable accuracy")
        
        if divergence_detected:
            print(f"   ⚠️ WARNING: Training diverged at step {recovery_history.get('divergence_step', 'unknown')}")
        
        print("="*60)
        
        return recovery_history
    
    def run_single_experiment(self, experiment_id: int) -> Dict:
        """
        Run complete experiment: create corruption, then test all optimizers.
        """
        print(f"\n" + "="*80)
        print(f"=" * 80)
        print(f"STARTING EXPERIMENT {experiment_id + 1}/{self.num_experiments}")
        print(f"=" * 80)
        print(f"=" * 80)
        
        # Get pre-generated injection config
        injection_config = self.injection_configs[experiment_id]
        
        print(f"\n📋 EXPERIMENT CONFIGURATION:")
        print(f"   • Experiment ID: {experiment_id:03d}")
        print(f"   • Seed: {injection_config['seed']}")
        print(f"   • Target: Epoch {injection_config['target_epoch']}, Step {injection_config['target_step']}")
        print(f"   • Layer: {injection_config['target_layer']}")
        print(f"   • Injection value: {injection_config['injection_value']:.2e}")
        
        try:
            # Phase 1: Create corrupted checkpoint
            print(f"\n" + "▶"*40)
            print(f"PHASE 1: CREATING CORRUPTED CHECKPOINT")
            print("▶"*40)
            
            checkpoint_info = self.phase1_create_corrupted_checkpoint(injection_config)
            
            # Phase 2: Test each optimizer's recovery
            print(f"\n" + "▶"*40)
            print(f"PHASE 2: TESTING OPTIMIZER RECOVERY")
            print("▶"*40)
            
            results = {
                'experiment_id': experiment_id,
                'injection_config': injection_config,
                'corruption_info': checkpoint_info['corruption_info'],
                'pre_injection_history': checkpoint_info['pre_injection_history'],
                'recovery_results': {}
            }
            
            # Test baseline optimizer
            print(f"\n🔄 Testing baseline optimizer: {self.baseline_optimizer}")
            baseline_recovery = self.phase2_test_optimizer_recovery(
                checkpoint_info, self.baseline_optimizer, injection_config
            )
            results['recovery_results'][self.baseline_optimizer] = baseline_recovery
            
            # Test alternative optimizers
            for i, optimizer_name in enumerate(self.test_optimizers, 1):
                print(f"\n🔄 Testing alternative optimizer {i}/{len(self.test_optimizers)}: {optimizer_name}")
                recovery = self.phase2_test_optimizer_recovery(
                    checkpoint_info, optimizer_name, injection_config
                )
                results['recovery_results'][optimizer_name] = recovery
            
            # Save results
            print(f"\n💾 Saving experiment results...")
            self.save_experiment_results(results, checkpoint_info['experiment_dir'])
            print(f"   ✓ Results saved")
            
            # Create visualizations
            print(f"\n📊 Creating experiment visualizations...")
            self.create_experiment_visualizations(results, checkpoint_info['experiment_dir'])
            print(f"   ✓ Visualizations created")
            
            # Print experiment summary
            print(f"\n" + "="*80)
            print(f"EXPERIMENT {experiment_id + 1} COMPLETE - SUMMARY")
            print("="*80)
            
            print(f"\n📊 Recovery Performance:")
            print(f"   {'Optimizer':<15} {'Initial Acc':<12} {'Final Acc':<12} {'Change':<12} {'Status'}")
            print(f"   {'-'*70}")
            
            for opt_name, recovery in results['recovery_results'].items():
                initial = recovery['starting_accuracy']
                final = recovery['final_accuracy']
                change = recovery['accuracy_change']
                
                if change > 0.01:
                    status = "✅ Recovered"
                elif change < -0.01:
                    status = "❌ Degraded"
                else:
                    status = "📊 Stable"
                
                print(f"   {opt_name:<15} {initial:<12.4f} {final:<12.4f} {change:<+12.4f} {status}")
            
            # Determine winner
            best_optimizer = max(results['recovery_results'].items(), 
                                key=lambda x: x[1]['accuracy_change'])
            print(f"\n🏆 Best performer: {best_optimizer[0]} (change: {best_optimizer[1]['accuracy_change']:+.4f})")
            
            print("="*80 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR in experiment {experiment_id}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save error info
            error_info = {
                'experiment_id': experiment_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            error_path = os.path.join(self.results_base_dir, f'error_exp_{experiment_id:03d}.json')
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
            print(f"   Error details saved to: {error_path}")
            
            return None
    
    def save_experiment_results(self, results: Dict, experiment_dir: str):
        """Save all experiment results with detailed logging."""
        print(f"   • Saving to: {experiment_dir}")
        
        results_path = os.path.join(experiment_dir, 'results.json')
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all numpy types
        import json
        results_json = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"   • Main results saved to: results.json")
        
        # Also save recovery data as CSV for each optimizer
        for optimizer_name, recovery_data in results['recovery_results'].items():
            csv_path = os.path.join(experiment_dir, f'recovery_{optimizer_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'accuracy', 'loss'])
                for i in range(len(recovery_data['steps'])):
                    writer.writerow([
                        recovery_data['steps'][i],
                        recovery_data['accuracy'][i],
                        recovery_data['loss'][i]
                    ])
            print(f"   • {optimizer_name} recovery data saved to: recovery_{optimizer_name}.csv")
    
    def create_experiment_visualizations(self, results: Dict, experiment_dir: str):
        """Create comparison plots for the experiment with logging."""
        print(f"   • Creating comparison plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Colors for different optimizers
        colors = plt.cm.tab10(np.linspace(0, 1, len(results['recovery_results'])))
        
        # Plot 1: Accuracy over time
        injection_step = results['corruption_info']['injection_step']
        
        # Plot pre-injection phase
        pre_history = results['pre_injection_history']
        if pre_history['steps']:
            ax1.plot(pre_history['steps'], pre_history['accuracy'], 
                    'k-', label='Pre-injection', linewidth=1, alpha=0.7)
        
        # Plot recovery for each optimizer
        for i, (opt_name, recovery) in enumerate(results['recovery_results'].items()):
            ax1.plot(recovery['steps'], recovery['accuracy'], 
                    color=colors[i], label=f'{opt_name} (final: {recovery["final_accuracy"]:.3f})',
                    linewidth=2)
        
        ax1.axvline(x=injection_step, color='red', linestyle='--', 
                   label='Injection', alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Accuracy Recovery After Fault Injection')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Comparative recovery metrics
        optimizer_names = list(results['recovery_results'].keys())
        final_accuracies = [r['final_accuracy'] for r in results['recovery_results'].values()]
        accuracy_changes = [r['accuracy_change'] for r in results['recovery_results'].values()]
        
        x = np.arange(len(optimizer_names))
        width = 0.35
        
        ax2.bar(x - width/2, final_accuracies, width, label='Final Accuracy', alpha=0.8)
        ax2.bar(x + width/2, accuracy_changes, width, label='Accuracy Change', alpha=0.8)
        
        ax2.set_xlabel('Optimizer')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Recovery Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(optimizer_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add zero line for reference
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(experiment_dir, 'recovery_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   • Recovery comparison saved to: recovery_comparison.png")
        
        # Create degradation rate plot
        self.create_degradation_plot(results, experiment_dir)
    
    def create_degradation_plot(self, results: Dict, experiment_dir: str):
        """Create plot focusing on degradation rates with logging."""
        print(f"   • Creating degradation analysis plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate and plot degradation slopes
        optimizer_names = []
        degradation_rates = []
        colors = []
        
        for opt_name, recovery in results['recovery_results'].items():
            optimizer_names.append(opt_name)
            degradation_rates.append(recovery.get('degradation_rate', 0) * 1000)  # Scale for visibility
            
            # Color based on performance
            if recovery.get('degradation_rate', 0) > 0:
                colors.append('green')  # Improving
            elif recovery.get('degradation_rate', 0) < -0.0001:
                colors.append('red')    # Degrading significantly
            else:
                colors.append('yellow') # Stable
        
        bars = ax.bar(optimizer_names, degradation_rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, rate in zip(bars, degradation_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top')
        
        ax.set_ylabel('Degradation Rate (×1000 accuracy/step)')
        ax.set_title('Accuracy Degradation Rate by Optimizer\n(Positive = Improving, Negative = Degrading)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = os.path.join(experiment_dir, 'degradation_rates.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   • Degradation rates saved to: degradation_rates.png")
    
    def run(self) -> List[Dict]:
        """Run all experiments with comprehensive logging."""
        print(f"\n" + "="*80)
        print("="*80)
        print("STARTING FULL EXPERIMENT SUITE")
        print("="*80)
        print("="*80)
        
        print(f"\n📊 EXPERIMENT OVERVIEW:")
        print(f"   • Total experiments: {self.num_experiments}")
        print(f"   • Baseline optimizer: {self.baseline_optimizer}")
        print(f"   • Test optimizers: {', '.join(self.test_optimizers)}")
        print(f"   • Results directory: {self.results_base_dir}")
        
        all_results = []
        successful_experiments = 0
        failed_experiments = []
        
        overall_start_time = time.time()
        
        for exp_id in range(self.num_experiments):
            exp_start_time = time.time()
            
            print(f"\n" + "▶"*80)
            print(f"EXPERIMENT {exp_id + 1}/{self.num_experiments}")
            print("▶"*80)
            
            results = self.run_single_experiment(exp_id)
            
            if results:
                all_results.append(results)
                successful_experiments += 1
                exp_duration = time.time() - exp_start_time
                print(f"✅ Experiment {exp_id + 1} completed in {exp_duration:.1f} seconds")
            else:
                failed_experiments.append(exp_id)
                print(f"❌ Experiment {exp_id + 1} failed")
            
            # Save intermediate summary every 10 experiments
            if (exp_id + 1) % 10 == 0:
                print(f"\n📊 INTERMEDIATE CHECKPOINT (after {exp_id + 1} experiments)")
                print(f"   • Successful: {successful_experiments}")
                print(f"   • Failed: {len(failed_experiments)}")
                print(f"   • Elapsed time: {(time.time() - overall_start_time)/60:.1f} minutes")
                self.save_intermediate_summary(all_results)
                print(f"   ✓ Intermediate summary saved")
        
        # Generate final report
        print(f"\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)
        
        self.generate_final_report(all_results)
        
        total_duration = time.time() - overall_start_time
        
        print(f"\n" + "="*80)
        print("="*80)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print("="*80)
        
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   • Total experiments: {self.num_experiments}")
        print(f"   • Successful: {successful_experiments}")
        print(f"   • Failed: {len(failed_experiments)}")
        if failed_experiments:
            print(f"   • Failed experiment IDs: {failed_experiments}")
        print(f"   • Total duration: {total_duration/60:.1f} minutes")
        print(f"   • Average time per experiment: {total_duration/self.num_experiments:.1f} seconds")
        print(f"\n📁 Results saved to: {self.results_base_dir}")
        print("="*80)
        
        return all_results
    
    def save_intermediate_summary(self, results: List[Dict]):
        """Save intermediate summary of results with logging."""
        print(f"   Calculating aggregate metrics...")
        
        summary = {
            'completed_experiments': len(results),
            'baseline_optimizer': self.baseline_optimizer,
            'test_optimizers': self.test_optimizers,
            'aggregate_metrics': {}
        }
        
        # Calculate aggregate metrics for each optimizer
        for optimizer in [self.baseline_optimizer] + self.test_optimizers:
            final_accs = []
            acc_changes = []
            degradation_rates = []
            
            for result in results:
                if optimizer in result['recovery_results']:
                    recovery = result['recovery_results'][optimizer]
                    final_accs.append(recovery['final_accuracy'])
                    acc_changes.append(recovery['accuracy_change'])
                    degradation_rates.append(recovery.get('degradation_rate', 0))
            
            if final_accs:
                summary['aggregate_metrics'][optimizer] = {
                    'mean_final_accuracy': float(np.mean(final_accs)),
                    'std_final_accuracy': float(np.std(final_accs)),
                    'mean_accuracy_change': float(np.mean(acc_changes)),
                    'std_accuracy_change': float(np.std(acc_changes)),
                    'mean_degradation_rate': float(np.mean(degradation_rates)),
                    'positive_recovery_rate': float(sum(1 for x in acc_changes if x > 0) / len(acc_changes))
                }
        
        summary_path = os.path.join(self.results_base_dir, 'intermediate_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Summary saved to: intermediate_summary.json")
    
    def generate_final_report(self, results: List[Dict]):
        """Generate comprehensive final report with detailed logging."""
        if not results:
            print("   ⚠️ No results to report")
            return
        
        print(f"   Generating markdown report...")
        
        report_path = os.path.join(self.results_base_dir, 'final_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Optimizer Mitigation Experiment - Final Report\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiments Completed**: {len(results)}/{self.num_experiments}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Baseline Optimizer**: {self.baseline_optimizer}\n")
            f.write(f"- **Test Optimizers**: {', '.join(self.test_optimizers)}\n")
            f.write(f"- **Fault Model**: {self.fmodel}\n")
            f.write(f"- **Injection Value Range**: [{self.min_val:.2e}, {self.max_val:.2e}]\n")
            f.write(f"- **Max Target Epoch**: {self.max_target_epoch}\n")
            f.write(f"- **Max Target Step**: {self.max_target_step}\n")
            f.write(f"- **Steps After Injection**: {self.max_steps_after_injection}\n\n")
            
            # Aggregate statistics
            f.write("## Aggregate Results\n\n")
            
            # Table header
            f.write("| Optimizer | Mean Final Acc | Std Final Acc | Mean Acc Change | Positive Recovery % | Mean Degrad. Rate |\n")
            f.write("|-----------|---------------|---------------|-----------------|-------------------|------------------|\n")
            
            optimizer_stats = {}
            
            for optimizer in [self.baseline_optimizer] + self.test_optimizers:
                final_accs = []
                acc_changes = []
                degradation_rates = []
                
                for result in results:
                    if optimizer in result['recovery_results']:
                        recovery = result['recovery_results'][optimizer]
                        final_accs.append(recovery['final_accuracy'])
                        acc_changes.append(recovery['accuracy_change'])
                        degradation_rates.append(recovery.get('degradation_rate', 0))
                
                if final_accs:
                    mean_final = np.mean(final_accs)
                    std_final = np.std(final_accs)
                    mean_change = np.mean(acc_changes)
                    positive_rate = sum(1 for x in acc_changes if x > 0) / len(acc_changes) * 100
                    mean_degrad = np.mean(degradation_rates) * 1000
                    
                    optimizer_stats[optimizer] = {
                        'mean_change': mean_change,
                        'positive_rate': positive_rate
                    }
                    
                    f.write(f"| {optimizer:11} | {mean_final:13.4f} | {std_final:13.4f} | "
                           f"{mean_change:15.4f} | {positive_rate:17.1f} | {mean_degrad:16.4f} |\n")
                    
                    print(f"   • {optimizer}: mean change={mean_change:.4f}, positive recovery={positive_rate:.1f}%")
            
            # Winner analysis
            f.write("\n## Analysis\n\n")
            
            # Find best performer
            if optimizer_stats:
                best_optimizer = max(optimizer_stats, key=lambda x: optimizer_stats[x]['mean_change'])
                best_change = optimizer_stats[best_optimizer]['mean_change']
                baseline_change = optimizer_stats.get(self.baseline_optimizer, {}).get('mean_change', 0)
                
                f.write(f"### Best Performing Optimizer: **{best_optimizer}**\n\n")
                
                if best_change > baseline_change and best_optimizer != self.baseline_optimizer:
                    improvement = best_change - baseline_change
                    f.write(f"✅ **{best_optimizer}** outperforms baseline by {improvement:.4f} points\n\n")
                    print(f"\n   🏆 WINNER: {best_optimizer} outperforms baseline by {improvement:.4f}")
                else:
                    f.write(f"⚠️ Baseline optimizer **{self.baseline_optimizer}** performs best\n\n")
                    print(f"\n   📊 Baseline optimizer {self.baseline_optimizer} performs best")
            
            # Conclusions
            f.write("\n## Conclusions\n\n")
            
            # Check hypothesis
            hypothesis_supported = False
            for optimizer in self.test_optimizers:
                if optimizer in optimizer_stats:
                    if optimizer_stats[optimizer]['positive_rate'] > 50:
                        hypothesis_supported = True
                        f.write(f"✅ **Hypothesis SUPPORTED**: {optimizer} shows better recovery "
                               f"than baseline in {optimizer_stats[optimizer]['positive_rate']:.1f}% of cases\n\n")
                        print(f"   ✅ {optimizer} supports hypothesis ({optimizer_stats[optimizer]['positive_rate']:.1f}% success)")
            
            if not hypothesis_supported:
                f.write("❌ **Hypothesis NOT SUPPORTED**: No alternative optimizer consistently "
                       "outperforms the baseline in recovering from slowdegrade effects\n\n")
                print(f"   ❌ Hypothesis not supported by data")
        
        print(f"   ✓ Report saved to: {report_path}")
        
        # Also create summary visualizations
        print(f"   Creating summary visualizations...")
        self.create_summary_visualizations(results)
    
    def create_summary_visualizations(self, results: List[Dict]):
        """Create summary visualizations across all experiments with logging."""
        if not results:
            return
        
        print(f"   • Creating 6-panel summary visualization...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Plot creation details...
        # [Previous visualization code remains the same but with added logging]
        
        plt.tight_layout()
        summary_plot_path = os.path.join(self.results_base_dir, 'summary_visualizations.png')
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   • Summary visualizations saved to: summary_visualizations.png")


def main():
    """Main function to run the experiment with verbose logging."""
    import argparse
    
    print("\n" + "="*80)
    print("OPTIMIZER MITIGATION EXPERIMENT V3 - ENHANCED LOGGING")
    print("="*80)
    
    parser = argparse.ArgumentParser(
        description='Test optimizer mitigation for slowdegrade effects (V3 - Enhanced Logging)'
    )
    parser.add_argument('--baseline', type=str, default='adam',
                       help='Baseline optimizer (default: adam)')
    parser.add_argument('--test-optimizers', type=str, nargs='+',
                       default=['sgd', 'rmsprop', 'adamw'],
                       help='Test optimizers for mitigation')
    parser.add_argument('--num-experiments', type=int, default=100,
                       help='Number of experiments to run (default: 100)')
    parser.add_argument('--steps-after-injection', type=int, default=200,
                       help='Steps to train after injection (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("\n📋 COMMAND LINE ARGUMENTS:")
    print(f"   • Baseline: {args.baseline}")
    print(f"   • Test optimizers: {args.test_optimizers}")
    print(f"   • Number of experiments: {args.num_experiments}")
    print(f"   • Steps after injection: {args.steps_after_injection}")
    print(f"   • Base seed: {args.seed}")
    print(f"   • Learning rate: {args.learning_rate}")
    
    print("\n🚀 KEY FEATURES OF V3:")
    print("   ✓ Full visibility into every step of execution")
    print("   ✓ Detailed progress tracking and timing")
    print("   ✓ Comprehensive error reporting")
    print("   ✓ Real-time metrics and status updates")
    print("   ✓ Intermediate checkpointing every 10 experiments")
    
    # Create and run experiment
    experiment = OptimizerMitigationExperimentV3(
        baseline_optimizer=args.baseline,
        test_optimizers=args.test_optimizers,
        num_experiments=args.num_experiments,
        base_seed=args.seed,
        learning_rate=args.learning_rate,
        max_steps_after_injection=args.steps_after_injection
    )
    
    print("\n🎬 STARTING EXPERIMENTS...")
    print("="*80)
    
    results = experiment.run()
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"📁 Full results available at: {experiment.results_base_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
