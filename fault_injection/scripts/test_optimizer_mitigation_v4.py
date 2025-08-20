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
                 # Optimizer comparison parameters
                 optimizers_to_test=None, num_experiments=10,
                 steps_after_injection=100):
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
            optimizers_to_test: List of optimizer names to compare
            num_experiments: Number of experiments to run
            steps_after_injection: Steps to continue after injection
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
        
        # Optimizer comparison parameters
        self.optimizers_to_test = optimizers_to_test or ['adam', 'sgd', 'rmsprop', 'adamw']
        self.num_experiments = num_experiments
        self.steps_after_injection = steps_after_injection
        
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
        
        # Use Docker volume path if available, fallback to local results directory
        if os.path.exists("/app/fault_injection/results"):
            # Running in Docker container with mounted volume
            self.results_base_dir = os.path.join("/app/fault_injection/results", f"optimizer_comparison_{timestamp}")
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
            'target_epoch': self.target_epoch if self.target_epoch is not None else random.randint(0, 3),
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
        elif optimizer_name.lower() == 'rmsprop':
            return optimizer_class(learning_rate=lr_schedule, rho=0.9)
        elif optimizer_name.lower() == 'adadelta':
            return optimizer_class(learning_rate=lr_schedule, rho=0.95)
        else:
            return optimizer_class(learning_rate=lr_schedule)
    
    def train_with_injection(self, optimizer_name: str, injection_params: Dict, 
                            experiment_dir: str) -> Dict:
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
        def train_step(iterator):
            images, labels = next(iterator)
            with tf.GradientTape() as tape:
                outputs, _, _, l_outputs = model(images, training=True, inject=False)
                predictions = outputs['logits']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            
            tvars = model.trainable_variables
            gradients = tape.gradient(avg_loss, tvars)
            model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss
        
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
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss, l_outputs
        
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
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss, l_outputs
        
        # Training loop
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        target_epoch = injection_params['target_epoch']
        target_step = injection_params['target_step']
        injection_global_step = target_epoch * steps_per_epoch + target_step
        
        record(f"Target injection: epoch {target_epoch}, step {target_step}\n")
        record(f"Global injection step: {injection_global_step}\n")
        
        # History tracking
        history = {
            'steps': [],
            'accuracy': [],
            'loss': []
        }
        
        # Train until injection point
        train_iterator = iter(train_dataset)
        global_step = 0
        epoch = 0
        step = 0
        
        injection_performed = False
        post_injection_accuracy = None
        post_injection_loss = None
        nan_weights = 0
        inf_weights = 0
        
        while global_step <= injection_global_step + self.steps_after_injection:
            # Check for early termination
            if injection_params['max_global_steps'] and global_step >= injection_params['max_global_steps']:
                record(f"Early termination at step {global_step}\n")
                break
            
            # Reset iterator at epoch boundaries
            if step >= steps_per_epoch:
                step = 0
                epoch += 1
                train_iterator = iter(train_dataset)
                train_loss.reset_state()
                train_accuracy.reset_state()
            
            if global_step == injection_global_step:
                # Perform injection
                record(f"\n🎯 Performing injection at step {global_step}\n")
                
                # Reset metrics before injection
                train_loss.reset_state()
                train_accuracy.reset_state()
                
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
                
                # Generate injection arguments using inject_utils
                if injection_params['min_val'] is not None and injection_params['max_val'] is not None:
                    record("Using random injection with min/max range\n")
                    inj_args, inj_flag = get_inj_args_with_random_range(
                        InjType[injection_params['fmodel']], None, inj_layer,
                        l_inputs, l_kernels, l_outputs, train_recorder,
                        inj_config, injection_params['min_val'], injection_params['max_val']
                    )
                elif injection_params['inj_pos'] and injection_params['inj_values']:
                    record("Using specific injection position and value\n")
                    inj_args, inj_flag = get_replay_args(
                        InjType[injection_params['fmodel']], inj_config, None, inj_layer,
                        l_inputs, l_kernels, l_outputs, train_recorder,
                        inj_pos=injection_params['inj_pos'], 
                        inj_values=injection_params['inj_values']
                    )
                else:
                    record("Using random injection\n")
                    inj_args, inj_flag = get_inj_args(
                        InjType[injection_params['fmodel']], None, inj_layer,
                        l_inputs, l_kernels, l_outputs, train_recorder, inj_config
                    )
                
                # Perform injection
                if 'fwrd' in injection_params['stage']:
                    losses, injected_layer_outputs = fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                else:
                    losses, injected_layer_outputs = bkwd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                
                # Check for NaN/Inf after injection
                for var in model.trainable_variables:
                    nan_weights += tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)).numpy()
                    inf_weights += tf.reduce_sum(tf.cast(tf.math.is_inf(var), tf.int32)).numpy()
                
                post_injection_accuracy = float(train_accuracy.result())
                post_injection_loss = float(train_loss.result())
                
                record(f"Post-injection: accuracy={post_injection_accuracy:.4f}, "
                      f"loss={post_injection_loss:.4f}, NaN={nan_weights}, Inf={inf_weights}\n")
                
                injection_performed = True
                
            else:
                # Normal training step
                try:
                    images, labels = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataset)
                    images, labels = next(train_iterator)
                
                losses = train_step(train_iterator)
            
            # Record metrics
            history['steps'].append(global_step)
            history['accuracy'].append(float(train_accuracy.result()))
            history['loss'].append(float(train_loss.result()))
            
            # Progress update
            if global_step % 50 == 0:
                record(f"Step {global_step}: accuracy={train_accuracy.result():.4f}, "
                      f"loss={train_loss.result():.4f}\n")
            
            global_step += 1
            step += 1
        
        # Close log file
        train_recorder.close()
        
        # Calculate recovery metrics
        pre_injection_acc = history['accuracy'][injection_global_step - 1] if injection_global_step > 0 else 0
        final_acc = history['accuracy'][-1] if history['accuracy'] else 0
        
        # Calculate degradation rate
        if len(history['accuracy']) > injection_global_step + 10:
            recovery_steps = history['steps'][injection_global_step + 1:]
            recovery_acc = history['accuracy'][injection_global_step + 1:]
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
            'injection_performed': injection_performed
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
        
        # Run each optimizer sequentially with the same injection
        optimizer_results = {}
        
        for optimizer_name in self.optimizers_to_test:
            try:
                result = self.train_with_injection(
                    optimizer_name,
                    injection_params,
                    experiment_dir
                )
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
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizers_to_test)))
        
        # Calculate injection step
        steps_per_epoch = math.ceil(1000 / config.BATCH_SIZE)  # Approximate
        injection_step = injection_params['target_epoch'] * steps_per_epoch + injection_params['target_step']
        
        # Plot 1: Accuracy over time
        for i, (opt_name, result) in enumerate(optimizer_results.items()):
            if 'history' in result:
                history = result['history']
                ax1.plot(history['steps'], history['accuracy'],
                        color=colors[i], label=f'{opt_name} (final: {result.get("final_accuracy", 0):.3f})',
                        linewidth=2, alpha=0.8)
        
        ax1.axvline(x=injection_step, color='red', linestyle='--', label='Injection', alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Accuracy During Training and Recovery')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss over time
        for i, (opt_name, result) in enumerate(optimizer_results.items()):
            if 'history' in result:
                history = result['history']
                ax2.plot(history['steps'], history['loss'],
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
        
        # Plot 4: Injection details
        ax4.axis('off')
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
        
        ax4.text(0.1, 0.9, injection_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
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
        print(f"Steps after injection: {self.steps_after_injection}")
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
                             optimizers_to_test=None, num_experiments=10,
                             steps_after_injection=100):
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
        optimizers_to_test=optimizers_to_test,
        num_experiments=num_experiments,
        steps_after_injection=steps_after_injection
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
                       help='Target epoch for injection (default: random 0-3)')
    parser.add_argument('--target-step', type=int, default=None,
                       help='Target step for injection (default: random 0-49)')
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
                       help='Steps to train after injection (default: 100)')
    
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
        optimizers_to_test=args.optimizers,
        num_experiments=args.num_experiments,
        steps_after_injection=args.steps_after_injection
    )
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total experiments run: {len(results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()