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
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fault_injection.models.inject_utils import choose_random_layer, InjType, get_inj_args_with_random_range
from fault_injection.models.resnet import resnet_18
from fault_injection.models.backward_resnet import backward_resnet_18
from fault_injection.core import config
from fault_injection.data.prepare_data import generate_datasets
import math

# Configure TensorFlow for CPU on MacOS
tf.config.set_visible_devices([], 'GPU')
tf.config.set_soft_device_placement(True)

class OptimizerMitigationExperiment:
    """
    Tests whether changing optimizers after fault injection can mitigate the slowdegrade effect.
    Creates 100 experiments with N16_RD fault model and compares different optimizer strategies.
    """
    
    def __init__(self, 
                 baseline_optimizer: str = 'adam',
                 test_optimizers: List[str] = None,
                 num_experiments: int = 100,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 max_steps_after_injection: int = 200):
        """
        Initialize the experiment.
        
        Args:
            baseline_optimizer: Original optimizer to use before injection
            test_optimizers: List of optimizers to test for mitigation
            num_experiments: Number of injection experiments to run
            base_seed: Base seed for reproducibility
            learning_rate: Initial learning rate
            max_steps_after_injection: Steps to train after injection
        """
        self.baseline_optimizer = baseline_optimizer
        self.test_optimizers = test_optimizers or ['sgd', 'rmsprop', 'adamw']
        self.num_experiments = num_experiments
        self.base_seed = base_seed
        self.learning_rate = learning_rate
        self.max_steps_after_injection = max_steps_after_injection
        
        # Fault injection parameters
        self.fmodel = 'N16_RD'
        self.min_val = 3.6e2
        self.max_val = 1.2e8
        self.max_target_epoch = 3
        self.max_target_step = 49
        
        # Results directory
        self.results_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "optimizer_comparison_results"
        )
        os.makedirs(self.results_base_dir, exist_ok=True)
        
    def create_optimizer(self, optimizer_name: str, learning_rate: float = None) -> tf.keras.optimizers.Optimizer:
        """
        Create an optimizer instance by name.
        
        Args:
            optimizer_name: Name of the optimizer
            learning_rate: Learning rate (uses instance default if None)
            
        Returns:
            Configured optimizer instance
        """
        lr = learning_rate or self.learning_rate
        
        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr,
            decay_steps=5000,
            end_learning_rate=0.0001
        )
        
        optimizer_map = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamw': tf.keras.optimizers.experimental.AdamW if hasattr(tf.keras.optimizers.experimental, 'AdamW') 
                     else tf.keras.optimizers.Adam,
            'adagrad': tf.keras.optimizers.Adagrad,
            'adadelta': tf.keras.optimizers.Adadelta,
            'nadam': tf.keras.optimizers.Nadam,
            'ftrl': tf.keras.optimizers.Ftrl
        }
        
        optimizer_class = optimizer_map.get(optimizer_name.lower(), tf.keras.optimizers.Adam)
        
        # Special handling for optimizers with different parameter names
        if optimizer_name.lower() == 'sgd':
            return optimizer_class(learning_rate=lr_schedule, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            return optimizer_class(learning_rate=lr_schedule, rho=0.9)
        else:
            return optimizer_class(learning_rate=lr_schedule)
    
    def generate_injection_params(self, experiment_id: int) -> Dict:
        """
        Generate random injection parameters for an experiment.
        
        Args:
            experiment_id: ID of the current experiment
            
        Returns:
            Dictionary of injection parameters
        """
        seed = self.base_seed + experiment_id
        random.seed(seed)
        np.random.seed(seed)
        
        params = {
            'seed': seed,
            'experiment_id': experiment_id,
            'model': 'resnet18',
            'stage': 'fwrd_inject',  # Focus on forward injection for clarity
            'fmodel': self.fmodel,
            'target_epoch': random.randint(0, self.max_target_epoch),
            'target_step': random.randint(0, self.max_target_step),
            'target_layer': choose_random_layer('resnet18', 'fwrd_inject'),
            'min_val': self.min_val,
            'max_val': self.max_val,
            'learning_rate': self.learning_rate
        }
        
        return params
    
    def save_checkpoint(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, 
                       checkpoint_dir: str, step: int) -> str:
        """
        Save model and optimizer checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            checkpoint_dir: Directory to save checkpoint
            step: Current training step
            
        Returns:
            Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights
        weights_path = os.path.join(checkpoint_dir, f'model_weights_step_{step}.h5')
        model.save_weights(weights_path)
        
        # Save optimizer weights
        optimizer_weights = optimizer.get_weights()
        optimizer_path = os.path.join(checkpoint_dir, f'optimizer_weights_step_{step}.npy')
        np.save(optimizer_path, optimizer_weights, allow_pickle=True)
        
        # Save optimizer config for recreation
        optimizer_config = {
            'class_name': optimizer.__class__.__name__,
            'config': optimizer.get_config()
        }
        config_path = os.path.join(checkpoint_dir, f'optimizer_config_step_{step}.json')
        with open(config_path, 'w') as f:
            json.dump(optimizer_config, f, indent=2, default=str)
        
        return weights_path
    
    def load_checkpoint(self, model: tf.keras.Model, checkpoint_dir: str, 
                       step: int, new_optimizer_name: Optional[str] = None) -> tf.keras.optimizers.Optimizer:
        """
        Load model checkpoint and optionally create a new optimizer.
        
        Args:
            model: Model to load weights into
            checkpoint_dir: Directory containing checkpoint
            step: Step number of checkpoint to load
            new_optimizer_name: Name of new optimizer to create (None to restore original)
            
        Returns:
            Optimizer instance (new or restored)
        """
        # Load model weights
        weights_path = os.path.join(checkpoint_dir, f'model_weights_step_{step}.h5')
        model.load_weights(weights_path)
        
        if new_optimizer_name:
            # Create new optimizer
            optimizer = self.create_optimizer(new_optimizer_name)
        else:
            # Restore original optimizer
            config_path = os.path.join(checkpoint_dir, f'optimizer_config_step_{step}.json')
            with open(config_path, 'r') as f:
                optimizer_config = json.load(f)
            
            # Recreate optimizer from config
            optimizer = tf.keras.optimizers.deserialize(optimizer_config)
            
            # Load optimizer weights
            optimizer_path = os.path.join(checkpoint_dir, f'optimizer_weights_step_{step}.npy')
            optimizer_weights = np.load(optimizer_path, allow_pickle=True)
            
            # Need to build optimizer by calling it once
            dummy_grads = [tf.zeros_like(w) for w in model.trainable_weights]
            optimizer.apply_gradients(zip(dummy_grads, model.trainable_weights))
            optimizer.set_weights(optimizer_weights)
        
        return optimizer
    
    def run_training_with_injection(self, injection_params: Dict, 
                                   optimizer_name: str,
                                   save_checkpoint_before_injection: bool = True) -> Dict:
        """
        Run training with fault injection and specified optimizer.
        
        Args:
            injection_params: Parameters for fault injection
            optimizer_name: Name of optimizer to use
            save_checkpoint_before_injection: Whether to save checkpoint before injection
            
        Returns:
            Dictionary containing training results and checkpoint info
        """
        # Set seeds for reproducibility
        seed = injection_params['seed']
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Get datasets
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        
        # Create model
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_resnet_18('resnet18')
        
        # Create optimizer
        model.optimizer = self.create_optimizer(optimizer_name)
        
        # Setup metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
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
            return avg_loss
        
        @tf.function
        def fwrd_inj_train_step1(iter_inputs, inj_layer):
            images, labels = iter_inputs
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
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
            
            gradients = manual_gradients + golden_gradients[-2:]  # resnet18 golden_grad_idx
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss
        
        # Training loop
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        target_epoch = injection_params['target_epoch']
        target_step = injection_params['target_step']
        
        # Tracking
        history = {
            'global_steps': [],
            'train_accuracy': [],
            'train_loss': [],
            'injection_step': None,
            'checkpoint_step': None,
            'checkpoint_path': None
        }
        
        # Training
        epoch = 0
        global_step = 0
        injection_performed = False
        checkpoint_saved = False
        steps_after_injection = 0
        
        train_iterator = iter(train_dataset)
        
        while True:
            # Check if we should stop
            if injection_performed and steps_after_injection >= self.max_steps_after_injection:
                break
            
            # Reset iterator if needed
            if global_step % steps_per_epoch == 0 and global_step > 0:
                epoch += 1
                train_iterator = iter(train_dataset)
                # Reset metrics at epoch boundary
                train_loss.reset_state()
                train_accuracy.reset_state()
            
            step_in_epoch = global_step % steps_per_epoch
            
            # Save checkpoint one step before injection
            if not checkpoint_saved and epoch == target_epoch and step_in_epoch == max(0, target_step - 1):
                if save_checkpoint_before_injection:
                    experiment_dir = os.path.join(
                        self.results_base_dir,
                        f"experiment_{injection_params['experiment_id']:03d}"
                    )
                    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
                    checkpoint_path = self.save_checkpoint(model, model.optimizer, checkpoint_dir, global_step)
                    history['checkpoint_step'] = global_step
                    history['checkpoint_path'] = checkpoint_path
                    checkpoint_saved = True
                    print(f"Checkpoint saved at step {global_step}")
            
            # Perform injection or normal training
            if not injection_performed and epoch == target_epoch and step_in_epoch == target_step:
                print(f"Performing injection at epoch {epoch}, step {step_in_epoch}")
                
                # Reset metrics before injection
                train_loss.reset_state()
                train_accuracy.reset_state()
                
                # Perform injection
                iter_inputs = next(train_iterator)
                inj_layer = injection_params['target_layer']
                
                # Get layer outputs
                l_inputs, l_kernels, l_outputs = fwrd_inj_train_step1(iter_inputs, inj_layer)
                
                # Create injection args
                class DummyRecorder:
                    def write(self, text):
                        pass
                    def flush(self):
                        pass
                
                dummy_recorder = DummyRecorder()
                
                # Create a simple config object for get_inj_args_with_random_range
                class InjConfig:
                    pass
                inj_config = InjConfig()
                
                inj_args, inj_flag = get_inj_args_with_random_range(
                    InjType[injection_params['fmodel']], None, inj_layer,
                    l_inputs, l_kernels, l_outputs, dummy_recorder,
                    inj_config, injection_params['min_val'], injection_params['max_val']
                )
                
                # Perform injection training step
                losses = fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                
                history['injection_step'] = global_step
                injection_performed = True
                print(f"Injection performed, loss: {train_loss.result():.5f}")
            else:
                # Normal training step
                losses = train_step(train_iterator)
            
            # Record metrics
            history['global_steps'].append(global_step)
            history['train_accuracy'].append(float(train_accuracy.result()))
            history['train_loss'].append(float(train_loss.result()))
            
            # Track steps after injection
            if injection_performed:
                steps_after_injection += 1
                if steps_after_injection % 50 == 0:
                    print(f"Steps after injection: {steps_after_injection}, "
                          f"accuracy: {train_accuracy.result():.4f}")
            
            global_step += 1
        
        return history
    
    def run_single_experiment(self, experiment_id: int) -> Dict:
        """
        Run a single experiment comparing baseline and test optimizers.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary containing all experiment results
        """
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {experiment_id}/{self.num_experiments}")
        print(f"{'='*70}")
        
        # Generate injection parameters
        injection_params = self.generate_injection_params(experiment_id)
        
        # Create experiment directory
        experiment_dir = os.path.join(self.results_base_dir, f"experiment_{experiment_id:03d}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save injection parameters
        params_path = os.path.join(experiment_dir, "injection_params.json")
        with open(params_path, 'w') as f:
            json.dump(injection_params, f, indent=2, default=str)
        
        print(f"Injection params: epoch={injection_params['target_epoch']}, "
              f"step={injection_params['target_step']}, layer={injection_params['target_layer']}")
        
        results = {
            'experiment_id': experiment_id,
            'injection_params': injection_params,
            'baseline': {},
            'mitigated': {}
        }
        
        # Step 1: Run baseline training with injection and save checkpoint
        print(f"\n1. Running baseline with {self.baseline_optimizer} optimizer...")
        baseline_history = self.run_training_with_injection(
            injection_params, 
            self.baseline_optimizer,
            save_checkpoint_before_injection=True
        )
        
        # Save baseline results
        baseline_dir = os.path.join(experiment_dir, f"baseline_{self.baseline_optimizer}")
        os.makedirs(baseline_dir, exist_ok=True)
        self.save_training_history(baseline_history, baseline_dir)
        results['baseline'][self.baseline_optimizer] = baseline_history
        
        # Step 2: For each test optimizer, restore from checkpoint and continue training
        checkpoint_step = baseline_history['checkpoint_step']
        
        for test_optimizer in self.test_optimizers:
            print(f"\n2. Testing mitigation with {test_optimizer} optimizer...")
            
            # Create fresh model
            seed = injection_params['seed']
            tf.random.set_seed(seed)
            model = resnet_18(seed, 'resnet18')
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            
            # Load checkpoint with new optimizer
            checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
            optimizer = self.load_checkpoint(model, checkpoint_dir, checkpoint_step, test_optimizer)
            model.optimizer = optimizer
            
            # Continue training from checkpoint with injection
            mitigated_history = self.continue_training_from_checkpoint(
                model, injection_params, checkpoint_step
            )
            
            # Save mitigated results
            mitigated_dir = os.path.join(experiment_dir, f"mitigated_{test_optimizer}")
            os.makedirs(mitigated_dir, exist_ok=True)
            self.save_training_history(mitigated_history, mitigated_dir)
            results['mitigated'][test_optimizer] = mitigated_history
        
        # Generate comparison plots
        self.create_comparison_plots(results, experiment_dir)
        
        return results
    
    def continue_training_from_checkpoint(self, model: tf.keras.Model, 
                                         injection_params: Dict,
                                         checkpoint_step: int) -> Dict:
        """
        Continue training from checkpoint with fault injection.
        
        Args:
            model: Model with loaded weights and new optimizer
            injection_params: Original injection parameters
            checkpoint_step: Step where checkpoint was saved
            
        Returns:
            Training history dictionary
        """
        # Set seeds
        seed = injection_params['seed']
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Get datasets
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        
        # Create backward model
        back_model = backward_resnet_18('resnet18')
        
        # Setup metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        # Training functions (same as before)
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
            return avg_loss
        
        @tf.function
        def fwrd_inj_train_step1(iter_inputs, inj_layer):
            images, labels = iter_inputs
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
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
            
            gradients = manual_gradients + golden_gradients[-2:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            return avg_loss
        
        # Calculate injection global step
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        target_epoch = injection_params['target_epoch']
        target_step = injection_params['target_step']
        injection_global_step = target_epoch * steps_per_epoch + target_step
        
        # Fast-forward dataset to checkpoint position
        train_iterator = iter(train_dataset)
        for _ in range(checkpoint_step + 1):
            next(train_iterator)
        
        # Tracking
        history = {
            'global_steps': [],
            'train_accuracy': [],
            'train_loss': [],
            'injection_step': injection_global_step,
            'checkpoint_step': checkpoint_step
        }
        
        # Continue training
        global_step = checkpoint_step + 1
        injection_performed = False
        steps_after_injection = 0
        
        while steps_after_injection < self.max_steps_after_injection:
            # Reset iterator if needed
            if global_step % steps_per_epoch == 0:
                train_iterator = iter(train_dataset)
            
            # Perform injection at the right step
            if global_step == injection_global_step:
                print(f"Performing injection at restored step {global_step}")
                
                # Reset metrics
                train_loss.reset_state()
                train_accuracy.reset_state()
                
                # Perform injection
                iter_inputs = next(train_iterator)
                inj_layer = injection_params['target_layer']
                
                l_inputs, l_kernels, l_outputs = fwrd_inj_train_step1(iter_inputs, inj_layer)
                
                class DummyRecorder:
                    def write(self, text):
                        pass
                    def flush(self):
                        pass
                
                dummy_recorder = DummyRecorder()
                
                class InjConfig:
                    pass
                inj_config = InjConfig()
                
                inj_args, inj_flag = get_inj_args_with_random_range(
                    InjType[injection_params['fmodel']], None, inj_layer,
                    l_inputs, l_kernels, l_outputs, dummy_recorder,
                    inj_config, injection_params['min_val'], injection_params['max_val']
                )
                
                losses = fwrd_inj_train_step2(iter_inputs, inj_args, inj_flag)
                injection_performed = True
            else:
                # Normal training step
                losses = train_step(train_iterator)
            
            # Record metrics
            history['global_steps'].append(global_step)
            history['train_accuracy'].append(float(train_accuracy.result()))
            history['train_loss'].append(float(train_loss.result()))
            
            # Track steps after injection
            if injection_performed:
                steps_after_injection += 1
            
            global_step += 1
        
        return history
    
    def save_training_history(self, history: Dict, output_dir: str):
        """
        Save training history to CSV and JSON files.
        
        Args:
            history: Training history dictionary
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "training_log.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['global_step', 'train_accuracy', 'train_loss'])
            for i in range(len(history['global_steps'])):
                writer.writerow([
                    history['global_steps'][i],
                    history['train_accuracy'][i],
                    history['train_loss'][i]
                ])
        
        # Save metadata as JSON
        metadata = {
            'injection_step': history['injection_step'],
            'checkpoint_step': history.get('checkpoint_step'),
            'final_accuracy': history['train_accuracy'][-1] if history['train_accuracy'] else None,
            'final_loss': history['train_loss'][-1] if history['train_loss'] else None
        }
        json_path = os.path.join(output_dir, "metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_comparison_plots(self, results: Dict, output_dir: str):
        """
        Create comparison plots for baseline vs mitigated training.
        
        Args:
            results: Experiment results dictionary
            output_dir: Directory to save plots
        """
        # Create figure with subplots for each optimizer
        n_optimizers = len(self.test_optimizers) + 1  # +1 for baseline
        fig, axes = plt.subplots(1, n_optimizers, figsize=(6*n_optimizers, 5))
        
        if n_optimizers == 1:
            axes = [axes]
        
        # Plot baseline
        baseline_opt = self.baseline_optimizer
        baseline_history = results['baseline'][baseline_opt]
        
        ax = axes[0]
        ax.plot(baseline_history['global_steps'], baseline_history['train_accuracy'], 
                label=f'Baseline ({baseline_opt})', linewidth=2)
        ax.axvline(x=baseline_history['injection_step'], color='red', 
                  linestyle='--', label='Injection', alpha=0.7)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Training Accuracy')
        ax.set_title(f'Baseline: {baseline_opt}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot each test optimizer
        for i, test_opt in enumerate(self.test_optimizers, 1):
            if test_opt in results['mitigated']:
                ax = axes[i]
                mitigated_history = results['mitigated'][test_opt]
                
                # Plot baseline for comparison
                ax.plot(baseline_history['global_steps'], baseline_history['train_accuracy'], 
                       label=f'Baseline ({baseline_opt})', linewidth=1, alpha=0.5)
                
                # Plot mitigated
                ax.plot(mitigated_history['global_steps'], mitigated_history['train_accuracy'], 
                       label=f'Mitigated ({test_opt})', linewidth=2)
                
                ax.axvline(x=mitigated_history['injection_step'], color='red', 
                          linestyle='--', label='Injection', alpha=0.7)
                ax.set_xlabel('Global Step')
                ax.set_ylabel('Training Accuracy')
                ax.set_title(f'Mitigated: {test_opt}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "comparison_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create degradation analysis plot
        self.create_degradation_analysis(results, output_dir)
    
    def create_degradation_analysis(self, results: Dict, output_dir: str):
        """
        Analyze and plot degradation rates for different optimizers.
        
        Args:
            results: Experiment results dictionary
            output_dir: Directory to save analysis
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate degradation rates
        degradation_rates = {}
        
        # Baseline degradation
        baseline_opt = self.baseline_optimizer
        baseline_history = results['baseline'][baseline_opt]
        injection_idx = baseline_history['global_steps'].index(baseline_history['injection_step'])
        
        if injection_idx < len(baseline_history['train_accuracy']) - 10:
            pre_injection_acc = baseline_history['train_accuracy'][injection_idx]
            post_injection_acc = baseline_history['train_accuracy'][-1]
            steps_after = len(baseline_history['train_accuracy']) - injection_idx
            degradation_rate = (pre_injection_acc - post_injection_acc) / steps_after * 100
            degradation_rates[f'Baseline ({baseline_opt})'] = degradation_rate
        
        # Test optimizer degradation
        for test_opt in self.test_optimizers:
            if test_opt in results['mitigated']:
                mitigated_history = results['mitigated'][test_opt]
                injection_idx = mitigated_history['global_steps'].index(mitigated_history['injection_step'])
                
                if injection_idx < len(mitigated_history['train_accuracy']) - 10:
                    pre_injection_acc = mitigated_history['train_accuracy'][injection_idx]
                    post_injection_acc = mitigated_history['train_accuracy'][-1]
                    steps_after = len(mitigated_history['train_accuracy']) - injection_idx
                    degradation_rate = (pre_injection_acc - post_injection_acc) / steps_after * 100
                    degradation_rates[f'Mitigated ({test_opt})'] = degradation_rate
        
        # Plot degradation rates
        if degradation_rates:
            ax1.bar(range(len(degradation_rates)), list(degradation_rates.values()))
            ax1.set_xticks(range(len(degradation_rates)))
            ax1.set_xticklabels(list(degradation_rates.keys()), rotation=45, ha='right')
            ax1.set_ylabel('Degradation Rate (% per 100 steps)')
            ax1.set_title('Accuracy Degradation Rate After Injection')
            ax1.grid(True, alpha=0.3)
        
        # Plot final accuracies
        final_accuracies = {}
        final_accuracies[f'Baseline ({baseline_opt})'] = baseline_history['train_accuracy'][-1]
        
        for test_opt in self.test_optimizers:
            if test_opt in results['mitigated']:
                mitigated_history = results['mitigated'][test_opt]
                final_accuracies[f'Mitigated ({test_opt})'] = mitigated_history['train_accuracy'][-1]
        
        ax2.bar(range(len(final_accuracies)), list(final_accuracies.values()))
        ax2.set_xticks(range(len(final_accuracies)))
        ax2.set_xticklabels(list(final_accuracies.keys()), rotation=45, ha='right')
        ax2.set_ylabel('Final Training Accuracy')
        ax2.set_title('Final Accuracy After 200 Steps')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_path = os.path.join(output_dir, "degradation_analysis.png")
        plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save numerical analysis
        analysis_data = {
            'degradation_rates': degradation_rates,
            'final_accuracies': final_accuracies
        }
        analysis_json_path = os.path.join(output_dir, "degradation_analysis.json")
        with open(analysis_json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    def run(self) -> List[Dict]:
        """
        Run all experiments.
        
        Returns:
            List of all experiment results
        """
        print(f"Starting Optimizer Mitigation Experiments")
        print(f"Baseline optimizer: {self.baseline_optimizer}")
        print(f"Test optimizers: {self.test_optimizers}")
        print(f"Number of experiments: {self.num_experiments}")
        print(f"Max steps after injection: {self.max_steps_after_injection}")
        print(f"Results directory: {self.results_base_dir}")
        
        all_results = []
        
        for exp_id in range(1, self.num_experiments + 1):
            try:
                results = self.run_single_experiment(exp_id)
                all_results.append(results)
                
                # Save intermediate summary
                self.save_summary(all_results)
                
            except Exception as e:
                print(f"Error in experiment {exp_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate final summary
        self.generate_final_report(all_results)
        
        return all_results
    
    def save_summary(self, results: List[Dict]):
        """
        Save summary of results so far.
        
        Args:
            results: List of experiment results
        """
        summary_path = os.path.join(self.results_base_dir, "summary.json")
        
        summary = {
            'total_experiments': len(results),
            'baseline_optimizer': self.baseline_optimizer,
            'test_optimizers': self.test_optimizers,
            'experiments': []
        }
        
        for result in results:
            exp_summary = {
                'experiment_id': result['experiment_id'],
                'target_layer': result['injection_params']['target_layer'],
                'target_epoch': result['injection_params']['target_epoch'],
                'target_step': result['injection_params']['target_step'],
                'baseline_final_acc': result['baseline'][self.baseline_optimizer]['train_accuracy'][-1],
                'mitigated_final_acc': {}
            }
            
            for opt in self.test_optimizers:
                if opt in result['mitigated']:
                    exp_summary['mitigated_final_acc'][opt] = result['mitigated'][opt]['train_accuracy'][-1]
            
            summary['experiments'].append(exp_summary)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_final_report(self, results: List[Dict]):
        """
        Generate final report with statistics and conclusions.
        
        Args:
            results: List of all experiment results
        """
        report_path = os.path.join(self.results_base_dir, "final_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("OPTIMIZER MITIGATION EXPERIMENT FINAL REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total experiments completed: {len(results)}\n")
            f.write(f"Baseline optimizer: {self.baseline_optimizer}\n")
            f.write(f"Test optimizers: {', '.join(self.test_optimizers)}\n")
            f.write(f"Fault model: {self.fmodel}\n")
            f.write(f"Injection value range: [{self.min_val:.2e}, {self.max_val:.2e}]\n")
            f.write(f"Steps after injection: {self.max_steps_after_injection}\n\n")
            
            # Calculate statistics
            baseline_improvements = {opt: [] for opt in self.test_optimizers}
            
            for result in results:
                baseline_final = result['baseline'][self.baseline_optimizer]['train_accuracy'][-1]
                
                for opt in self.test_optimizers:
                    if opt in result['mitigated']:
                        mitigated_final = result['mitigated'][opt]['train_accuracy'][-1]
                        improvement = mitigated_final - baseline_final
                        baseline_improvements[opt].append(improvement)
            
            f.write("IMPROVEMENT STATISTICS (vs baseline):\n")
            f.write("-"*40 + "\n")
            
            for opt in self.test_optimizers:
                if baseline_improvements[opt]:
                    improvements = baseline_improvements[opt]
                    mean_improvement = np.mean(improvements)
                    std_improvement = np.std(improvements)
                    positive_cases = sum(1 for x in improvements if x > 0)
                    
                    f.write(f"\n{opt.upper()}:\n")
                    f.write(f"  Mean improvement: {mean_improvement:.4f} ± {std_improvement:.4f}\n")
                    f.write(f"  Positive improvements: {positive_cases}/{len(improvements)} "
                           f"({positive_cases/len(improvements)*100:.1f}%)\n")
                    f.write(f"  Best improvement: {max(improvements):.4f}\n")
                    f.write(f"  Worst degradation: {min(improvements):.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("CONCLUSIONS:\n")
            f.write("-"*40 + "\n")
            
            # Determine best optimizer
            mean_improvements = {}
            for opt in self.test_optimizers:
                if baseline_improvements[opt]:
                    mean_improvements[opt] = np.mean(baseline_improvements[opt])
            
            if mean_improvements:
                best_optimizer = max(mean_improvements, key=mean_improvements.get)
                best_improvement = mean_improvements[best_optimizer]
                
                if best_improvement > 0:
                    f.write(f"✓ Best mitigation strategy: Switch to {best_optimizer} "
                           f"(avg improvement: {best_improvement:.4f})\n")
                else:
                    f.write(f"✗ No optimizer showed consistent improvement over baseline\n")
                
                f.write(f"\nDetailed rankings:\n")
                for opt, imp in sorted(mean_improvements.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {opt}: {imp:+.4f}\n")
        
        print(f"\nFinal report saved to: {report_path}")


def main():
    """
    Main function to run the experiment.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test optimizer mitigation for slowdegrade effects')
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
    
    # Create and run experiment
    experiment = OptimizerMitigationExperiment(
        baseline_optimizer=args.baseline,
        test_optimizers=args.test_optimizers,
        num_experiments=args.num_experiments,
        base_seed=args.seed,
        learning_rate=args.learning_rate,
        max_steps_after_injection=args.steps_after_injection
    )
    
    results = experiment.run()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {experiment.results_base_dir}")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()