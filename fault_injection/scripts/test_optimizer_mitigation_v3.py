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

class ParallelOptimizerMitigationExperiment:
    """
    VERSION 3: Tests optimizer mitigation by training ALL optimizers from scratch
    for each experiment, ensuring fair comparison with full training context.
    
    Key improvements over V2:
    1. All optimizers train from scratch to the injection point
    2. Identical injection applied to all models simultaneously
    3. Direct comparison with same training history for all optimizers
    4. Simplified single-phase execution per experiment
    """
    
    def __init__(self, 
                 optimizers_to_test: List[str] = None,
                 num_experiments: int = 10,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 steps_after_injection: int = 100):
        """
        Initialize the parallel optimizer experiment.
        
        Args:
            optimizers_to_test: List of optimizer names to compare
            num_experiments: Number of experiments to run
            base_seed: Base random seed for reproducibility
            learning_rate: Initial learning rate for all optimizers
            steps_after_injection: Steps to continue after injection (default 100)
        """
        self.optimizers_to_test = optimizers_to_test or ['adam', 'sgd', 'rmsprop', 'adamw']
        self.num_experiments = num_experiments
        self.base_seed = base_seed
        self.learning_rate = learning_rate
        self.steps_after_injection = steps_after_injection
        
        # Fault injection parameters
        self.fmodel = 'N16_RD'
        self.min_val = 3.6e2
        self.max_val = 1.2e8
        self.max_target_epoch = 3
        self.max_target_step = 49
        
        # Results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            f"parallel_optimizer_results_{timestamp}"
        )
        os.makedirs(self.results_base_dir, exist_ok=True)
        
        # Pre-generate all injection configurations
        self.injection_configs = self._pre_generate_injection_configs()
        
    def _pre_generate_injection_configs(self) -> List[Dict]:
        """
        Pre-generate all injection configurations to ensure reproducibility.
        """
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
            
            config = {
                'experiment_id': exp_id,
                'seed': seed,
                'model': 'resnet18',
                'stage': 'fwrd_inject',
                'fmodel': self.fmodel,
                'target_epoch': random.randint(0, self.max_target_epoch),
                'target_step': random.randint(0, self.max_target_step),
                'target_layer': choose_random_layer('resnet18', 'fwrd_inject'),
                'injection_position': injection_position,
                'injection_value': float(injection_value),
                'learning_rate': self.learning_rate
            }
            
            configs.append(config)
        
        # Save configs for reference
        configs_path = os.path.join(self.results_base_dir, 'all_injection_configs.json')
        with open(configs_path, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        
        print(f"Pre-generated {len(configs)} injection configurations")
        return configs
    
    def create_optimizer(self, optimizer_name: str, learning_rate: float = None) -> tf.keras.optimizers.Optimizer:
        """
        Create optimizer with specified learning rate schedule.
        """
        lr = learning_rate or self.learning_rate
        
        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr,
            decay_steps=5000,
            end_learning_rate=0.0001,
            power=1.0
        )
        
        # Handle different TensorFlow/Keras versions for AdamW
        adamw_optimizer = tf.keras.optimizers.Adam  # Default fallback
        try:
            if hasattr(tf.keras.optimizers, 'AdamW'):
                adamw_optimizer = tf.keras.optimizers.AdamW
            elif hasattr(tf.keras, 'optimizers'):
                experimental = getattr(tf.keras.optimizers, 'experimental', None)
                if experimental and hasattr(experimental, 'AdamW'):
                    adamw_optimizer = experimental.AdamW
        except Exception:
            pass
        
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
        
        # Create optimizer with appropriate parameters
        if optimizer_name.lower() == 'sgd':
            return optimizer_class(learning_rate=lr_schedule, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            return optimizer_class(learning_rate=lr_schedule, rho=0.9)
        elif optimizer_name.lower() == 'adadelta':
            return optimizer_class(learning_rate=lr_schedule, rho=0.95)
        else:
            return optimizer_class(learning_rate=lr_schedule)
    
    def create_models_and_optimizers(self, seed: int) -> Dict:
        """
        Create all models and optimizers for parallel training.
        
        Returns:
            Dictionary with optimizer names as keys, containing models and optimizers
        """
        models_dict = {}
        
        for optimizer_name in self.optimizers_to_test:
            # Create model
            model = resnet_18(seed, f'resnet18_{optimizer_name}')
            model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
            
            # Create backward model
            back_model = backward_resnet_18(f'resnet18_{optimizer_name}')
            
            # Create optimizer
            optimizer = self.create_optimizer(optimizer_name)
            model.optimizer = optimizer
            
            # Setup metrics
            train_loss = tf.keras.metrics.Mean(name=f'train_loss_{optimizer_name}')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name=f'train_accuracy_{optimizer_name}')
            
            models_dict[optimizer_name] = {
                'model': model,
                'back_model': back_model,
                'optimizer': optimizer,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'history': {
                    'steps': [],
                    'accuracy': [],
                    'loss': []
                }
            }
        
        return models_dict
    
    def perform_injection(self, models_dict: Dict, images: tf.Tensor, labels: tf.Tensor,
                         injection_config: Dict) -> Dict:
        """
        Perform identical injection on all models.
        
        Returns:
            Dictionary of injection results for each optimizer
        """
        injection_results = {}
        
        inj_layer = injection_config['target_layer']
        inj_position = injection_config['injection_position']
        inj_value = injection_config['injection_value']
        
        for optimizer_name, model_info in models_dict.items():
            model = model_info['model']
            back_model = model_info['back_model']
            train_loss = model_info['train_loss']
            train_accuracy = model_info['train_accuracy']
            
            # Reset metrics before injection
            train_loss.reset_state()
            train_accuracy.reset_state()
            
            with tf.GradientTape() as tape:
                # Get layer outputs for injection
                outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
                
                # Extract specific layer outputs
                layer_inputs = l_inputs[inj_layer]
                layer_kernels = l_kernels[inj_layer]
                layer_outputs = l_outputs[inj_layer]
                
                # Create injection configuration
                class DummyRecorder:
                    def write(self, text):
                        pass
                    def flush(self):
                        pass
                
                dummy_recorder = DummyRecorder()
                
                class InjectionConfig:
                    def __init__(self):
                        self.inj_pos = [inj_position]
                        self.inj_values = [inj_value]
                
                inj_config = InjectionConfig()
                
                # Create injection arguments
                inj_args, inj_flag = get_inj_args_with_random_range(
                    InjType[injection_config['fmodel']], 
                    None,
                    inj_layer,
                    layer_inputs, 
                    layer_kernels, 
                    layer_outputs, 
                    dummy_recorder,
                    inj_config,
                    inj_value,
                    inj_value
                )
                
                # Perform forward pass with injection
                outputs, l_inputs_inj, l_kernels_inj, l_outputs_inj = model(
                    images, training=True, inject=inj_flag, inj_args=inj_args
                )
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            
            # Backward pass
            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs_inj, l_kernels_inj, 
                                                        inject=False, inj_args=None)
            
            gradients = manual_gradients + golden_gradients[-2:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            
            # Check for NaN/Inf
            nan_count = 0
            inf_count = 0
            for var in model.trainable_variables:
                nan_count += tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)).numpy()
                inf_count += tf.reduce_sum(tf.cast(tf.math.is_inf(var), tf.int32)).numpy()
            
            injection_results[optimizer_name] = {
                'post_injection_accuracy': float(train_accuracy.result()),
                'post_injection_loss': float(train_loss.result()),
                'nan_weights': int(nan_count),
                'inf_weights': int(inf_count)
            }
        
        return injection_results
    
    def run_single_experiment(self, experiment_id: int) -> Dict:
        """
        Run a single experiment with all optimizers training in parallel.
        """
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id + 1}/{self.num_experiments}")
        print(f"{'='*80}")
        
        # Get injection configuration
        injection_config = self.injection_configs[experiment_id]
        seed = injection_config['seed']
        
        # Set seeds
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create experiment directory
        experiment_dir = os.path.join(self.results_base_dir, f"experiment_{experiment_id:03d}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save injection config
        config_path = os.path.join(experiment_dir, "injection_config.json")
        with open(config_path, 'w') as f:
            json.dump(injection_config, f, indent=2, default=str)
        
        # Get datasets
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        
        # Create all models and optimizers
        models_dict = self.create_models_and_optimizers(seed)
        
        # Calculate injection point
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        target_epoch = injection_config['target_epoch']
        target_step = injection_config['target_step']
        injection_global_step = target_epoch * steps_per_epoch + target_step
        
        print(f"Training all optimizers until injection at epoch {target_epoch}, step {target_step}")
        print(f"Total steps to injection: {injection_global_step}")
        
        # Training functions for each optimizer
        def create_train_step(model_info):
            @tf.function
            def train_step(images, labels):
                model = model_info['model']
                with tf.GradientTape() as tape:
                    outputs, _, _, _ = model(images, training=True, inject=False)
                    predictions = outputs['logits']
                    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                    avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
                
                tvars = model.trainable_variables
                gradients = tape.gradient(avg_loss, tvars)
                model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))
                
                model_info['train_loss'].update_state(avg_loss)
                model_info['train_accuracy'].update_state(labels, predictions)
                return avg_loss
            return train_step
        
        # Create train steps for all models
        train_steps = {opt: create_train_step(models_dict[opt]) for opt in self.optimizers_to_test}
        
        # Train all models until injection point
        train_iterator = iter(train_dataset)
        
        for global_step in range(injection_global_step):
            # Reset iterator at epoch boundaries
            if global_step > 0 and global_step % steps_per_epoch == 0:
                train_iterator = iter(train_dataset)
                for model_info in models_dict.values():
                    model_info['train_loss'].reset_state()
                    model_info['train_accuracy'].reset_state()
            
            # Get batch
            images, labels = next(train_iterator)
            
            # Train all models on the same batch
            for optimizer_name in self.optimizers_to_test:
                loss = train_steps[optimizer_name](images, labels)
                
                # Record metrics
                models_dict[optimizer_name]['history']['steps'].append(global_step)
                models_dict[optimizer_name]['history']['accuracy'].append(
                    float(models_dict[optimizer_name]['train_accuracy'].result())
                )
                models_dict[optimizer_name]['history']['loss'].append(
                    float(models_dict[optimizer_name]['train_loss'].result())
                )
            
            # Progress update
            if global_step % 50 == 0:
                print(f"Step {global_step}/{injection_global_step}:")
                for opt_name in self.optimizers_to_test:
                    acc = models_dict[opt_name]['train_accuracy'].result()
                    print(f"  {opt_name}: accuracy={acc:.4f}")
        
        # Perform injection on all models
        print(f"\n🎯 Injecting fault at step {injection_global_step}")
        print(f"  Layer: {injection_config['target_layer']}")
        print(f"  Position: {injection_config['injection_position']}")
        print(f"  Value: {injection_config['injection_value']:.2e}")
        
        # Get batch for injection
        images, labels = next(train_iterator)
        
        # Save batch data for reproduction
        batch_data_path = os.path.join(experiment_dir, 'injection_batch.npz')
        np.savez(batch_data_path, images=images.numpy(), labels=labels.numpy())
        
        # Perform injection on all models
        injection_results = self.perform_injection(models_dict, images, labels, injection_config)
        
        print("\nPost-injection state:")
        for opt_name, results in injection_results.items():
            print(f"  {opt_name}: accuracy={results['post_injection_accuracy']:.4f}, "
                  f"loss={results['post_injection_loss']:.4f}, "
                  f"NaN={results['nan_weights']}, Inf={results['inf_weights']}")
        
        # Continue training for specified steps after injection
        print(f"\nContinuing training for {self.steps_after_injection} steps...")
        
        for step in range(self.steps_after_injection):
            global_step = injection_global_step + 1 + step
            
            # Reset iterator at epoch boundaries
            if global_step % steps_per_epoch == 0:
                train_iterator = iter(train_dataset)
            
            # Get batch
            try:
                images, labels = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataset)
                images, labels = next(train_iterator)
            
            # Train all models
            for optimizer_name in self.optimizers_to_test:
                model_info = models_dict[optimizer_name]
                
                # Check if already diverged
                if 'diverged' in model_info and model_info['diverged']:
                    continue
                
                loss = train_steps[optimizer_name](images, labels)
                
                # Check for divergence
                if not tf.math.is_finite(loss):
                    print(f"  ⚠️ {optimizer_name} diverged at step {step}")
                    model_info['diverged'] = True
                    model_info['divergence_step'] = step
                    continue
                
                # Record metrics
                model_info['history']['steps'].append(global_step)
                model_info['history']['accuracy'].append(
                    float(model_info['train_accuracy'].result())
                )
                model_info['history']['loss'].append(
                    float(model_info['train_loss'].result())
                )
            
            # Progress update
            if step % 20 == 0 or step == self.steps_after_injection - 1:
                print(f"Step {step + 1}/{self.steps_after_injection}:")
                for opt_name in self.optimizers_to_test:
                    if 'diverged' not in models_dict[opt_name] or not models_dict[opt_name]['diverged']:
                        acc = models_dict[opt_name]['train_accuracy'].result()
                        loss = models_dict[opt_name]['train_loss'].result()
                        print(f"  {opt_name}: accuracy={acc:.4f}, loss={loss:.4f}")
        
        # Compile results
        experiment_results = {
            'experiment_id': experiment_id,
            'injection_config': injection_config,
            'injection_results': injection_results,
            'optimizer_results': {}
        }
        
        for optimizer_name in self.optimizers_to_test:
            model_info = models_dict[optimizer_name]
            history = model_info['history']
            
            # Calculate recovery metrics
            pre_injection_acc = history['accuracy'][injection_global_step - 1] if injection_global_step > 0 else 0
            post_injection_acc = injection_results[optimizer_name]['post_injection_accuracy']
            final_acc = history['accuracy'][-1] if history['accuracy'] else 0
            
            # Calculate degradation rate (slope over recovery period)
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
            
            experiment_results['optimizer_results'][optimizer_name] = {
                'history': history,
                'pre_injection_accuracy': pre_injection_acc,
                'post_injection_accuracy': post_injection_acc,
                'final_accuracy': final_acc,
                'accuracy_change': final_acc - post_injection_acc,
                'total_recovery': final_acc - pre_injection_acc,
                'degradation_rate': degradation_rate,
                'diverged': model_info.get('diverged', False),
                'divergence_step': model_info.get('divergence_step', None)
            }
        
        # Save results
        self.save_experiment_results(experiment_results, experiment_dir)
        
        # Create visualizations
        self.create_experiment_visualizations(experiment_results, experiment_dir)
        
        return experiment_results
    
    def save_experiment_results(self, results: Dict, experiment_dir: str):
        """Save experiment results to files."""
        # Save JSON results
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
        
        results_json = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save CSV for each optimizer
        for optimizer_name, opt_results in results['optimizer_results'].items():
            csv_path = os.path.join(experiment_dir, f'history_{optimizer_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'accuracy', 'loss'])
                history = opt_results['history']
                for i in range(len(history['steps'])):
                    writer.writerow([
                        history['steps'][i],
                        history['accuracy'][i],
                        history['loss'][i]
                    ])
    
    def create_experiment_visualizations(self, results: Dict, experiment_dir: str):
        """Create visualization plots for the experiment."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        injection_step = results['injection_config']['target_epoch'] * \
                        math.ceil(1000 / config.BATCH_SIZE) + \
                        results['injection_config']['target_step']
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizers_to_test)))
        
        # Plot 1: Accuracy over time
        for i, optimizer_name in enumerate(self.optimizers_to_test):
            opt_results = results['optimizer_results'][optimizer_name]
            history = opt_results['history']
            
            ax1.plot(history['steps'], history['accuracy'],
                    color=colors[i], label=f'{optimizer_name} (final: {opt_results["final_accuracy"]:.3f})',
                    linewidth=2, alpha=0.8)
        
        ax1.axvline(x=injection_step, color='red', linestyle='--',
                   label='Injection', alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title('Accuracy During Training and Recovery')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss over time
        for i, optimizer_name in enumerate(self.optimizers_to_test):
            opt_results = results['optimizer_results'][optimizer_name]
            history = opt_results['history']
            
            ax2.plot(history['steps'], history['loss'],
                    color=colors[i], label=optimizer_name,
                    linewidth=2, alpha=0.8)
        
        ax2.axvline(x=injection_step, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Training Loss')
        ax2.set_title('Loss During Training and Recovery')
        ax2.set_yscale('log')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Recovery comparison bar chart
        optimizer_names = list(results['optimizer_results'].keys())
        final_accuracies = [r['final_accuracy'] for r in results['optimizer_results'].values()]
        accuracy_changes = [r['accuracy_change'] for r in results['optimizer_results'].values()]
        
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
        
        # Plot 4: Degradation rates
        degradation_rates = [r['degradation_rate'] * 1000 for r in results['optimizer_results'].values()]
        
        bars = ax4.bar(optimizer_names, degradation_rates, alpha=0.7)
        
        # Color based on rate
        for bar, rate in zip(bars, degradation_rates):
            if rate > 0:
                bar.set_color('green')  # Improving
            elif rate < -0.1:
                bar.set_color('red')    # Degrading
            else:
                bar.set_color('yellow') # Stable
        
        # Add value labels
        for bar, rate in zip(bars, degradation_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        ax4.set_ylabel('Degradation Rate (×1000 accuracy/step)')
        ax4.set_title('Accuracy Change Rate During Recovery\n(Positive = Improving, Negative = Degrading)')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = os.path.join(experiment_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create zoomed recovery plot
        self.create_recovery_zoom_plot(results, experiment_dir)
    
    def create_recovery_zoom_plot(self, results: Dict, experiment_dir: str):
        """Create zoomed plot focusing on recovery period."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        injection_step = results['injection_config']['target_epoch'] * \
                        math.ceil(1000 / config.BATCH_SIZE) + \
                        results['injection_config']['target_step']
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimizers_to_test)))
        
        # Plot only recovery period
        for i, optimizer_name in enumerate(self.optimizers_to_test):
            opt_results = results['optimizer_results'][optimizer_name]
            history = opt_results['history']
            
            # Get recovery period data
            recovery_indices = [j for j, step in enumerate(history['steps']) 
                              if step >= injection_step - 10]
            
            if recovery_indices:
                recovery_steps = [history['steps'][j] for j in recovery_indices]
                recovery_acc = [history['accuracy'][j] for j in recovery_indices]
                
                ax.plot(recovery_steps, recovery_acc,
                       color=colors[i], label=optimizer_name,
                       linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax.axvline(x=injection_step, color='red', linestyle='--',
                  label='Injection Point', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Training Accuracy')
        ax.set_title('Recovery Period Detail (Zoomed)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(experiment_dir, 'recovery_zoom.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self) -> List[Dict]:
        """Run all experiments."""
        print(f"\n{'='*80}")
        print(f"Starting Parallel Optimizer Mitigation Experiments")
        print(f"{'='*80}")
        print(f"Optimizers to test: {self.optimizers_to_test}")
        print(f"Number of experiments: {self.num_experiments}")
        print(f"Steps after injection: {self.steps_after_injection}")
        print(f"Results directory: {self.results_base_dir}")
        
        all_results = []
        successful_experiments = 0
        
        for exp_id in range(self.num_experiments):
            try:
                results = self.run_single_experiment(exp_id)
                all_results.append(results)
                successful_experiments += 1
                
                # Save intermediate summary every 5 experiments
                if (exp_id + 1) % 5 == 0:
                    self.save_intermediate_summary(all_results)
                
            except Exception as e:
                print(f"\n❌ Error in experiment {exp_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Save error info
                error_info = {
                    'experiment_id': exp_id,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                error_path = os.path.join(self.results_base_dir, f'error_exp_{exp_id:03d}.json')
                with open(error_path, 'w') as f:
                    json.dump(error_info, f, indent=2)
        
        # Generate final report
        self.generate_final_report(all_results)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENTS COMPLETE")
        print(f"Successful: {successful_experiments}/{self.num_experiments}")
        print(f"Results saved to: {self.results_base_dir}")
        print(f"{'='*80}")
        
        return all_results
    
    def save_intermediate_summary(self, results: List[Dict]):
        """Save intermediate summary of results."""
        summary = {
            'completed_experiments': len(results),
            'optimizers_tested': self.optimizers_to_test,
            'aggregate_metrics': {}
        }
        
        # Calculate aggregate metrics for each optimizer
        for optimizer in self.optimizers_to_test:
            final_accs = []
            acc_changes = []
            degradation_rates = []
            divergence_count = 0
            
            for result in results:
                opt_result = result['optimizer_results'][optimizer]
                final_accs.append(opt_result['final_accuracy'])
                acc_changes.append(opt_result['accuracy_change'])
                degradation_rates.append(opt_result.get('degradation_rate', 0))
                if opt_result.get('diverged', False):
                    divergence_count += 1
            
            summary['aggregate_metrics'][optimizer] = {
                'mean_final_accuracy': float(np.mean(final_accs)),
                'std_final_accuracy': float(np.std(final_accs)),
                'mean_accuracy_change': float(np.mean(acc_changes)),
                'std_accuracy_change': float(np.std(acc_changes)),
                'mean_degradation_rate': float(np.mean(degradation_rates)),
                'positive_recovery_rate': float(sum(1 for x in acc_changes if x > 0) / len(acc_changes)),
                'divergence_rate': float(divergence_count / len(results))
            }
        
        summary_path = os.path.join(self.results_base_dir, 'intermediate_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_final_report(self, results: List[Dict]):
        """Generate comprehensive final report."""
        if not results:
            print("No results to report")
            return
        
        report_path = os.path.join(self.results_base_dir, 'final_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Parallel Optimizer Mitigation Experiment - Final Report\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Experiments Completed**: {len(results)}/{self.num_experiments}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Optimizers Tested**: {', '.join(self.optimizers_to_test)}\n")
            f.write(f"- **Fault Model**: {self.fmodel}\n")
            f.write(f"- **Injection Value Range**: [{self.min_val:.2e}, {self.max_val:.2e}]\n")
            f.write(f"- **Max Target Epoch**: {self.max_target_epoch}\n")
            f.write(f"- **Max Target Step**: {self.max_target_step}\n")
            f.write(f"- **Steps After Injection**: {self.steps_after_injection}\n\n")
            
            # Aggregate statistics
            f.write("## Aggregate Results\n\n")
            
            # Table header
            f.write("| Optimizer | Mean Final Acc | Std Final Acc | Mean Acc Change | Positive Recovery % | Mean Degrad. Rate | Divergence % |\n")
            f.write("|-----------|---------------|---------------|-----------------|-------------------|------------------|-------------|\n")
            
            best_mean_change = -float('inf')
            best_optimizer = None
            
            for optimizer in self.optimizers_to_test:
                final_accs = []
                acc_changes = []
                degradation_rates = []
                divergence_count = 0
                
                for result in results:
                    opt_result = result['optimizer_results'][optimizer]
                    final_accs.append(opt_result['final_accuracy'])
                    acc_changes.append(opt_result['accuracy_change'])
                    degradation_rates.append(opt_result.get('degradation_rate', 0))
                    if opt_result.get('diverged', False):
                        divergence_count += 1
                
                mean_final = np.mean(final_accs)
                std_final = np.std(final_accs)
                mean_change = np.mean(acc_changes)
                positive_rate = sum(1 for x in acc_changes if x > 0) / len(acc_changes) * 100
                mean_degrad = np.mean(degradation_rates) * 1000
                divergence_rate = divergence_count / len(results) * 100
                
                f.write(f"| {optimizer:11} | {mean_final:13.4f} | {std_final:13.4f} | "
                       f"{mean_change:15.4f} | {positive_rate:17.1f} | {mean_degrad:16.4f} | {divergence_rate:11.1f} |\n")
                
                if mean_change > best_mean_change:
                    best_mean_change = mean_change
                    best_optimizer = optimizer
            
            # Analysis
            f.write("\n## Analysis\n\n")
            
            f.write(f"### Best Performing Optimizer: **{best_optimizer}**\n\n")
            f.write(f"The {best_optimizer} optimizer showed the best average recovery with a mean accuracy "
                   f"change of {best_mean_change:.4f} after fault injection.\n\n")
            
            # Head-to-head comparisons
            f.write("### Head-to-Head Win Rates\n\n")
            f.write("| Optimizer A | Optimizer B | A Win Rate (%) |\n")
            f.write("|-------------|-------------|----------------|\n")
            
            for i, opt_a in enumerate(self.optimizers_to_test):
                for j, opt_b in enumerate(self.optimizers_to_test):
                    if i >= j:
                        continue
                    
                    wins_a = 0
                    total_comparisons = 0
                    
                    for result in results:
                        final_a = result['optimizer_results'][opt_a]['final_accuracy']
                        final_b = result['optimizer_results'][opt_b]['final_accuracy']
                        if final_a > final_b:
                            wins_a += 1
                        total_comparisons += 1
                    
                    win_rate = (wins_a / total_comparisons * 100) if total_comparisons > 0 else 0
                    f.write(f"| {opt_a:11} | {opt_b:11} | {win_rate:14.1f} |\n")
            
            # Breakdown by injection timing
            f.write("\n## Breakdown by Injection Characteristics\n\n")
            
            early_injections = [r for r in results if r['injection_config']['target_epoch'] <= 1]
            late_injections = [r for r in results if r['injection_config']['target_epoch'] > 1]
            
            f.write(f"### Early Injections (Epoch ≤ 1): {len(early_injections)} experiments\n\n")
            self._write_group_analysis(f, early_injections)
            
            f.write(f"\n### Late Injections (Epoch > 1): {len(late_injections)} experiments\n\n")
            self._write_group_analysis(f, late_injections)
            
            # Conclusions
            f.write("\n## Conclusions\n\n")
            f.write("This experiment trained all optimizers from scratch for each fault injection, ")
            f.write("ensuring a fair comparison with identical training context for all optimizers.\n\n")
            
            f.write("Key findings:\n")
            f.write(f"- **{best_optimizer}** demonstrated the best average recovery performance\n")
            
            # Find most resilient optimizer
            resilience_scores = {}
            for optimizer in self.optimizers_to_test:
                positive_recoveries = sum(1 for r in results 
                                        if r['optimizer_results'][optimizer]['accuracy_change'] > 0)
                resilience_scores[optimizer] = positive_recoveries / len(results)
            
            most_resilient = max(resilience_scores, key=resilience_scores.get)
            f.write(f"- **{most_resilient}** was most resilient with {resilience_scores[most_resilient]*100:.1f}% positive recovery rate\n")
            
            # Find least stable
            divergence_rates = {}
            for optimizer in self.optimizers_to_test:
                diverged = sum(1 for r in results 
                             if r['optimizer_results'][optimizer].get('diverged', False))
                divergence_rates[optimizer] = diverged / len(results)
            
            if any(divergence_rates.values()):
                least_stable = max(divergence_rates, key=divergence_rates.get)
                f.write(f"- **{least_stable}** was least stable with {divergence_rates[least_stable]*100:.1f}% divergence rate\n")
        
        print(f"\nFinal report saved to: {report_path}")
        
        # Create summary visualizations
        self.create_summary_visualizations(results)
    
    def _write_group_analysis(self, f, group_results):
        """Helper to write analysis for a group of results."""
        if not group_results:
            f.write("No experiments in this group\n")
            return
        
        f.write("| Optimizer | Mean Acc Change | Best Case | Worst Case |\n")
        f.write("|-----------|----------------|-----------|------------|\n")
        
        for optimizer in self.optimizers_to_test:
            acc_changes = []
            for result in group_results:
                acc_changes.append(result['optimizer_results'][optimizer]['accuracy_change'])
            
            if acc_changes:
                f.write(f"| {optimizer:11} | {np.mean(acc_changes):14.4f} | "
                       f"{max(acc_changes):9.4f} | {min(acc_changes):10.4f} |\n")
    
    def create_summary_visualizations(self, results: List[Dict]):
        """Create summary visualizations across all experiments."""
        if not results:
            return
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Box plot of accuracy changes
        ax1 = plt.subplot(2, 3, 1)
        acc_changes_by_opt = {opt: [] for opt in self.optimizers_to_test}
        
        for result in results:
            for opt in self.optimizers_to_test:
                acc_changes_by_opt[opt].append(
                    result['optimizer_results'][opt]['accuracy_change']
                )
        
        box_data = [acc_changes_by_opt[opt] for opt in self.optimizers_to_test]
        bp = ax1.boxplot(box_data, labels=self.optimizers_to_test, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Accuracy Change')
        ax1.set_title('Accuracy Change Distribution')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Average recovery trajectories
        ax2 = plt.subplot(2, 3, 2)
        
        max_steps = self.steps_after_injection + 50
        for optimizer in self.optimizers_to_test:
            avg_trajectory = []
            
            for step_idx in range(max_steps):
                step_accuracies = []
                for result in results:
                    history = result['optimizer_results'][optimizer]['history']
                    if step_idx < len(history['accuracy']):
                        # Get accuracy relative to injection point
                        injection_idx = result['injection_config']['target_epoch'] * \
                                      math.ceil(1000 / config.BATCH_SIZE) + \
                                      result['injection_config']['target_step']
                        if injection_idx + step_idx - 10 < len(history['accuracy']):
                            step_accuracies.append(history['accuracy'][injection_idx + step_idx - 10])
                
                if step_accuracies:
                    avg_trajectory.append(np.mean(step_accuracies))
            
            if avg_trajectory:
                ax2.plot(range(len(avg_trajectory)), avg_trajectory, label=optimizer, linewidth=2)
        
        ax2.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Injection')
        ax2.set_xlabel('Steps Relative to Injection')
        ax2.set_ylabel('Average Accuracy')
        ax2.set_title('Average Recovery Trajectories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Win rate matrix heatmap
        ax3 = plt.subplot(2, 3, 3)
        
        win_matrix = np.zeros((len(self.optimizers_to_test), len(self.optimizers_to_test)))
        for i, opt_a in enumerate(self.optimizers_to_test):
            for j, opt_b in enumerate(self.optimizers_to_test):
                if i == j:
                    win_matrix[i, j] = 0.5
                else:
                    wins = sum(1 for r in results 
                             if r['optimizer_results'][opt_a]['final_accuracy'] > 
                                r['optimizer_results'][opt_b]['final_accuracy'])
                    win_matrix[i, j] = wins / len(results)
        
        im = ax3.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax3.set_xticks(range(len(self.optimizers_to_test)))
        ax3.set_xticklabels(self.optimizers_to_test, rotation=45, ha='right')
        ax3.set_yticks(range(len(self.optimizers_to_test)))
        ax3.set_yticklabels(self.optimizers_to_test)
        ax3.set_title('Head-to-Head Win Rates')
        plt.colorbar(im, ax=ax3)
        
        # Add values to heatmap
        for i in range(len(self.optimizers_to_test)):
            for j in range(len(self.optimizers_to_test)):
                text = ax3.text(j, i, f'{win_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        # 4. Degradation rate distribution
        ax4 = plt.subplot(2, 3, 4)
        
        degrad_rates_by_opt = {opt: [] for opt in self.optimizers_to_test}
        for result in results:
            for opt in self.optimizers_to_test:
                rate = result['optimizer_results'][opt].get('degradation_rate', 0) * 1000
                degrad_rates_by_opt[opt].append(rate)
        
        positions = range(len(self.optimizers_to_test))
        for i, opt in enumerate(self.optimizers_to_test):
            if degrad_rates_by_opt[opt]:
                violin = ax4.violinplot([degrad_rates_by_opt[opt]], [i], widths=0.7,
                                       showmeans=True, showmedians=True)
        
        ax4.set_xticks(positions)
        ax4.set_xticklabels(self.optimizers_to_test, rotation=45, ha='right')
        ax4.set_ylabel('Degradation Rate (×1000)')
        ax4.set_title('Degradation Rate Distribution')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Scatter: Initial corruption vs recovery
        ax5 = plt.subplot(2, 3, 5)
        
        colors_map = plt.cm.tab10(np.linspace(0, 1, len(self.optimizers_to_test)))
        for i, opt in enumerate(self.optimizers_to_test):
            initial_corruptions = []
            recoveries = []
            
            for result in results:
                initial_acc = result['injection_results'][opt]['post_injection_accuracy']
                recovery = result['optimizer_results'][opt]['accuracy_change']
                initial_corruptions.append(initial_acc)
                recoveries.append(recovery)
            
            ax5.scatter(initial_corruptions, recoveries, label=opt, 
                       alpha=0.6, s=50, color=colors_map[i])
        
        ax5.set_xlabel('Post-Injection Accuracy')
        ax5.set_ylabel('Accuracy Recovery')
        ax5.set_title('Recovery vs Initial Corruption')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance by injection epoch
        ax6 = plt.subplot(2, 3, 6)
        
        epochs = sorted(set(r['injection_config']['target_epoch'] for r in results))
        performance_matrix = []
        
        for opt in self.optimizers_to_test:
            opt_performance = []
            for epoch in epochs:
                epoch_results = [r for r in results if r['injection_config']['target_epoch'] == epoch]
                epoch_changes = []
                for r in epoch_results:
                    epoch_changes.append(r['optimizer_results'][opt]['accuracy_change'])
                
                opt_performance.append(np.mean(epoch_changes) if epoch_changes else 0)
            
            performance_matrix.append(opt_performance)
        
        im = ax6.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        ax6.set_xticks(range(len(epochs)))
        ax6.set_xticklabels([f'Epoch {e}' for e in epochs])
        ax6.set_yticks(range(len(self.optimizers_to_test)))
        ax6.set_yticklabels(self.optimizers_to_test)
        ax6.set_title('Recovery Performance by Injection Epoch')
        plt.colorbar(im, ax=ax6, label='Avg Accuracy Change')
        
        # Add values
        for i in range(len(self.optimizers_to_test)):
            for j in range(len(epochs)):
                text = ax6.text(j, i, f'{performance_matrix[i][j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        summary_plot_path = os.path.join(self.results_base_dir, 'summary_visualizations.png')
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualizations saved to: {summary_plot_path}")


def main():
    """Main function to run the parallel optimizer experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test optimizer mitigation with parallel training (Version 3)'
    )
    parser.add_argument('--optimizers', type=str, nargs='+',
                       default=['adam', 'sgd', 'rmsprop', 'adamw'],
                       help='Optimizers to test in parallel')
    parser.add_argument('--num-experiments', type=int, default=10,
                       help='Number of experiments to run (default: 10)')
    parser.add_argument('--steps-after-injection', type=int, default=100,
                       help='Steps to train after injection (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PARALLEL OPTIMIZER MITIGATION EXPERIMENT - VERSION 3")
    print("="*80)
    print("\nKey features:")
    print("✓ All optimizers train from scratch for each experiment")
    print("✓ Identical injection applied to all models simultaneously")
    print("✓ Fair comparison with same training context")
    print("✓ Comprehensive visualization and analysis")
    print("="*80 + "\n")
    
    # Debug: Show parsed arguments
    print(f"Parsed arguments:")
    print(f"  num_experiments: {args.num_experiments}")
    print(f"  steps_after_injection: {args.steps_after_injection}")
    print(f"  optimizers: {args.optimizers}")
    print(f"  seed: {args.seed}")
    print(f"  learning_rate: {args.learning_rate}")
    print("="*80 + "\n")
    
    # Create and run experiment
    experiment = ParallelOptimizerMitigationExperiment(
        optimizers_to_test=args.optimizers,
        num_experiments=args.num_experiments,
        base_seed=args.seed,
        learning_rate=args.learning_rate,
        steps_after_injection=args.steps_after_injection
    )
    
    results = experiment.run()
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results directory: {experiment.results_base_dir}")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()