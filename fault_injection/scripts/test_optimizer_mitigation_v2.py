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

class OptimizerMitigationExperimentV2:
    """
    CORRECTED VERSION: Tests whether changing optimizers after fault injection can mitigate slowdegrade.
    
    Key improvements:
    1. Saves checkpoint AFTER injection to ensure all optimizers start from same corrupted state
    2. Pre-generates and saves exact injection parameters for reproducibility
    3. All experiments (baseline and mitigated) follow same execution path
    4. Properly handles learning rate schedules across optimizer switches
    """
    
    def __init__(self, 
                 baseline_optimizer: str = 'adam',
                 test_optimizers: List[str] = None,
                 num_experiments: int = 100,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 max_steps_after_injection: int = 200):
        """
        Initialize the experiment with corrected design.
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            f"optimizer_comparison_results_{timestamp}"
        )
        os.makedirs(self.results_base_dir, exist_ok=True)
        
        # Pre-generate all injection configurations for reproducibility
        self.injection_configs = self._pre_generate_injection_configs()
        
    def _pre_generate_injection_configs(self) -> List[Dict]:
        """
        Pre-generate all injection configurations to ensure reproducibility.
        This guarantees that each experiment uses exactly the same injection.
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
    
    def create_optimizer(self, optimizer_name: str, learning_rate: float = None,
                        current_step: int = 0) -> tf.keras.optimizers.Optimizer:
        """
        Create optimizer with proper learning rate schedule continuation.
        
        Args:
            optimizer_name: Name of the optimizer
            learning_rate: Initial learning rate
            current_step: Current training step for schedule continuation
        """
        lr = learning_rate or self.learning_rate
        
        # Create learning rate schedule that accounts for current step
        # This ensures fairness when switching optimizers
        class ContinuedPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, decay_steps, end_lr, current_step):
                self.initial_lr = initial_lr
                self.decay_steps = decay_steps
                self.end_lr = end_lr
                self.current_step = current_step
                
            def __call__(self, step):
                # Continue from current position in schedule
                effective_step = step + self.current_step
                completion = tf.minimum(effective_step / self.decay_steps, 1.0)
                return self.initial_lr + (self.end_lr - self.initial_lr) * completion
            
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
            pass  # Keep default fallback
        
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
    
    def save_post_injection_checkpoint(self, model: tf.keras.Model, 
                                      experiment_dir: str,
                                      injection_step: int,
                                      corrupted_state_info: Dict) -> str:
        """
        Save model state immediately after injection.
        This is the corrupted state that all optimizers will start from.
        """
        checkpoint_dir = os.path.join(experiment_dir, "post_injection_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights (now corrupted)
        weights_path = os.path.join(checkpoint_dir, 'corrupted_model.weights.h5')
        model.save_weights(weights_path)
        
        # Save corruption info
        info_path = os.path.join(checkpoint_dir, 'corruption_info.json')
        with open(info_path, 'w') as f:
            json.dump(corrupted_state_info, f, indent=2, default=str)
        
        print(f"Saved post-injection checkpoint with corruption info")
        return checkpoint_dir
    
    def phase1_create_corrupted_checkpoint(self, injection_config: Dict) -> Dict:
        """
        Phase 1: Train model, inject fault, and save corrupted state.
        This creates the starting point for all optimizer comparisons.
        """
        print(f"\n=== Phase 1: Creating corrupted checkpoint ===")
        
        # Set seeds
        seed = injection_config['seed']
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create experiment directory
        exp_id = injection_config['experiment_id']
        experiment_dir = os.path.join(self.results_base_dir, f"experiment_{exp_id:03d}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save injection config
        config_path = os.path.join(experiment_dir, "injection_config.json")
        with open(config_path, 'w') as f:
            json.dump(injection_config, f, indent=2, default=str)
        
        # Get datasets
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        
        # Create model
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_resnet_18('resnet18')
        
        # Create baseline optimizer
        model.optimizer = self.create_optimizer(self.baseline_optimizer)
        
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
            return avg_loss, images, labels  # Return batch data for injection
        
        @tf.function
        def get_layer_outputs(images, inj_layer):
            outputs, l_inputs, l_kernels, l_outputs = model(images, training=True, inject=False)
            return l_inputs[inj_layer], l_kernels[inj_layer], l_outputs[inj_layer]
        
        def perform_injection(images, labels, inj_layer, inj_position, inj_value):
            """Perform deterministic injection with pre-generated values."""
            with tf.GradientTape() as tape:
                # Get layer outputs for injection
                l_inputs, l_kernels, l_outputs = get_layer_outputs(images, inj_layer)
                
                # Import necessary injection utilities
                from fault_injection.models.inject_utils import get_inj_args_with_random_range
                
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
                
                # Perform forward pass with injection
                outputs, l_inputs_inj, l_kernels_inj, l_outputs_inj = model(
                    images, training=True, inject=inj_flag, inj_args=inj_args
                )
                predictions = outputs['logits']
                grad_start = outputs['grad_start']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)
            
            # Backward pass with manual gradient computation
            man_grad_start, golden_gradients = tape.gradient(avg_loss, [grad_start, model.trainable_variables])
            manual_gradients, _, _, _ = back_model.call(man_grad_start, l_inputs_inj, l_kernels_inj, 
                                                        inject=False, inj_args=None)
            
            gradients = manual_gradients + golden_gradients[-2:]
            model.optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
            
            train_loss.update_state(avg_loss)
            train_accuracy.update_state(labels, predictions)
            
            return avg_loss, l_outputs_inj
        
        # Training parameters
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        target_epoch = injection_config['target_epoch']
        target_step = injection_config['target_step']
        injection_global_step = target_epoch * steps_per_epoch + target_step
        
        # Training history
        pre_injection_history = {
            'steps': [],
            'accuracy': [],
            'loss': []
        }
        
        # Train until injection point
        print(f"Training until injection at epoch {target_epoch}, step {target_step}")
        train_iterator = iter(train_dataset)
        
        for global_step in range(injection_global_step + 1):
            # Reset iterator at epoch boundaries
            if global_step > 0 and global_step % steps_per_epoch == 0:
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
                
                if global_step % 50 == 0:
                    print(f"Step {global_step}: accuracy={train_accuracy.result():.4f}")
            
            elif global_step == injection_global_step:
                # Perform injection with deterministic parameters
                print(f"\n🎯 Injecting fault at step {global_step}")
                print(f"  Layer: {injection_config['target_layer']}")
                print(f"  Position: {injection_config['injection_position']}")
                print(f"  Value: {injection_config['injection_value']:.2e}")
                
                # Get batch for injection
                images, labels = next(train_iterator)
                
                # Save batch data for exact reproduction
                batch_data_path = os.path.join(experiment_dir, 'injection_batch.npz')
                np.savez(batch_data_path, 
                         images=images.numpy(),
                         labels=labels.numpy())
                
                # Reset metrics before injection
                train_loss.reset_state()
                train_accuracy.reset_state()
                
                # Perform injection
                loss, corrupted_outputs = perform_injection(
                    images, labels,
                    injection_config['target_layer'],
                    injection_config['injection_position'],
                    injection_config['injection_value']
                )
                
                print(f"Post-injection: accuracy={train_accuracy.result():.4f}, loss={loss:.4f}")
                
                # Analyze corruption
                corruption_info = {
                    'injection_step': global_step,
                    'pre_injection_accuracy': pre_injection_history['accuracy'][-1] if pre_injection_history['accuracy'] else 0,
                    'post_injection_accuracy': float(train_accuracy.result()),
                    'post_injection_loss': float(loss),
                    'injection_position': injection_config['injection_position'],
                    'injection_value': injection_config['injection_value']
                }
                
                # Check for NaN/Inf in weights
                nan_count = 0
                inf_count = 0
                for var in model.trainable_variables:
                    nan_count += tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)).numpy()
                    inf_count += tf.reduce_sum(tf.cast(tf.math.is_inf(var), tf.int32)).numpy()
                
                corruption_info['nan_weights'] = int(nan_count)
                corruption_info['inf_weights'] = int(inf_count)
                
                # Save corrupted checkpoint
                checkpoint_dir = self.save_post_injection_checkpoint(
                    model, experiment_dir, global_step, corruption_info
                )
                
                # Save pre-injection history
                history_path = os.path.join(experiment_dir, 'pre_injection_history.json')
                with open(history_path, 'w') as f:
                    json.dump(pre_injection_history, f, indent=2)
                
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
        print(f"\n=== Phase 2: Testing {optimizer_name} recovery ===")
        
        # Set seeds for reproducibility
        seed = injection_config['seed']
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create fresh model
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_resnet_18('resnet18')
        
        # Load corrupted weights
        weights_path = os.path.join(checkpoint_info['checkpoint_dir'], 'corrupted_model.weights.h5')
        model.load_weights(weights_path)
        print(f"Loaded corrupted weights from {weights_path}")
        
        # Create optimizer (with proper schedule continuation)
        injection_step = checkpoint_info['corruption_info']['injection_step']
        model.optimizer = self.create_optimizer(optimizer_name, 
                                               current_step=injection_step + 1)
        
        # Setup metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
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
        train_dataset, _, train_count, _ = generate_datasets(seed)
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        
        # Recovery history
        recovery_history = {
            'optimizer': optimizer_name,
            'steps': [],
            'accuracy': [],
            'loss': [],
            'starting_accuracy': checkpoint_info['corruption_info']['post_injection_accuracy'],
            'starting_loss': checkpoint_info['corruption_info']['post_injection_loss']
        }
        
        # Position dataset to continue from injection point
        train_iterator = iter(train_dataset)
        
        # Skip to the injection point in dataset
        for _ in range(injection_step + 1):
            try:
                next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataset)
        
        # Train for specified steps after injection
        print(f"Training for {self.max_steps_after_injection} steps with {optimizer_name}")
        
        for step in range(self.max_steps_after_injection):
            global_step = injection_step + 1 + step
            
            # Reset iterator at epoch boundaries
            if global_step % steps_per_epoch == 0:
                train_iterator = iter(train_dataset)
            
            # Get batch
            try:
                images, labels = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataset)
                images, labels = next(train_iterator)
            
            # Train step
            loss = train_step(images, labels)
            
            # Record metrics
            recovery_history['steps'].append(global_step)
            recovery_history['accuracy'].append(float(train_accuracy.result()))
            recovery_history['loss'].append(float(train_loss.result()))
            
            # Progress update
            if step % 50 == 0 or step == self.max_steps_after_injection - 1:
                print(f"  Step {step}/{self.max_steps_after_injection}: "
                      f"accuracy={train_accuracy.result():.4f}, loss={train_loss.result():.4f}")
            
            # Check for divergence
            if not tf.math.is_finite(loss):
                print(f"  ⚠️ Training diverged (NaN/Inf loss) at step {step}")
                recovery_history['diverged'] = True
                recovery_history['divergence_step'] = step
                break
        
        # Calculate recovery metrics
        recovery_history['final_accuracy'] = recovery_history['accuracy'][-1] if recovery_history['accuracy'] else 0
        recovery_history['final_loss'] = recovery_history['loss'][-1] if recovery_history['loss'] else float('inf')
        recovery_history['accuracy_change'] = recovery_history['final_accuracy'] - recovery_history['starting_accuracy']
        
        # Calculate degradation rate
        if len(recovery_history['accuracy']) > 10:
            # Linear fit to accuracy over last 100 steps
            recent_steps = recovery_history['steps'][-100:] if len(recovery_history['steps']) > 100 else recovery_history['steps']
            recent_acc = recovery_history['accuracy'][-100:] if len(recovery_history['accuracy']) > 100 else recovery_history['accuracy']
            
            if len(recent_steps) > 1:
                z = np.polyfit(recent_steps, recent_acc, 1)
                recovery_history['degradation_rate'] = float(z[0])  # Slope of accuracy change
            else:
                recovery_history['degradation_rate'] = 0
        else:
            recovery_history['degradation_rate'] = 0
        
        return recovery_history
    
    def run_single_experiment(self, experiment_id: int) -> Dict:
        """
        Run complete experiment: create corruption, then test all optimizers.
        """
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {experiment_id + 1}/{self.num_experiments}")
        print(f"{'='*80}")
        
        # Get pre-generated injection config
        injection_config = self.injection_configs[experiment_id]
        
        # Phase 1: Create corrupted checkpoint
        checkpoint_info = self.phase1_create_corrupted_checkpoint(injection_config)
        
        # Phase 2: Test each optimizer's recovery
        results = {
            'experiment_id': experiment_id,
            'injection_config': injection_config,
            'corruption_info': checkpoint_info['corruption_info'],
            'pre_injection_history': checkpoint_info['pre_injection_history'],
            'recovery_results': {}
        }
        
        # Test baseline optimizer
        baseline_recovery = self.phase2_test_optimizer_recovery(
            checkpoint_info, self.baseline_optimizer, injection_config
        )
        results['recovery_results'][self.baseline_optimizer] = baseline_recovery
        
        # Test alternative optimizers
        for optimizer_name in self.test_optimizers:
            recovery = self.phase2_test_optimizer_recovery(
                checkpoint_info, optimizer_name, injection_config
            )
            results['recovery_results'][optimizer_name] = recovery
        
        # Save results
        self.save_experiment_results(results, checkpoint_info['experiment_dir'])
        
        # Create visualizations
        self.create_experiment_visualizations(results, checkpoint_info['experiment_dir'])
        
        return results
    
    def save_experiment_results(self, results: Dict, experiment_dir: str):
        """Save all experiment results."""
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
    
    def create_experiment_visualizations(self, results: Dict, experiment_dir: str):
        """Create comparison plots for the experiment."""
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
        
        # Create degradation rate plot
        self.create_degradation_plot(results, experiment_dir)
    
    def create_degradation_plot(self, results: Dict, experiment_dir: str):
        """Create plot focusing on degradation rates."""
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
    
    def run(self) -> List[Dict]:
        """Run all experiments."""
        print(f"\n{'='*80}")
        print(f"Starting Optimizer Mitigation Experiments (Corrected Version)")
        print(f"{'='*80}")
        print(f"Baseline optimizer: {self.baseline_optimizer}")
        print(f"Test optimizers: {self.test_optimizers}")
        print(f"Number of experiments: {self.num_experiments}")
        print(f"Steps after injection: {self.max_steps_after_injection}")
        print(f"Results directory: {self.results_base_dir}")
        
        all_results = []
        successful_experiments = 0
        
        for exp_id in range(self.num_experiments):
            try:
                results = self.run_single_experiment(exp_id)
                all_results.append(results)
                successful_experiments += 1
                
                # Save intermediate summary
                if (exp_id + 1) % 10 == 0:
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
    
    def generate_final_report(self, results: List[Dict]):
        """Generate comprehensive final report."""
        if not results:
            print("No results to report")
            return
        
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
                    
                    f.write(f"| {optimizer:11} | {mean_final:13.4f} | {std_final:13.4f} | "
                           f"{mean_change:15.4f} | {positive_rate:17.1f} | {mean_degrad:16.4f} |\n")
            
            # Winner analysis
            f.write("\n## Analysis\n\n")
            
            # Find best performer
            optimizer_scores = {}
            for optimizer in [self.baseline_optimizer] + self.test_optimizers:
                acc_changes = []
                for result in results:
                    if optimizer in result['recovery_results']:
                        acc_changes.append(result['recovery_results'][optimizer]['accuracy_change'])
                
                if acc_changes:
                    # Score based on mean improvement and consistency
                    mean_improvement = np.mean(acc_changes)
                    consistency = 1 / (1 + np.std(acc_changes))
                    optimizer_scores[optimizer] = mean_improvement + 0.1 * consistency
            
            if optimizer_scores:
                best_optimizer = max(optimizer_scores, key=optimizer_scores.get)
                best_score = optimizer_scores[best_optimizer]
                
                f.write(f"### Best Performing Optimizer: **{best_optimizer}**\n\n")
                
                if best_score > optimizer_scores.get(self.baseline_optimizer, 0):
                    improvement = best_score - optimizer_scores.get(self.baseline_optimizer, 0)
                    f.write(f"✅ **{best_optimizer}** outperforms baseline by {improvement:.4f} points\n\n")
                else:
                    f.write(f"⚠️ Baseline optimizer **{self.baseline_optimizer}** performs best\n\n")
            
            # Detailed breakdown by injection characteristics
            f.write("## Breakdown by Injection Characteristics\n\n")
            
            # Group by early vs late injection
            early_injections = [r for r in results if r['injection_config']['target_epoch'] <= 1]
            late_injections = [r for r in results if r['injection_config']['target_epoch'] > 1]
            
            f.write(f"### Early Injections (Epoch ≤ 1): {len(early_injections)} experiments\n\n")
            self._write_group_analysis(f, early_injections)
            
            f.write(f"\n### Late Injections (Epoch > 1): {len(late_injections)} experiments\n\n")
            self._write_group_analysis(f, late_injections)
            
            # Conclusions
            f.write("\n## Conclusions\n\n")
            
            # Check hypothesis
            hypothesis_supported = False
            for optimizer in self.test_optimizers:
                scores = []
                for result in results:
                    if optimizer in result['recovery_results'] and self.baseline_optimizer in result['recovery_results']:
                        test_change = result['recovery_results'][optimizer]['accuracy_change']
                        baseline_change = result['recovery_results'][self.baseline_optimizer]['accuracy_change']
                        if test_change > baseline_change:
                            scores.append(1)
                        else:
                            scores.append(0)
                
                if scores and np.mean(scores) > 0.5:
                    hypothesis_supported = True
                    f.write(f"✅ **Hypothesis SUPPORTED**: {optimizer} shows better recovery "
                           f"than {self.baseline_optimizer} in {np.mean(scores)*100:.1f}% of cases\n\n")
            
            if not hypothesis_supported:
                f.write("❌ **Hypothesis NOT SUPPORTED**: No alternative optimizer consistently "
                       "outperforms the baseline in recovering from slowdegrade effects\n\n")
        
        print(f"\nFinal report saved to: {report_path}")
        
        # Also create summary visualizations
        self.create_summary_visualizations(results)
    
    def _write_group_analysis(self, f, group_results):
        """Helper to write analysis for a group of results."""
        if not group_results:
            f.write("No experiments in this group\n")
            return
        
        f.write("| Optimizer | Mean Acc Change | Best Case | Worst Case |\n")
        f.write("|-----------|----------------|-----------|------------|\n")
        
        for optimizer in [self.baseline_optimizer] + self.test_optimizers:
            acc_changes = []
            for result in group_results:
                if optimizer in result['recovery_results']:
                    acc_changes.append(result['recovery_results'][optimizer]['accuracy_change'])
            
            if acc_changes:
                f.write(f"| {optimizer:11} | {np.mean(acc_changes):14.4f} | "
                       f"{max(acc_changes):9.4f} | {min(acc_changes):10.4f} |\n")
    
    def create_summary_visualizations(self, results: List[Dict]):
        """Create summary visualizations across all experiments."""
        if not results:
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Box plot of accuracy changes
        ax1 = plt.subplot(2, 3, 1)
        optimizer_names = [self.baseline_optimizer] + self.test_optimizers
        acc_changes_by_opt = {opt: [] for opt in optimizer_names}
        
        for result in results:
            for opt in optimizer_names:
                if opt in result['recovery_results']:
                    acc_changes_by_opt[opt].append(
                        result['recovery_results'][opt]['accuracy_change']
                    )
        
        box_data = [acc_changes_by_opt[opt] for opt in optimizer_names]
        bp = ax1.boxplot(box_data, labels=optimizer_names, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightgray']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Accuracy Change')
        ax1.set_title('Accuracy Change Distribution by Optimizer')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. Win rate comparison
        ax2 = plt.subplot(2, 3, 2)
        win_rates = {}
        
        for test_opt in self.test_optimizers:
            wins = 0
            total = 0
            for result in results:
                if test_opt in result['recovery_results'] and self.baseline_optimizer in result['recovery_results']:
                    test_final = result['recovery_results'][test_opt]['final_accuracy']
                    baseline_final = result['recovery_results'][self.baseline_optimizer]['final_accuracy']
                    if test_final > baseline_final:
                        wins += 1
                    total += 1
            
            if total > 0:
                win_rates[test_opt] = wins / total * 100
        
        if win_rates:
            bars = ax2.bar(win_rates.keys(), win_rates.values())
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_title(f'Win Rate vs {self.baseline_optimizer}')
            ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
            
            # Color bars based on performance
            for bar, rate in zip(bars, win_rates.values()):
                if rate > 50:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Degradation rate distribution
        ax3 = plt.subplot(2, 3, 3)
        degrad_rates_by_opt = {opt: [] for opt in optimizer_names}
        
        for result in results:
            for opt in optimizer_names:
                if opt in result['recovery_results']:
                    rate = result['recovery_results'][opt].get('degradation_rate', 0) * 1000
                    degrad_rates_by_opt[opt].append(rate)
        
        positions = range(len(optimizer_names))
        for i, opt in enumerate(optimizer_names):
            if degrad_rates_by_opt[opt]:
                violin = ax3.violinplot([degrad_rates_by_opt[opt]], [i], widths=0.7,
                                       showmeans=True, showmedians=True)
        
        ax3.set_xticks(positions)
        ax3.set_xticklabels(optimizer_names, rotation=45, ha='right')
        ax3.set_ylabel('Degradation Rate (×1000)')
        ax3.set_title('Degradation Rate Distribution')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Scatter plot: Initial corruption vs recovery
        ax4 = plt.subplot(2, 3, 4)
        for opt in optimizer_names:
            initial_corruptions = []
            recoveries = []
            
            for result in results:
                if opt in result['recovery_results']:
                    initial_acc = result['corruption_info']['post_injection_accuracy']
                    recovery = result['recovery_results'][opt]['accuracy_change']
                    initial_corruptions.append(initial_acc)
                    recoveries.append(recovery)
            
            if initial_corruptions:
                ax4.scatter(initial_corruptions, recoveries, label=opt, alpha=0.6, s=30)
        
        ax4.set_xlabel('Initial Post-Injection Accuracy')
        ax4.set_ylabel('Accuracy Recovery')
        ax4.set_title('Recovery vs Initial Corruption')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Time series: Average accuracy trajectory
        ax5 = plt.subplot(2, 3, 5)
        
        # Compute average trajectories
        max_steps = max([len(r['recovery_results'][self.baseline_optimizer]['steps']) 
                        for r in results if self.baseline_optimizer in r['recovery_results']])
        
        for opt in optimizer_names:
            avg_trajectory = []
            
            for step_idx in range(max_steps):
                step_accuracies = []
                for result in results:
                    if opt in result['recovery_results']:
                        recovery = result['recovery_results'][opt]
                        if step_idx < len(recovery['accuracy']):
                            step_accuracies.append(recovery['accuracy'][step_idx])
                
                if step_accuracies:
                    avg_trajectory.append(np.mean(step_accuracies))
            
            if avg_trajectory:
                ax5.plot(avg_trajectory, label=opt, linewidth=2)
        
        ax5.set_xlabel('Steps After Injection')
        ax5.set_ylabel('Average Accuracy')
        ax5.set_title('Average Recovery Trajectories')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Heatmap: Performance by injection timing
        ax6 = plt.subplot(2, 3, 6)
        
        # Create matrix: optimizers × epochs
        epochs = sorted(set(r['injection_config']['target_epoch'] for r in results))
        performance_matrix = []
        
        for opt in optimizer_names:
            opt_performance = []
            for epoch in epochs:
                epoch_results = [r for r in results if r['injection_config']['target_epoch'] == epoch]
                epoch_changes = []
                for r in epoch_results:
                    if opt in r['recovery_results']:
                        epoch_changes.append(r['recovery_results'][opt]['accuracy_change'])
                
                opt_performance.append(np.mean(epoch_changes) if epoch_changes else 0)
            
            performance_matrix.append(opt_performance)
        
        im = ax6.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')
        ax6.set_xticks(range(len(epochs)))
        ax6.set_xticklabels([f'Epoch {e}' for e in epochs])
        ax6.set_yticks(range(len(optimizer_names)))
        ax6.set_yticklabels(optimizer_names)
        ax6.set_title('Recovery Performance by Injection Epoch')
        plt.colorbar(im, ax=ax6, label='Avg Accuracy Change')
        
        # Add values to heatmap
        for i in range(len(optimizer_names)):
            for j in range(len(epochs)):
                text = ax6.text(j, i, f'{performance_matrix[i][j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        summary_plot_path = os.path.join(self.results_base_dir, 'summary_visualizations.png')
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualizations saved to: {summary_plot_path}")


def main():
    """Main function to run the corrected experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test optimizer mitigation for slowdegrade effects (Corrected Version)'
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
    
    print("\n" + "="*80)
    print("OPTIMIZER MITIGATION EXPERIMENT - CORRECTED VERSION")
    print("="*80)
    print("\nThis version ensures:")
    print("✓ Identical injection for all optimizers")
    print("✓ Fair comparison from same corrupted state")
    print("✓ Proper learning rate schedule continuation")
    print("✓ Reproducible results via pre-generated configs")
    print("="*80 + "\n")
    
    # Create and run experiment
    experiment = OptimizerMitigationExperimentV2(
        baseline_optimizer=args.baseline,
        test_optimizers=args.test_optimizers,
        num_experiments=args.num_experiments,
        base_seed=args.seed,
        learning_rate=args.learning_rate,
        max_steps_after_injection=args.steps_after_injection
    )
    
    results = experiment.run()
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results directory: {experiment.results_base_dir}")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()