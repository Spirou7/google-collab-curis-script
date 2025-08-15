import os
import json
import time
import random
import numpy as np
import tensorflow as tf
import math
from typing import Dict, List

from fault_injection.models.resnet import resnet_18
from fault_injection.models.backward_resnet import backward_resnet_18
from fault_injection.data.prepare_data import generate_datasets
from fault_injection.core import config

from .checkpoint_utils import save_post_injection_checkpoint, load_corrupted_checkpoint, analyze_weight_corruption
from .training_utils import (
    create_train_step_function, create_recovery_train_step,
    create_get_layer_outputs_function, perform_injection,
    setup_training_metrics, log_training_progress
)
from .optimizer_utils import create_optimizer
from .metrics_utils import calculate_recovery_metrics, check_for_divergence


class PhaseRunner:
    """Handles running different phases of the experiment."""
    
    @staticmethod
    def create_corrupted_checkpoint(injection_config: Dict, results_base_dir: str) -> Dict:
        """
        Phase 1: Train model, inject fault, and save corrupted state.
        """
        print(f"\n" + "="*80)
        print(f"PHASE 1: CREATING CORRUPTED CHECKPOINT")
        print("="*80)
        
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
        
        # Create experiment directory
        experiment_dir = os.path.join(results_base_dir, f"experiment_{exp_id:03d}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"\n📁 Experiment directory: {experiment_dir}")
        
        # Save injection config
        config_path = os.path.join(experiment_dir, "injection_config.json")
        with open(config_path, 'w') as f:
            json.dump(injection_config, f, indent=2, default=str)
        
        # Get datasets
        print(f"\n📊 Loading datasets...")
        train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)
        steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
        
        # Create models
        print(f"\n🏗️ Creating models...")
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        back_model = backward_resnet_18('resnet18')
        
        # Create optimizer
        model.optimizer = create_optimizer(injection_config.get('baseline_optimizer', 'adam'), 
                                          injection_config['learning_rate'])
        
        # Setup metrics
        train_loss, train_accuracy = setup_training_metrics()
        
        # Create training functions
        train_step = create_train_step_function(model, train_loss, train_accuracy)
        get_layer_outputs = create_get_layer_outputs_function(model)
        
        # Calculate injection point
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
        print(f"\n🏃 TRAINING PHASE (Pre-injection)")
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
                
                # Record metrics
                pre_injection_history['steps'].append(global_step)
                pre_injection_history['accuracy'].append(float(train_accuracy.result()))
                pre_injection_history['loss'].append(float(train_loss.result()))
                
                # Log progress
                if global_step % 10 == 0:
                    log_training_progress(global_step, injection_global_step, 
                                        train_loss, train_accuracy, 
                                        start_time, current_epoch)
            
            elif global_step == injection_global_step:
                # Perform injection
                print(f"\n" + "="*60)
                print(f"🎯 INJECTION POINT REACHED")
                print("="*60)
                
                # Get batch for injection
                images, labels = next(train_iterator)
                
                # Save batch data
                batch_data_path = os.path.join(experiment_dir, 'injection_batch.npz')
                np.savez(batch_data_path, images=images.numpy(), labels=labels.numpy())
                
                # Record pre-injection state
                pre_injection_acc = float(train_accuracy.result())
                pre_injection_loss = float(train_loss.result())
                
                # Reset metrics before injection
                train_loss.reset_state()
                train_accuracy.reset_state()
                
                # Perform injection
                loss, corrupted_outputs = perform_injection(
                    model, back_model, images, labels, injection_config,
                    train_loss, train_accuracy, get_layer_outputs
                )
                
                # Analyze corruption
                corruption_info = analyze_weight_corruption(model)
                corruption_info.update({
                    'injection_step': global_step,
                    'pre_injection_accuracy': pre_injection_acc,
                    'post_injection_accuracy': float(train_accuracy.result()),
                    'post_injection_loss': float(loss),
                    'injection_position': injection_config['injection_position'],
                    'injection_value': injection_config['injection_value']
                })
                
                # Save corrupted checkpoint
                checkpoint_dir = save_post_injection_checkpoint(
                    model, experiment_dir, global_step, corruption_info
                )
                
                # Save pre-injection history
                history_path = os.path.join(experiment_dir, 'pre_injection_history.json')
                with open(history_path, 'w') as f:
                    json.dump(pre_injection_history, f, indent=2)
                
                print(f"\n✅ PHASE 1 COMPLETE")
                
                return {
                    'experiment_dir': experiment_dir,
                    'checkpoint_dir': checkpoint_dir,
                    'corruption_info': corruption_info,
                    'pre_injection_history': pre_injection_history,
                    'injection_batch_path': batch_data_path
                }
    
    @staticmethod
    def test_optimizer_recovery(checkpoint_info: Dict, optimizer_name: str, 
                               injection_config: Dict, max_steps_after_injection: int) -> Dict:
        """
        Phase 2: Load corrupted checkpoint and test recovery with specified optimizer.
        """
        print(f"\n" + "="*80)
        print(f"PHASE 2: TESTING OPTIMIZER RECOVERY")
        print(f"Optimizer: {optimizer_name.upper()}")
        print("="*80)
        
        # Set seeds
        seed = injection_config['seed']
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create fresh model
        print(f"\n🏗️ Creating fresh model...")
        model = resnet_18(seed, 'resnet18')
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        
        # Load corrupted weights
        corruption_info = load_corrupted_checkpoint(model, checkpoint_info['checkpoint_dir'])
        
        # Create backward model
        back_model = backward_resnet_18('resnet18')
        
        # Create optimizer
        injection_step = checkpoint_info['corruption_info']['injection_step']
        model.optimizer = create_optimizer(optimizer_name, 
                                          injection_config['learning_rate'],
                                          current_step=injection_step + 1)
        
        # Setup metrics
        train_loss, train_accuracy = setup_training_metrics()
        
        # Create training function
        train_step = create_recovery_train_step(model, train_loss, train_accuracy)
        
        # Get dataset
        print(f"\n📊 Loading dataset for recovery...")
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
        
        # Position dataset
        print(f"\n⏩ Fast-forwarding dataset...")
        train_iterator = iter(train_dataset)
        for skip_step in range(injection_step + 1):
            try:
                next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataset)
        
        # Train for recovery
        print(f"\n🏃 RECOVERY TRAINING PHASE")
        print(f"   Training for {max_steps_after_injection} steps with {optimizer_name}")
        
        start_time = time.time()
        divergence_detected = False
        
        for step in range(max_steps_after_injection):
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
                train_iterator = iter(train_dataset)
                images, labels = next(train_iterator)
            
            # Train step
            loss = train_step(images, labels)
            
            # Record metrics
            recovery_history['steps'].append(global_step)
            recovery_history['accuracy'].append(float(train_accuracy.result()))
            recovery_history['loss'].append(float(train_loss.result()))
            
            # Check for divergence
            diverged, message = check_for_divergence(loss, step)
            if diverged:
                print(message)
                recovery_history['diverged'] = True
                recovery_history['divergence_step'] = step
                divergence_detected = True
                break
            
            # Log progress
            if step % 10 == 0 or step == max_steps_after_injection - 1:
                current_acc = train_accuracy.result()
                acc_change = current_acc - recovery_history['starting_accuracy']
                
                print(f"   Step {step:3d}/{max_steps_after_injection} | "
                      f"Acc: {current_acc:.4f} ({acc_change:+.4f}) | "
                      f"Loss: {train_loss.result():.4f}")
        
        # Calculate recovery metrics
        recovery_history = calculate_recovery_metrics(recovery_history)
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"RECOVERY SUMMARY: {optimizer_name.upper()}")
        print("="*60)
        print(f"   Initial → Final: {recovery_history['starting_accuracy']:.4f} → "
              f"{recovery_history['final_accuracy']:.4f}")
        print(f"   Net change: {recovery_history['accuracy_change']:+.4f}")
        
        if divergence_detected:
            print(f"   ⚠️ Training diverged at step {recovery_history.get('divergence_step', 'unknown')}")
        
        return recovery_history