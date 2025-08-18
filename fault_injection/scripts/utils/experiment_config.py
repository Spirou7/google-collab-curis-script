import json
import os
import random
import numpy as np
from typing import List, Dict, Optional
import datetime
import tensorflow as tf

# Import new determinism and manifest systems
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.determinism import DeterminismController, seed_everything
from core.manifest import RunManifest, create_manifest_for_run


class ExperimentConfig:
    """Configuration and setup for optimizer mitigation experiments."""
    
    def __init__(self, 
                 baseline_optimizer: str = 'adam',
                 test_optimizers: List[str] = None,
                 num_experiments: int = 100,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 max_steps_after_injection: int = 200,
                 enable_determinism: bool = True,
                 create_manifest: bool = True):
        """Initialize experiment configuration with enhanced determinism and tracking."""
        self.baseline_optimizer = baseline_optimizer
        self.test_optimizers = test_optimizers or ['sgd', 'rmsprop', 'adamw']
        self.num_experiments = num_experiments
        self.base_seed = base_seed
        self.learning_rate = learning_rate
        self.max_steps_after_injection = max_steps_after_injection
        self.enable_determinism = enable_determinism
        self.create_manifest = create_manifest
        
        # Initialize determinism controller
        self.determinism_controller = None
        if self.enable_determinism:
            self.determinism_controller = DeterminismController(self.base_seed)
            self.determinism_controller.seed_all()
        
        # Fault injection parameters
        self.fmodel = 'N16_RD'
        self.min_val = 3.6e2
        self.max_val = 1.2e8
        self.max_target_epoch = 3
        self.max_target_step = 49
        
        # Setup results directory
        self.results_base_dir = self._setup_results_directory()
        
        # Initialize manifest
        self.manifest = None
        if self.create_manifest:
            self.manifest = self._create_experiment_manifest()
        
        # Pre-generate injection configurations
        self.injection_configs = self._pre_generate_injection_configs()
        
        # Save manifest after injection configs are generated
        if self.manifest:
            self._save_manifest()
    
    def _setup_results_directory(self) -> str:
        """Setup results directory with Docker volume support."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if running in Docker with mounted volume
        docker_volume_path = "/app/fault_injection/optimizer_comparison_results"
        if os.path.exists(docker_volume_path):
            results_dir = os.path.join(docker_volume_path, f"run_{timestamp}")
        else:
            # Fallback for non-Docker runs
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            results_dir = os.path.join(base_path, f"optimizer_comparison_results_{timestamp}")
        
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def _create_experiment_manifest(self) -> RunManifest:
        """Create manifest for the experiment run."""
        manifest = create_manifest_for_run(capture_all=True)
        
        # Set determinism seeds if available
        if self.determinism_controller:
            manifest.set_seeds(self.determinism_controller.seeds)
        else:
            manifest.set_seeds({'global': self.base_seed})
        
        # Set optimizer information
        manifest.set_optimizer_info(
            primary_optimizer={
                'name': self.baseline_optimizer,
                'learning_rate': self.learning_rate,
            },
            shadow_optimizers=[]  # Will be populated when shadows are created
        )
        
        # Set training information
        manifest.set_training_info(
            total_steps=self.max_steps_after_injection,
            checkpoint_step=0,  # Will be set when checkpoint is created
            learning_rate_schedule={
                'type': 'polynomial_decay',
                'initial_lr': self.learning_rate,
                'decay_steps': 5000,
                'end_lr': 0.0001,
            },
            loss_function='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            mixed_precision=False
        )
        
        # Add experiment-specific fields
        manifest.add_custom_field('num_experiments', self.num_experiments)
        manifest.add_custom_field('test_optimizers', self.test_optimizers)
        manifest.add_custom_field('fault_model', self.fmodel)
        manifest.add_custom_field('injection_value_range', [self.min_val, self.max_val])
        
        return manifest
    
    def _save_manifest(self) -> None:
        """Save manifest to results directory."""
        if self.manifest:
            manifest_path = os.path.join(self.results_base_dir, 'run_manifest.json')
            self.manifest.save(manifest_path)
    
    def _pre_generate_injection_configs(self) -> List[Dict]:
        """Pre-generate all injection configurations for reproducibility."""
        configs = []
        
        for exp_id in range(self.num_experiments):
            # Use determinism controller if available
            if self.determinism_controller:
                # Create experiment-specific RNG
                exp_rng = self.determinism_controller.create_rng_generator(f"exp_{exp_id}")
                seed = self.base_seed + exp_id
                
                # Use TensorFlow's random for consistency
                injection_position = [
                    int(exp_rng.uniform([], 0, 1000, dtype=tf.int32)),
                    int(exp_rng.uniform([], 0, 100, dtype=tf.int32)),
                    int(exp_rng.uniform([], 0, 100, dtype=tf.int32)),
                    int(exp_rng.uniform([], 0, 100, dtype=tf.int32))
                ]
                
                # Generate injection value
                log_min = np.log10(self.min_val)
                log_max = np.log10(self.max_val)
                log_val = float(exp_rng.uniform([], log_min, log_max))
                injection_value = 10 ** log_val
                
                target_epoch = int(exp_rng.uniform([], 0, self.max_target_epoch + 1, dtype=tf.int32))
                target_step = int(exp_rng.uniform([], 0, self.max_target_step + 1, dtype=tf.int32))
            else:
                # Fallback to original method
                seed = self.base_seed + exp_id
                
                # Set seed for this config generation
                random.seed(seed)
                np.random.seed(seed)
                
                # Generate injection position and value deterministically
                injection_position = [
                    np.random.randint(0, 1000),
                    np.random.randint(0, 100),
                    np.random.randint(0, 100),
                    np.random.randint(0, 100)
                ]
                
                # Generate injection value in range
                log_min = np.log10(self.min_val)
                log_max = np.log10(self.max_val)
                injection_value = 10 ** np.random.uniform(log_min, log_max)
                
                target_epoch = random.randint(0, self.max_target_epoch)
                target_step = random.randint(0, self.max_target_step)
            
            # Import choose_random_layer here to avoid circular imports
            from fault_injection.models.inject_utils import choose_random_layer
            
            # Temporarily set seeds for layer selection
            random.seed(seed)
            np.random.seed(seed)
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
        
        # Save configs for reference
        configs_path = os.path.join(self.results_base_dir, 'all_injection_configs.json')
        with open(configs_path, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        
        return configs
    
    def get_injection_config(self, experiment_id: int) -> Dict:
        """Get injection configuration for specific experiment."""
        return self.injection_configs[experiment_id]
    
    def log_configuration(self):
        """Log experiment configuration details."""
        print("\n" + "="*80)
        print("EXPERIMENT CONFIGURATION")
        print("="*80)
        print(f"📊 SETTINGS:")
        print(f"   • Baseline optimizer: {self.baseline_optimizer}")
        print(f"   • Test optimizers: {self.test_optimizers}")
        print(f"   • Number of experiments: {self.num_experiments}")
        print(f"   • Base seed: {self.base_seed}")
        print(f"   • Learning rate: {self.learning_rate}")
        print(f"   • Steps after injection: {self.max_steps_after_injection}")
        print(f"\n🎲 DETERMINISM:")
        print(f"   • Determinism enabled: {self.enable_determinism}")
        if self.determinism_controller:
            print(f"   • Global seed: {self.determinism_controller.global_seed}")
            print(f"   • TF deterministic ops: ENABLED")
        print(f"\n📝 MANIFEST:")
        print(f"   • Manifest tracking: {self.create_manifest}")
        if self.manifest:
            print(f"   • Run ID: {self.manifest.run_id}")
            print(f"   • Schema version: {self.manifest.SCHEMA_VERSION}")
        print(f"\n🎯 FAULT INJECTION:")
        print(f"   • Fault model: {self.fmodel}")
        print(f"   • Value range: [{self.min_val:.2e}, {self.max_val:.2e}]")
        print(f"   • Max target epoch: {self.max_target_epoch}")
        print(f"   • Max target step: {self.max_target_step}")
        print(f"\n📁 OUTPUT:")
        print(f"   • Results directory: {self.results_base_dir}")
        print("="*80)
    
    def get_determinism_controller(self) -> Optional[DeterminismController]:
        """Get the determinism controller if available."""
        return self.determinism_controller
    
    def get_manifest(self) -> Optional[RunManifest]:
        """Get the run manifest if available."""
        return self.manifest
    
    def update_manifest_model_info(self, model_name: str, num_parameters: int, layers: List[str]):
        """Update manifest with model information."""
        if self.manifest:
            self.manifest.set_model_info(
                model_name=model_name,
                architecture='ResNet18',
                num_parameters=num_parameters,
                layers=layers
            )
            self._save_manifest()
    
    def update_manifest_dataset_info(self, name: str, split: str, num_samples: int, batch_size: int):
        """Update manifest with dataset information."""
        if self.manifest:
            self.manifest.set_dataset_info(
                name=name,
                split=split,
                num_samples=num_samples,
                batch_size=batch_size
            )
            self._save_manifest()