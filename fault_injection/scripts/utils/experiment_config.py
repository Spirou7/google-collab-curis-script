import json
import os
import random
import numpy as np
from typing import List, Dict
import datetime


class ExperimentConfig:
    """Configuration and setup for optimizer mitigation experiments."""
    
    def __init__(self, 
                 baseline_optimizer: str = 'adam',
                 test_optimizers: List[str] = None,
                 num_experiments: int = 100,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 max_steps_after_injection: int = 200):
        """Initialize experiment configuration."""
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
        
        # Setup results directory
        self.results_base_dir = self._setup_results_directory()
        
        # Pre-generate injection configurations
        self.injection_configs = self._pre_generate_injection_configs()
    
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
    
    def _pre_generate_injection_configs(self) -> List[Dict]:
        """Pre-generate all injection configurations for reproducibility."""
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
            
            # Import choose_random_layer here to avoid circular imports
            from fault_injection.models.inject_utils import choose_random_layer
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
        print(f"\n🎯 FAULT INJECTION:")
        print(f"   • Fault model: {self.fmodel}")
        print(f"   • Value range: [{self.min_val:.2e}, {self.max_val:.2e}]")
        print(f"   • Max target epoch: {self.max_target_epoch}")
        print(f"   • Max target step: {self.max_target_step}")
        print(f"\n📁 OUTPUT:")
        print(f"   • Results directory: {self.results_base_dir}")
        print("="*80)