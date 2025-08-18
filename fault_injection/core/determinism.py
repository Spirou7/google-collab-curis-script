"""
Determinism Control Module for Reproducible Experiments

This module provides comprehensive determinism controls for TensorFlow training,
ensuring that experiments can be perfectly reproduced given the same seed.
"""

import os
import random
import hashlib
import json
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf


class DeterminismController:
    """
    Manages all aspects of determinism in TensorFlow experiments.
    
    This includes:
    - Global seed management
    - TensorFlow operation determinism
    - Dataset ordering and shuffling
    - Augmentation seeding
    """
    
    def __init__(self, global_seed: int = 42):
        """
        Initialize the determinism controller.
        
        Args:
            global_seed: Master seed for all random number generators
        """
        self.global_seed = global_seed
        self.seeds = self._generate_seeds()
        self.dataset_options = None
        self._initialized = False
        
    def _generate_seeds(self) -> Dict[str, int]:
        """
        Generate domain-specific seeds from the global seed.
        
        Returns:
            Dictionary mapping seed domains to their values
        """
        # Use a deterministic hash to generate seeds for different components
        base = str(self.global_seed).encode()
        
        seeds = {
            'global': self.global_seed,
            'python': int(hashlib.md5(base + b'python').hexdigest()[:8], 16),
            'numpy': int(hashlib.md5(base + b'numpy').hexdigest()[:8], 16),
            'tensorflow': int(hashlib.md5(base + b'tensorflow').hexdigest()[:8], 16),
            'data': int(hashlib.md5(base + b'data').hexdigest()[:8], 16),
            'augmentation': int(hashlib.md5(base + b'augmentation').hexdigest()[:8], 16),
            'model_init': int(hashlib.md5(base + b'model_init').hexdigest()[:8], 16),
        }
        
        return seeds
    
    def seed_all(self) -> None:
        """
        Set all random seeds for complete determinism.
        """
        print(f"\n🎲 SETTING DETERMINISTIC SEEDS")
        print(f"   • Global seed: {self.global_seed}")
        
        # Python's built-in random
        random.seed(self.seeds['python'])
        print(f"   • Python random seed: {self.seeds['python']}")
        
        # NumPy
        np.random.seed(self.seeds['numpy'])
        print(f"   • NumPy seed: {self.seeds['numpy']}")
        
        # TensorFlow
        tf.random.set_seed(self.seeds['tensorflow'])
        print(f"   • TensorFlow seed: {self.seeds['tensorflow']}")
        
        # Environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(self.seeds['python'])
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # TensorFlow operation-level determinism
        tf.config.experimental.enable_op_determinism()
        print(f"   • TensorFlow operation determinism: ENABLED")
        
        self._initialized = True
        print(f"   ✓ All seeds set for deterministic execution")
    
    def get_dataset_options(self) -> tf.data.Options:
        """
        Get deterministic dataset options.
        
        Returns:
            tf.data.Options configured for determinism
        """
        if self.dataset_options is None:
            options = tf.data.Options()
            
            # Enable deterministic ordering
            options.experimental_deterministic = True
            
            # Disable parallel processing that could introduce non-determinism
            options.threading.max_intra_op_parallelism = 1
            options.threading.private_threadpool_size = 1
            
            # Set seed for any random operations in the dataset
            options.experimental_seed = self.seeds['data']
            
            self.dataset_options = options
            
        return self.dataset_options
    
    def make_dataset_deterministic(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Apply deterministic options to a dataset.
        
        Args:
            dataset: The dataset to make deterministic
            
        Returns:
            Dataset with deterministic behavior
        """
        if not self._initialized:
            raise RuntimeError("DeterminismController not initialized. Call seed_all() first.")
        
        # Apply deterministic options
        dataset = dataset.with_options(self.get_dataset_options())
        
        return dataset
    
    def get_augmentation_seed(self, step: int = 0) -> int:
        """
        Get a deterministic seed for data augmentation at a given step.
        
        Args:
            step: Current training step
            
        Returns:
            Seed for augmentation operations
        """
        # Combine augmentation seed with step for step-specific determinism
        combined = f"{self.seeds['augmentation']}_{step}"
        return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
    
    def get_model_init_seed(self) -> int:
        """
        Get seed for model weight initialization.
        
        Returns:
            Seed for initializing model weights
        """
        return self.seeds['model_init']
    
    def create_rng_generator(self, key: str) -> tf.random.Generator:
        """
        Create a stateful random number generator for a specific purpose.
        
        Args:
            key: Identifier for this generator's purpose
            
        Returns:
            Stateful RNG generator
        """
        # Generate a unique seed for this generator
        seed_bytes = f"{self.global_seed}_{key}".encode()
        seed = int(hashlib.md5(seed_bytes).hexdigest()[:8], 16)
        
        # Create stateful generator
        return tf.random.Generator.from_seed(seed)
    
    def verify_determinism(self, 
                          func,
                          num_runs: int = 3,
                          tolerance: float = 1e-7) -> bool:
        """
        Verify that a function produces deterministic results.
        
        Args:
            func: Function to test (should return a tensor or numeric value)
            num_runs: Number of times to run the function
            tolerance: Maximum allowed difference between runs
            
        Returns:
            True if function is deterministic within tolerance
        """
        results = []
        
        for i in range(num_runs):
            # Reset seeds before each run
            self.seed_all()
            result = func()
            
            # Convert to numpy for comparison
            if isinstance(result, tf.Tensor):
                result = result.numpy()
            
            results.append(result)
        
        # Check if all results are identical
        for i in range(1, num_runs):
            diff = np.abs(results[i] - results[0])
            max_diff = np.max(diff) if isinstance(diff, np.ndarray) else diff
            
            if max_diff > tolerance:
                print(f"⚠️ Determinism check failed: max difference = {max_diff}")
                return False
        
        print(f"✓ Determinism verified over {num_runs} runs")
        return True
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the current state of the determinism controller.
        
        Returns:
            Dictionary containing all determinism settings
        """
        return {
            'global_seed': self.global_seed,
            'seeds': self.seeds,
            'initialized': self._initialized,
            'tf_deterministic_ops': os.environ.get('TF_DETERMINISTIC_OPS', '0'),
            'tf_version': tf.__version__,
        }
    
    def compute_dataset_hash(self, 
                           dataset: tf.data.Dataset,
                           num_samples: int = 100) -> str:
        """
        Compute a hash of dataset samples to verify ordering.
        
        Args:
            dataset: Dataset to hash
            num_samples: Number of samples to include in hash
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        
        for i, sample in enumerate(dataset.take(num_samples)):
            # Convert sample to bytes for hashing
            if isinstance(sample, tuple):
                # Handle (input, label) format
                for tensor in sample:
                    hasher.update(tensor.numpy().tobytes())
            else:
                hasher.update(sample.numpy().tobytes())
            
            if i >= num_samples - 1:
                break
        
        return hasher.hexdigest()
    
    def log_state(self) -> None:
        """
        Log the current determinism state for debugging.
        """
        print("\n📊 DETERMINISM STATE:")
        state = self.get_state_dict()
        for key, value in state.items():
            if isinstance(value, dict):
                print(f"   • {key}:")
                for k, v in value.items():
                    print(f"     - {k}: {v}")
            else:
                print(f"   • {key}: {value}")


# Global instance for easy access
_global_controller = None


def get_global_controller(seed: Optional[int] = None) -> DeterminismController:
    """
    Get or create the global determinism controller.
    
    Args:
        seed: Optional seed to use if creating new controller
        
    Returns:
        Global DeterminismController instance
    """
    global _global_controller
    
    if _global_controller is None:
        if seed is None:
            seed = 42  # Default seed
        _global_controller = DeterminismController(seed)
    
    return _global_controller


def seed_everything(seed: int = 42) -> DeterminismController:
    """
    Convenience function to seed everything at once.
    
    Args:
        seed: Global seed value
        
    Returns:
        Configured DeterminismController
    """
    controller = get_global_controller(seed)
    controller.seed_all()
    return controller


def make_reproducible(dataset: tf.data.Dataset, 
                     shuffle: bool = True,
                     shuffle_buffer: int = 10000,
                     seed: Optional[int] = None) -> tf.data.Dataset:
    """
    Make a dataset reproducible with optional shuffling.
    
    Args:
        dataset: Input dataset
        shuffle: Whether to shuffle (deterministically)
        shuffle_buffer: Size of shuffle buffer
        seed: Shuffle seed (uses controller's data seed if None)
        
    Returns:
        Reproducible dataset
    """
    controller = get_global_controller()
    
    # Apply deterministic options
    dataset = controller.make_dataset_deterministic(dataset)
    
    # Apply deterministic shuffling if requested
    if shuffle:
        if seed is None:
            seed = controller.seeds['data']
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            seed=seed,
            reshuffle_each_iteration=False  # Critical for reproducibility
        )
    
    return dataset