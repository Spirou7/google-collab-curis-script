"""
Run Manifest System for Experiment Tracking

This module provides a comprehensive manifest system to track all aspects
of an experiment run, ensuring complete reproducibility and traceability.
"""

import os
import json
import hashlib
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import tensorflow as tf


class RunManifest:
    """
    Comprehensive manifest for tracking experiment runs.
    
    Captures:
    - Environment information
    - Seeds and determinism settings
    - Model architecture
    - Optimizer configurations
    - Dataset information
    - Training hyperparameters
    - Git commit information
    - Hardware specifications
    """
    
    SCHEMA_VERSION = 2  # Increment when making breaking changes
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize a new run manifest.
        
        Args:
            run_id: Unique identifier for this run (auto-generated if None)
        """
        if run_id is None:
            run_id = self._generate_run_id()
        
        self.run_id = run_id
        self.created_at = datetime.now().isoformat()
        self.data = {
            'schema_version': self.SCHEMA_VERSION,
            'run_id': run_id,
            'created_at': self.created_at,
            'environment': {},
            'seeds': {},
            'model': {},
            'optimizer': {},
            'dataset': {},
            'training': {},
            'checkpoint': {},
            'git': {},
            'hardware': {},
            'shadows': {},
        }
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:6]
        return f"exp_{timestamp}_{random_suffix}"
    
    def set_environment(self) -> None:
        """Capture environment information."""
        self.data['environment'] = {
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__,
            'platform': platform.platform(),
            'hostname': platform.node(),
            'working_directory': os.getcwd(),
        }
        
        # Capture key environment variables
        env_vars = ['PYTHONHASHSEED', 'TF_DETERMINISTIC_OPS', 'TF_CUDNN_DETERMINISTIC']
        self.data['environment']['env_vars'] = {
            var: os.environ.get(var, 'not set') for var in env_vars
        }
    
    def set_seeds(self, seeds: Dict[str, int]) -> None:
        """
        Set seed information.
        
        Args:
            seeds: Dictionary of seed values by domain
        """
        self.data['seeds'] = seeds
    
    def set_model_info(self, 
                      model_name: str,
                      architecture: str,
                      num_parameters: int,
                      layers: Optional[List[str]] = None,
                      config: Optional[Dict] = None) -> None:
        """
        Set model information.
        
        Args:
            model_name: Name of the model
            architecture: Architecture type (e.g., 'ResNet18')
            num_parameters: Total number of trainable parameters
            layers: List of layer names
            config: Additional model configuration
        """
        self.data['model'] = {
            'name': model_name,
            'architecture': architecture,
            'num_parameters': num_parameters,
            'layers': layers or [],
            'config': config or {},
        }
    
    def set_optimizer_info(self,
                         primary_optimizer: Dict[str, Any],
                         shadow_optimizers: Optional[List[str]] = None) -> None:
        """
        Set optimizer information.
        
        Args:
            primary_optimizer: Primary optimizer configuration
            shadow_optimizers: List of shadow optimizer names
        """
        self.data['optimizer'] = {
            'primary': primary_optimizer,
            'shadows': shadow_optimizers or [],
        }
    
    def set_dataset_info(self,
                       name: str,
                       split: str,
                       num_samples: int,
                       batch_size: int,
                       preprocessing: Optional[Dict] = None,
                       order_hash: Optional[str] = None) -> None:
        """
        Set dataset information.
        
        Args:
            name: Dataset name
            split: Data split (train/val/test)
            num_samples: Number of samples
            batch_size: Batch size
            preprocessing: Preprocessing configuration
            order_hash: Hash of dataset ordering for verification
        """
        self.data['dataset'] = {
            'name': name,
            'split': split,
            'num_samples': num_samples,
            'batch_size': batch_size,
            'preprocessing': preprocessing or {},
            'order_hash': order_hash,
        }
    
    def set_training_info(self,
                        total_steps: int,
                        checkpoint_step: int,
                        learning_rate_schedule: Dict[str, Any],
                        loss_function: str,
                        metrics: List[str],
                        mixed_precision: bool = False) -> None:
        """
        Set training configuration.
        
        Args:
            total_steps: Total training steps
            checkpoint_step: Step at which checkpoint is saved
            learning_rate_schedule: LR schedule configuration
            loss_function: Name of loss function
            metrics: List of metric names
            mixed_precision: Whether mixed precision is enabled
        """
        self.data['training'] = {
            'total_steps': total_steps,
            'checkpoint_step': checkpoint_step,
            'learning_rate_schedule': learning_rate_schedule,
            'loss_function': loss_function,
            'metrics': metrics,
            'mixed_precision': mixed_precision,
        }
    
    def set_checkpoint_info(self,
                          checkpoint_dir: str,
                          checkpoint_step: int,
                          includes_shadows: bool,
                          includes_buffers: bool,
                          includes_scheduler: bool) -> None:
        """
        Set checkpoint information.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            checkpoint_step: Step number of checkpoint
            includes_shadows: Whether shadow states are included
            includes_buffers: Whether model buffers are included
            includes_scheduler: Whether scheduler state is included
        """
        self.data['checkpoint'] = {
            'directory': checkpoint_dir,
            'step': checkpoint_step,
            'includes_shadows': includes_shadows,
            'includes_buffers': includes_buffers,
            'includes_scheduler': includes_scheduler,
            'saved_at': datetime.now().isoformat(),
        }
    
    def set_git_info(self) -> None:
        """Capture git repository information."""
        try:
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Check for uncommitted changes
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            has_changes = len(status) > 0
            
            # Get commit message
            commit_message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            self.data['git'] = {
                'commit_hash': commit_hash,
                'branch': branch,
                'has_uncommitted_changes': has_changes,
                'commit_message': commit_message[:200],  # Truncate long messages
            }
            
            if has_changes:
                # Save list of changed files
                changed_files = [line.split()[-1] for line in status.split('\n') if line]
                self.data['git']['changed_files'] = changed_files[:20]  # Limit to 20 files
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.data['git'] = {'error': 'Git information unavailable'}
    
    def set_hardware_info(self) -> None:
        """Capture hardware information."""
        self.data['hardware'] = {
            'cpu_count': os.cpu_count(),
            'platform_machine': platform.machine(),
            'platform_processor': platform.processor(),
        }
        
        # TensorFlow device information
        devices = tf.config.list_physical_devices()
        self.data['hardware']['tf_devices'] = [
            {
                'name': device.name,
                'type': device.device_type
            }
            for device in devices
        ]
        
        # Memory info (if available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.data['hardware']['memory_gb'] = round(memory.total / (1024**3), 2)
        except ImportError:
            pass
    
    def set_shadow_info(self, shadows: Dict[str, Dict[str, Any]]) -> None:
        """
        Set shadow optimizer information.
        
        Args:
            shadows: Dictionary mapping shadow names to their configurations
        """
        self.data['shadows'] = shadows
    
    def add_custom_field(self, key: str, value: Any) -> None:
        """
        Add a custom field to the manifest.
        
        Args:
            key: Field name
            value: Field value
        """
        if 'custom' not in self.data:
            self.data['custom'] = {}
        self.data['custom'][key] = value
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save manifest to JSON file.
        
        Args:
            filepath: Path to save the manifest
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        print(f"✓ Manifest saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'RunManifest':
        """
        Load manifest from JSON file.
        
        Args:
            filepath: Path to the manifest file
            
        Returns:
            Loaded RunManifest instance
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check schema version
        file_version = data.get('schema_version', 1)
        if file_version != cls.SCHEMA_VERSION:
            print(f"⚠️ Warning: Manifest schema version mismatch "
                  f"(file: {file_version}, current: {cls.SCHEMA_VERSION})")
        
        # Create manifest and populate
        manifest = cls(run_id=data.get('run_id'))
        manifest.data = data
        manifest.created_at = data.get('created_at', 'unknown')
        
        return manifest
    
    def validate(self) -> List[str]:
        """
        Validate that all required fields are present.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        required_fields = ['schema_version', 'run_id', 'created_at']
        
        for field in required_fields:
            if field not in self.data:
                errors.append(f"Missing required field: {field}")
        
        # Check for essential sub-fields
        if not self.data.get('seeds'):
            errors.append("No seed information recorded")
        
        if not self.data.get('environment'):
            errors.append("No environment information recorded")
        
        return errors
    
    def compute_hash(self) -> str:
        """
        Compute a hash of the manifest for integrity checking.
        
        Returns:
            SHA256 hash of the manifest
        """
        # Sort keys for consistent hashing
        content = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of the manifest.
        
        Returns:
            Summary string
        """
        lines = [
            f"Run ID: {self.run_id}",
            f"Created: {self.created_at}",
            f"Schema Version: {self.data.get('schema_version', 'unknown')}",
            ""
        ]
        
        if self.data.get('environment'):
            env = self.data['environment']
            lines.extend([
                "Environment:",
                f"  TensorFlow: {env.get('tensorflow_version', 'unknown')}",
                f"  Python: {env.get('python_version', 'unknown')}",
                ""
            ])
        
        if self.data.get('model'):
            model = self.data['model']
            lines.extend([
                "Model:",
                f"  Architecture: {model.get('architecture', 'unknown')}",
                f"  Parameters: {model.get('num_parameters', 'unknown'):,}",
                ""
            ])
        
        if self.data.get('optimizer'):
            opt = self.data['optimizer']
            primary = opt.get('primary', {})
            lines.extend([
                "Optimizer:",
                f"  Primary: {primary.get('name', 'unknown')}",
                f"  Shadows: {', '.join(opt.get('shadows', [])) or 'none'}",
                ""
            ])
        
        if self.data.get('git'):
            git = self.data['git']
            if 'error' not in git:
                lines.extend([
                    "Git:",
                    f"  Commit: {git.get('commit_hash', 'unknown')[:8]}",
                    f"  Branch: {git.get('branch', 'unknown')}",
                    f"  Clean: {'No' if git.get('has_uncommitted_changes') else 'Yes'}",
                ])
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return f"RunManifest(run_id='{self.run_id}', created_at='{self.created_at}')"


def create_manifest_for_run(
    run_id: Optional[str] = None,
    capture_all: bool = True
) -> RunManifest:
    """
    Create a manifest with automatic capture of common information.
    
    Args:
        run_id: Optional run ID
        capture_all: Whether to automatically capture environment, git, hardware
        
    Returns:
        Configured RunManifest
    """
    manifest = RunManifest(run_id)
    
    if capture_all:
        manifest.set_environment()
        manifest.set_git_info()
        manifest.set_hardware_info()
    
    return manifest