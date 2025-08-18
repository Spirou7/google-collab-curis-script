"""
Shadow Optimizer Package

Provides shadow implementations of popular optimizers that track state
without updating model weights.
"""

from .shadow_base import ShadowOptimizerBase
from .shadow_sgd import ShadowSGD
from .shadow_adam import ShadowAdam
from .shadow_rmsprop import ShadowRMSProp
from .shadow_adagrad import ShadowAdagrad

# Export all shadow optimizers
__all__ = [
    'ShadowOptimizerBase',
    'ShadowSGD',
    'ShadowAdam',
    'ShadowRMSProp',
    'ShadowAdagrad',
    'create_shadow_optimizer',
    'create_shadow_optimizers',
    'get_shadow_optimizer_names',
]


def create_shadow_optimizer(optimizer_name: str, **kwargs) -> ShadowOptimizerBase:
    """
    Factory function to create a shadow optimizer by name.
    
    Args:
        optimizer_name: Name of the optimizer ('sgd', 'adam', 'rmsprop', 'adagrad')
        **kwargs: Optimizer-specific parameters
        
    Returns:
        Shadow optimizer instance
        
    Raises:
        ValueError: If optimizer name is not recognized
    """
    optimizer_map = {
        'sgd': ShadowSGD,
        'adam': ShadowAdam,
        'adamw': ShadowAdam,  # AdamW uses same shadow as Adam
        'rmsprop': ShadowRMSProp,
        'adagrad': ShadowAdagrad,
    }
    
    optimizer_name_lower = optimizer_name.lower()
    
    if optimizer_name_lower not in optimizer_map:
        available = ', '.join(optimizer_map.keys())
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available options: {available}"
        )
    
    # Create shadow optimizer with custom name
    shadow_class = optimizer_map[optimizer_name_lower]
    shadow_name = f"Shadow{optimizer_name.upper()}"
    
    return shadow_class(name=shadow_name, **kwargs)


def create_shadow_optimizers(optimizer_names: list, **kwargs) -> dict:
    """
    Create multiple shadow optimizers at once.
    
    Args:
        optimizer_names: List of optimizer names
        **kwargs: Common parameters for all optimizers
        
    Returns:
        Dictionary mapping optimizer names to shadow instances
    """
    shadows = {}
    
    for name in optimizer_names:
        try:
            shadow = create_shadow_optimizer(name, **kwargs)
            shadows[name] = shadow
        except ValueError as e:
            print(f"⚠️ Skipping {name}: {e}")
    
    return shadows


def get_shadow_optimizer_names() -> list:
    """
    Get list of available shadow optimizer names.
    
    Returns:
        List of optimizer names
    """
    return ['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad']


def get_optimizer_config_defaults(optimizer_name: str) -> dict:
    """
    Get default configuration for a specific optimizer.
    
    Args:
        optimizer_name: Name of the optimizer
        
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'sgd': {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'nesterov': False,
        },
        'adam': {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'amsgrad': False,
        },
        'adamw': {
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'amsgrad': False,
        },
        'rmsprop': {
            'learning_rate': 0.001,
            'rho': 0.9,
            'momentum': 0.0,
            'epsilon': 1e-7,
            'centered': False,
        },
        'adagrad': {
            'learning_rate': 0.01,
            'initial_accumulator_value': 0.1,
            'epsilon': 1e-7,
        },
    }
    
    return defaults.get(optimizer_name.lower(), {})