import tensorflow as tf
from typing import Optional


class ContinuedPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule that continues from a specific step."""
    
    def __init__(self, initial_lr, decay_steps, end_lr, current_step):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.current_step = current_step
    
    def __call__(self, step):
        # Continue from current position in schedule
        effective_step = step + self.current_step
        completion = tf.minimum(effective_step / self.decay_steps, 1.0)
        current_lr = self.initial_lr + (self.end_lr - self.initial_lr) * completion
        return current_lr
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'decay_steps': self.decay_steps,
            'end_lr': self.end_lr,
            'current_step': self.current_step
        }


def create_optimizer(optimizer_name: str, 
                    learning_rate: float = 0.001,
                    current_step: int = 0) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer with proper learning rate schedule continuation.
    
    Args:
        optimizer_name: Name of the optimizer
        learning_rate: Initial learning rate
        current_step: Current training step for schedule continuation
    
    Returns:
        Configured optimizer instance
    """
    print(f"\n🔨 CREATING OPTIMIZER: {optimizer_name}")
    print(f"   • Initial learning rate: {learning_rate}")
    print(f"   • Current step for schedule: {current_step}")
    
    # Create learning rate schedule
    lr_schedule = ContinuedPolynomialDecay(
        initial_lr=learning_rate,
        decay_steps=5000,
        end_lr=0.0001,
        current_step=current_step
    )
    
    # Calculate current learning rate for logging
    current_lr_value = lr_schedule(0).numpy()
    print(f"   • Current learning rate value: {current_lr_value:.6f}")
    
    # Get optimizer class
    optimizer_class = get_optimizer_class(optimizer_name)
    print(f"   • Optimizer class: {optimizer_class.__name__}")
    
    # Create optimizer with appropriate parameters
    optimizer = create_optimizer_instance(optimizer_name, optimizer_class, lr_schedule)
    
    print(f"   ✓ Optimizer {optimizer_name} created successfully")
    return optimizer


def get_optimizer_class(optimizer_name: str):
    """Get the optimizer class for the given name."""
    # Handle AdamW compatibility across TensorFlow versions
    adamw_optimizer = tf.keras.optimizers.Adam  # Default fallback
    
    try:
        if hasattr(tf.keras.optimizers, 'AdamW'):
            adamw_optimizer = tf.keras.optimizers.AdamW
            print(f"   • Found AdamW in tf.keras.optimizers")
        elif hasattr(tf.keras, 'optimizers'):
            # Check for experimental submodule safely
            try:
                experimental = getattr(tf.keras.optimizers, 'experimental', None)
                if experimental and hasattr(experimental, 'AdamW'):
                    adamw_optimizer = experimental.AdamW
                    print(f"   • Found AdamW in tf.keras.optimizers.experimental")
                else:
                    print(f"   • AdamW not found, using Adam as fallback")
            except AttributeError:
                print(f"   • AdamW not found, using Adam as fallback")
        else:
            print(f"   • AdamW not found, using Adam as fallback")
    except Exception as e:
        print(f"   • Error checking for AdamW: {e}, using Adam as fallback")
    
    optimizer_map = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adamw': adamw_optimizer,
        'adagrad': tf.keras.optimizers.Adagrad,
        'adadelta': tf.keras.optimizers.Adadelta,
        'nadam': tf.keras.optimizers.Nadam,
    }
    
    return optimizer_map.get(optimizer_name.lower(), tf.keras.optimizers.Adam)


def create_optimizer_instance(optimizer_name: str, 
                             optimizer_class, 
                             lr_schedule):
    """Create optimizer instance with appropriate parameters."""
    optimizer_name_lower = optimizer_name.lower()
    
    if optimizer_name_lower == 'sgd':
        print(f"   • Adding momentum: 0.9")
        return optimizer_class(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer_name_lower == 'rmsprop':
        print(f"   • Adding rho: 0.9")
        return optimizer_class(learning_rate=lr_schedule, rho=0.9)
    elif optimizer_name_lower == 'adadelta':
        print(f"   • Adding rho: 0.95")
        return optimizer_class(learning_rate=lr_schedule, rho=0.95)
    else:
        return optimizer_class(learning_rate=lr_schedule)