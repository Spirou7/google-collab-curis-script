"""
Shadow Adagrad Optimizer

Tracks Adagrad optimizer state without updating weights.
"""

import tensorflow as tf
from typing import Dict, List, Optional
from .shadow_base import ShadowOptimizerBase


class ShadowAdagrad(ShadowOptimizerBase):
    """
    Shadow implementation of Adagrad optimizer.
    
    Tracks accumulated sum of squared gradients without updating weights.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 initial_accumulator_value: float = 0.1,
                 epsilon: float = 1e-7,
                 name: str = "ShadowAdagrad"):
        """
        Initialize Shadow Adagrad optimizer.
        
        Args:
            learning_rate: Learning rate
            initial_accumulator_value: Starting value for accumulators
            epsilon: Small constant for numerical stability
            name: Name of the optimizer
        """
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            initial_accumulator_value=initial_accumulator_value,
            epsilon=epsilon
        )
        
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        
        print(f"📊 Initializing {name}:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Initial accumulator: {initial_accumulator_value}")
        print(f"   • Epsilon: {epsilon}")
    
    def _create_slots(self, var: tf.Variable) -> Dict[str, tf.Variable]:
        """
        Create accumulator slot for Adagrad.
        
        Args:
            var: Model variable to create slots for
            
        Returns:
            Dictionary with accumulator slot
        """
        slots = {}
        
        # Accumulator for sum of squared gradients
        # Initialize to initial_accumulator_value (not zero)
        slots['accumulator'] = tf.Variable(
            tf.ones_like(var) * self.initial_accumulator_value,
            dtype=var.dtype,
            trainable=False,
            name=f"accumulator_{var.name}"
        )
        
        return slots
    
    def _update_slots(self,
                     grad: tf.Tensor,
                     var: tf.Variable,
                     slots: Dict[str, tf.Variable]) -> None:
        """
        Update Adagrad accumulator.
        
        Adagrad update rule:
        accumulator_t = accumulator_{t-1} + gradient^2
        
        The actual parameter update would be:
        theta_t = theta_{t-1} - lr * gradient / sqrt(accumulator_t + epsilon)
        
        Args:
            grad: Gradient for this variable
            var: Model variable (not updated)
            slots: Dictionary with accumulator slot
        """
        accumulator = slots['accumulator']
        
        # Add squared gradient to accumulator
        accumulator.assign_add(tf.square(grad))
    
    def get_effective_gradient(self,
                              var: tf.Variable,
                              grad: tf.Tensor) -> tf.Tensor:
        """
        Get the effective gradient that Adagrad would use for updates.
        
        This shows the adaptive learning rate effect.
        
        Args:
            var: Model variable
            grad: Current gradient
            
        Returns:
            Effective gradient after Adagrad scaling
        """
        accumulator = self.get_slot(var, 'accumulator')
        
        if accumulator is None:
            return grad
        
        # Apply Adagrad scaling
        return grad / (tf.sqrt(accumulator) + self.epsilon)
    
    def get_adaptive_learning_rates(self, var: tf.Variable) -> tf.Tensor:
        """
        Get the per-parameter adaptive learning rates.
        
        Args:
            var: Model variable
            
        Returns:
            Adaptive learning rates for each parameter
        """
        accumulator = self.get_slot(var, 'accumulator')
        
        if accumulator is None:
            return tf.ones_like(var) * self.learning_rate
        
        # Effective per-parameter learning rate
        return self.learning_rate / (tf.sqrt(accumulator) + self.epsilon)
    
    def get_keras_compatible_config(self) -> Dict[str, any]:
        """
        Get configuration compatible with tf.keras.optimizers.Adagrad.
        
        Returns:
            Configuration dictionary for Keras Adagrad
        """
        return {
            'learning_rate': self.learning_rate,
            'initial_accumulator_value': self.initial_accumulator_value,
            'epsilon': self.epsilon,
        }
    
    def transplant_to_keras_slots(self,
                                 keras_optimizer: tf.keras.optimizers.Optimizer,
                                 variables: List[tf.Variable]) -> None:
        """
        Transplant shadow state into a Keras Adagrad optimizer.
        
        Args:
            keras_optimizer: Target Keras Adagrad optimizer
            variables: Model variables
        """
        valid_types = (tf.keras.optimizers.Adagrad,)
        if hasattr(tf.keras.optimizers, 'legacy'):
            valid_types += (tf.keras.optimizers.legacy.Adagrad,)
        
        if not isinstance(keras_optimizer, valid_types):
            raise ValueError(f"Expected Adagrad optimizer, got {type(keras_optimizer)}")
        
        print(f"\n🔄 Transplanting {self.name} state to Keras Adagrad")
        
        # Transplant iteration count
        if hasattr(keras_optimizer, 'iterations'):
            keras_optimizer.iterations.assign(self.iterations)
            print(f"   • Iterations: {self.iterations.numpy()}")
        
        # Transplant accumulators
        transplanted = 0
        
        for var in variables:
            var_key = var.ref()
            
            if var_key not in self.slots:
                continue
            
            shadow_slots = self.slots[var_key]
            
            # Transplant accumulator
            if 'accumulator' in shadow_slots:
                shadow_acc = shadow_slots['accumulator']
                
                # Adagrad uses 'accumulator' as slot name
                keras_acc = keras_optimizer.get_slot(var, 'accumulator')
                
                if keras_acc is None:
                    # Force slot creation
                    self._force_keras_slot_creation(keras_optimizer, var)
                    keras_acc = keras_optimizer.get_slot(var, 'accumulator')
                
                if keras_acc is not None:
                    keras_acc.assign(shadow_acc)
                    transplanted += 1
        
        print(f"   • Transplanted {transplanted} accumulators")
        print(f"   ✓ Transplant complete")
    
    def _force_keras_slot_creation(self,
                                  keras_optimizer: tf.keras.optimizers.Optimizer,
                                  var: tf.Variable) -> None:
        """
        Force Keras optimizer to create slots for a variable.
        
        Args:
            keras_optimizer: Keras optimizer
            var: Variable to create slots for
        """
        # Do a dummy gradient update to force slot creation
        with tf.GradientTape() as tape:
            dummy_loss = tf.reduce_sum(var * 0)
        dummy_grad = tape.gradient(dummy_loss, [var])
        keras_optimizer.apply_gradients(zip(dummy_grad, [var]))
    
    def analyze_learning_rate_decay(self) -> Dict[str, float]:
        """
        Analyze how much the learning rates have decayed.
        
        Returns:
            Statistics about learning rate decay
        """
        if not self._built:
            return {}
        
        all_accumulators = []
        for var_slots in self.slots.values():
            if 'accumulator' in var_slots:
                acc = var_slots['accumulator'].numpy().flatten()
                all_accumulators.append(acc)
        
        if not all_accumulators:
            return {}
        
        all_acc = tf.concat(all_accumulators, axis=0)
        
        # Calculate effective learning rate reduction factors
        reduction_factors = 1.0 / (tf.sqrt(all_acc) + self.epsilon)
        
        return {
            'mean_reduction': float(tf.reduce_mean(reduction_factors)),
            'min_reduction': float(tf.reduce_min(reduction_factors)),
            'max_reduction': float(tf.reduce_max(reduction_factors)),
            'median_reduction': float(tf.experimental.numpy.median(reduction_factors)),
        }