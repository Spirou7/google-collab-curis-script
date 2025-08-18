"""
Shadow RMSProp Optimizer

Tracks RMSProp optimizer state without updating weights.
"""

import tensorflow as tf
from typing import Dict, List, Optional
from .shadow_base import ShadowOptimizerBase


class ShadowRMSProp(ShadowOptimizerBase):
    """
    Shadow implementation of RMSProp optimizer.
    
    Tracks moving average of squared gradients (and optionally momentum).
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 rho: float = 0.9,
                 momentum: float = 0.0,
                 epsilon: float = 1e-7,
                 centered: bool = False,
                 name: str = "ShadowRMSProp"):
        """
        Initialize Shadow RMSProp optimizer.
        
        Args:
            learning_rate: Learning rate
            rho: Discounting factor for the history of squared gradients
            momentum: Momentum factor
            epsilon: Small constant for numerical stability
            centered: If True, center the gradients before scaling
            name: Name of the optimizer
        """
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered
        )
        
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.use_momentum = momentum > 0
        
        print(f"📊 Initializing {name}:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Rho (decay rate): {rho}")
        print(f"   • Momentum: {momentum}")
        print(f"   • Epsilon: {epsilon}")
        print(f"   • Centered: {centered}")
    
    def _create_slots(self, var: tf.Variable) -> Dict[str, tf.Variable]:
        """
        Create RMSProp slots for a variable.
        
        Args:
            var: Model variable to create slots for
            
        Returns:
            Dictionary with RMS accumulator, optional momentum, and mg for centered
        """
        slots = {}
        
        # RMS accumulator (moving average of squared gradients)
        slots['rms'] = tf.Variable(
            tf.ones_like(var),  # Initialize to 1 as per RMSProp convention
            dtype=var.dtype,
            trainable=False,
            name=f"rms_{var.name}"
        )
        
        # Momentum buffer if using momentum
        if self.use_momentum:
            slots['momentum'] = tf.Variable(
                tf.zeros_like(var),
                dtype=var.dtype,
                trainable=False,
                name=f"momentum_{var.name}"
            )
        
        # For centered RMSProp: track moving average of gradients
        if self.centered:
            slots['mg'] = tf.Variable(
                tf.zeros_like(var),
                dtype=var.dtype,
                trainable=False,
                name=f"mg_{var.name}"
            )
        
        return slots
    
    def _update_slots(self,
                     grad: tf.Tensor,
                     var: tf.Variable,
                     slots: Dict[str, tf.Variable]) -> None:
        """
        Update RMSProp accumulator and optional momentum.
        
        Standard RMSProp:
        rms_t = rho * rms_{t-1} + (1 - rho) * gradient^2
        
        Centered RMSProp:
        mg_t = rho * mg_{t-1} + (1 - rho) * gradient
        rms_t = rho * rms_{t-1} + (1 - rho) * gradient^2
        
        With momentum:
        momentum_t = momentum * momentum_{t-1} + gradient / sqrt(rms_t + epsilon)
        
        Args:
            grad: Gradient for this variable
            var: Model variable (not updated)
            slots: Dictionary with RMSProp slots
        """
        rms = slots['rms']
        
        # Update moving average of squared gradients
        rms.assign(self.rho * rms + (1 - self.rho) * tf.square(grad))
        
        # Update moving average of gradients for centered variant
        if self.centered and 'mg' in slots:
            mg = slots['mg']
            mg.assign(self.rho * mg + (1 - self.rho) * grad)
        
        # Update momentum if used
        if self.use_momentum and 'momentum' in slots:
            momentum_var = slots['momentum']
            
            # Compute the RMSProp-scaled gradient
            if self.centered and 'mg' in slots:
                # Centered: use (rms - mg^2) for variance
                mg = slots['mg']
                scaled_grad = grad / (tf.sqrt(rms - tf.square(mg) + self.epsilon))
            else:
                # Standard RMSProp
                scaled_grad = grad / (tf.sqrt(rms) + self.epsilon)
            
            # Update momentum with scaled gradient
            momentum_var.assign(
                self.momentum * momentum_var + scaled_grad
            )
    
    def get_effective_gradient(self,
                              var: tf.Variable,
                              grad: tf.Tensor) -> tf.Tensor:
        """
        Get the effective gradient that RMSProp would use for updates.
        
        Args:
            var: Model variable
            grad: Current gradient
            
        Returns:
            Effective gradient after RMSProp scaling
        """
        rms = self.get_slot(var, 'rms')
        if rms is None:
            return grad
        
        # Apply RMSProp scaling
        if self.centered:
            mg = self.get_slot(var, 'mg')
            if mg is not None:
                # Centered variance
                effective_grad = grad / (tf.sqrt(rms - tf.square(mg) + self.epsilon))
            else:
                effective_grad = grad / (tf.sqrt(rms) + self.epsilon)
        else:
            effective_grad = grad / (tf.sqrt(rms) + self.epsilon)
        
        # Apply momentum if used
        if self.use_momentum:
            momentum_buffer = self.get_slot(var, 'momentum')
            if momentum_buffer is not None:
                return momentum_buffer
        
        return effective_grad
    
    def get_keras_compatible_config(self) -> Dict[str, any]:
        """
        Get configuration compatible with tf.keras.optimizers.RMSprop.
        
        Returns:
            Configuration dictionary for Keras RMSprop
        """
        return {
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'centered': self.centered,
        }
    
    def transplant_to_keras_slots(self,
                                 keras_optimizer: tf.keras.optimizers.Optimizer,
                                 variables: List[tf.Variable]) -> None:
        """
        Transplant shadow state into a Keras RMSprop optimizer.
        
        Args:
            keras_optimizer: Target Keras RMSprop optimizer
            variables: Model variables
        """
        valid_types = (tf.keras.optimizers.RMSprop,)
        if hasattr(tf.keras.optimizers, 'legacy'):
            valid_types += (tf.keras.optimizers.legacy.RMSprop,)
        
        if not isinstance(keras_optimizer, valid_types):
            raise ValueError(f"Expected RMSprop optimizer, got {type(keras_optimizer)}")
        
        print(f"\n🔄 Transplanting {self.name} state to Keras RMSprop")
        
        # Transplant iteration count
        if hasattr(keras_optimizer, 'iterations'):
            keras_optimizer.iterations.assign(self.iterations)
            print(f"   • Iterations: {self.iterations.numpy()}")
        
        # Transplant RMS accumulator and momentum
        transplanted_rms = 0
        transplanted_momentum = 0
        transplanted_mg = 0
        
        for var in variables:
            var_key = var.ref()
            
            if var_key not in self.slots:
                continue
            
            shadow_slots = self.slots[var_key]
            
            # Transplant RMS accumulator
            if 'rms' in shadow_slots:
                shadow_rms = shadow_slots['rms']
                
                # RMSprop uses 'rms' as slot name
                keras_rms = keras_optimizer.get_slot(var, 'rms')
                
                if keras_rms is None:
                    # Force slot creation
                    self._force_keras_slot_creation(keras_optimizer, var)
                    keras_rms = keras_optimizer.get_slot(var, 'rms')
                
                if keras_rms is not None:
                    keras_rms.assign(shadow_rms)
                    transplanted_rms += 1
            
            # Transplant momentum if used
            if self.use_momentum and 'momentum' in shadow_slots:
                shadow_momentum = shadow_slots['momentum']
                keras_momentum = keras_optimizer.get_slot(var, 'momentum')
                
                if keras_momentum is None:
                    self._force_keras_slot_creation(keras_optimizer, var)
                    keras_momentum = keras_optimizer.get_slot(var, 'momentum')
                
                if keras_momentum is not None:
                    keras_momentum.assign(shadow_momentum)
                    transplanted_momentum += 1
            
            # Transplant mg for centered variant
            if self.centered and 'mg' in shadow_slots:
                shadow_mg = shadow_slots['mg']
                keras_mg = keras_optimizer.get_slot(var, 'mg')
                
                if keras_mg is not None:
                    keras_mg.assign(shadow_mg)
                    transplanted_mg += 1
        
        print(f"   • Transplanted {transplanted_rms} RMS accumulators")
        if self.use_momentum:
            print(f"   • Transplanted {transplanted_momentum} momentum buffers")
        if self.centered:
            print(f"   • Transplanted {transplanted_mg} mg buffers")
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