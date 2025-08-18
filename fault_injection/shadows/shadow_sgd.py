"""
Shadow SGD Optimizer

Tracks SGD with momentum state without updating weights.
"""

import tensorflow as tf
from typing import Dict, List, Optional
from .shadow_base import ShadowOptimizerBase


class ShadowSGD(ShadowOptimizerBase):
    """
    Shadow implementation of SGD with optional momentum.
    
    Tracks momentum buffers without updating model weights.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 name: str = "ShadowSGD"):
        """
        Initialize Shadow SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 = no momentum)
            nesterov: Whether to use Nesterov momentum
            name: Name of the optimizer
        """
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            momentum=momentum,
            nesterov=nesterov
        )
        
        self.momentum = momentum
        self.nesterov = nesterov
        self.use_momentum = momentum > 0
        
        print(f"📊 Initializing {name}:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Momentum: {momentum}")
        print(f"   • Nesterov: {nesterov}")
    
    def _create_slots(self, var: tf.Variable) -> Dict[str, tf.Variable]:
        """
        Create momentum slot for SGD.
        
        Args:
            var: Model variable to create slots for
            
        Returns:
            Dictionary with momentum slot (if momentum > 0)
        """
        slots = {}
        
        if self.use_momentum:
            # Create momentum buffer initialized to zeros
            slots['momentum'] = tf.Variable(
                tf.zeros_like(var),
                dtype=var.dtype,
                trainable=False,
                name=f"momentum_{var.name}"
            )
        
        return slots
    
    def _update_slots(self,
                     grad: tf.Tensor,
                     var: tf.Variable,
                     slots: Dict[str, tf.Variable]) -> None:
        """
        Update momentum buffer based on gradient.
        
        Implements SGD momentum update:
        v_t = momentum * v_{t-1} + gradient
        
        For Nesterov momentum:
        v_t = momentum * v_{t-1} + gradient
        (The actual parameter update would use momentum * v_t + gradient,
         but we don't update parameters in shadows)
        
        Args:
            grad: Gradient for this variable
            var: Model variable (not updated)
            slots: Dictionary with momentum slot
        """
        if self.use_momentum and 'momentum' in slots:
            momentum_var = slots['momentum']
            
            # Standard momentum update: v = momentum * v + grad
            # Note: We don't scale by learning rate here since we're not updating weights
            momentum_var.assign(
                self.momentum * momentum_var + grad
            )
            
            # For Nesterov, the actual update would use the "look-ahead" gradient,
            # but since we're not updating weights, we just track the momentum
    
    def get_effective_gradient(self, 
                              var: tf.Variable,
                              grad: tf.Tensor) -> tf.Tensor:
        """
        Get the effective gradient that would be used for weight update.
        
        This is useful for analysis but not used in the shadow optimizer itself.
        
        Args:
            var: Model variable
            grad: Current gradient
            
        Returns:
            Effective gradient after momentum
        """
        if not self.use_momentum:
            return grad
        
        momentum_buffer = self.get_slot(var, 'momentum')
        if momentum_buffer is None:
            return grad
        
        if self.nesterov:
            # Nesterov look-ahead: grad + momentum * v_next
            # v_next = momentum * v + grad
            v_next = self.momentum * momentum_buffer + grad
            return grad + self.momentum * v_next
        else:
            # Standard momentum: just use the momentum buffer
            return momentum_buffer
    
    def get_keras_compatible_config(self) -> Dict[str, any]:
        """
        Get configuration compatible with tf.keras.optimizers.SGD.
        
        Returns:
            Configuration dictionary for Keras SGD
        """
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
        }
    
    def transplant_to_keras_slots(self, 
                                 keras_optimizer: tf.keras.optimizers.Optimizer,
                                 variables: List[tf.Variable]) -> None:
        """
        Transplant shadow state into a Keras SGD optimizer.
        
        Args:
            keras_optimizer: Target Keras SGD optimizer
            variables: Model variables
        """
        if not isinstance(keras_optimizer, (tf.keras.optimizers.SGD, 
                                           tf.keras.optimizers.legacy.SGD)):
            raise ValueError(f"Expected SGD optimizer, got {type(keras_optimizer)}")
        
        print(f"\n🔄 Transplanting {self.name} state to Keras SGD")
        
        # Transplant iteration count
        if hasattr(keras_optimizer, 'iterations'):
            keras_optimizer.iterations.assign(self.iterations)
            print(f"   • Iterations: {self.iterations.numpy()}")
        
        # Transplant momentum buffers
        if self.use_momentum:
            transplanted = 0
            for var in variables:
                var_key = var.ref()
                
                if var_key in self.slots and 'momentum' in self.slots[var_key]:
                    shadow_momentum = self.slots[var_key]['momentum']
                    
                    # Get or create Keras momentum slot
                    keras_momentum = keras_optimizer.get_slot(var, 'momentum')
                    if keras_momentum is None:
                        # Force creation by doing a dummy update
                        with tf.GradientTape() as tape:
                            dummy_loss = tf.reduce_sum(var * 0)
                        dummy_grad = tape.gradient(dummy_loss, [var])
                        keras_optimizer.apply_gradients(zip(dummy_grad, [var]))
                        keras_momentum = keras_optimizer.get_slot(var, 'momentum')
                    
                    if keras_momentum is not None:
                        keras_momentum.assign(shadow_momentum)
                        transplanted += 1
            
            print(f"   • Transplanted {transplanted} momentum buffers")
        
        print(f"   ✓ Transplant complete")