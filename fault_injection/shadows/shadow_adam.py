"""
Shadow Adam Optimizer

Tracks Adam optimizer state (m and v) without updating weights.
"""

import tensorflow as tf
from typing import Dict, List, Optional
from .shadow_base import ShadowOptimizerBase


class ShadowAdam(ShadowOptimizerBase):
    """
    Shadow implementation of Adam optimizer.
    
    Tracks first moment (m) and second moment (v) estimates without updating weights.
    """
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-7,
                 amsgrad: bool = False,
                 name: str = "ShadowAdam"):
        """
        Initialize Shadow Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta_1: Exponential decay rate for first moment estimates
            beta_2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            amsgrad: Whether to use AMSGrad variant
            name: Name of the optimizer
        """
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad
        )
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        
        print(f"📊 Initializing {name}:")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Beta_1 (momentum): {beta_1}")
        print(f"   • Beta_2 (RMSprop): {beta_2}")
        print(f"   • Epsilon: {epsilon}")
        print(f"   • AMSGrad: {amsgrad}")
    
    def _create_slots(self, var: tf.Variable) -> Dict[str, tf.Variable]:
        """
        Create Adam slots (m and v) for a variable.
        
        Args:
            var: Model variable to create slots for
            
        Returns:
            Dictionary with m and v slots (and vhat for AMSGrad)
        """
        slots = {}
        
        # First moment estimate (momentum)
        slots['m'] = tf.Variable(
            tf.zeros_like(var),
            dtype=var.dtype,
            trainable=False,
            name=f"m_{var.name}"
        )
        
        # Second moment estimate (uncentered variance)
        slots['v'] = tf.Variable(
            tf.zeros_like(var),
            dtype=var.dtype,
            trainable=False,
            name=f"v_{var.name}"
        )
        
        # AMSGrad: track maximum of v
        if self.amsgrad:
            slots['vhat'] = tf.Variable(
                tf.zeros_like(var),
                dtype=var.dtype,
                trainable=False,
                name=f"vhat_{var.name}"
            )
        
        return slots
    
    def _update_slots(self,
                     grad: tf.Tensor,
                     var: tf.Variable,
                     slots: Dict[str, tf.Variable]) -> None:
        """
        Update Adam momentum and variance estimates.
        
        Implements Adam update rules:
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * gradient
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * gradient^2
        
        For AMSGrad:
        vhat_t = max(vhat_{t-1}, v_t)
        
        Args:
            grad: Gradient for this variable
            var: Model variable (not updated)
            slots: Dictionary with m and v slots
        """
        m = slots['m']
        v = slots['v']
        
        # Update biased first moment estimate
        m.assign(self.beta_1 * m + (1 - self.beta_1) * grad)
        
        # Update biased second raw moment estimate
        v.assign(self.beta_2 * v + (1 - self.beta_2) * tf.square(grad))
        
        # Update vhat for AMSGrad
        if self.amsgrad and 'vhat' in slots:
            vhat = slots['vhat']
            vhat.assign(tf.maximum(vhat, v))
    
    def get_bias_corrected_moments(self, var: tf.Variable) -> tuple:
        """
        Get bias-corrected first and second moment estimates.
        
        Args:
            var: Model variable
            
        Returns:
            Tuple of (m_hat, v_hat) - bias corrected moments
        """
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        
        if m is None or v is None:
            return None, None
        
        # Get current iteration (t)
        t = tf.cast(self.iterations, tf.float32)
        
        # Bias correction
        bias_correction1 = 1 - tf.pow(self.beta_1, t)
        bias_correction2 = 1 - tf.pow(self.beta_2, t)
        
        m_hat = m / bias_correction1
        
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            if vhat is not None:
                v_hat = vhat / bias_correction2
            else:
                v_hat = v / bias_correction2
        else:
            v_hat = v / bias_correction2
        
        return m_hat, v_hat
    
    def get_effective_gradient(self,
                              var: tf.Variable) -> tf.Tensor:
        """
        Get the effective gradient that Adam would use for updates.
        
        This shows what the actual parameter update would be.
        
        Args:
            var: Model variable
            
        Returns:
            Effective gradient after Adam transformation
        """
        m_hat, v_hat = self.get_bias_corrected_moments(var)
        
        if m_hat is None or v_hat is None:
            return tf.zeros_like(var)
        
        # Adam update direction (without learning rate)
        return m_hat / (tf.sqrt(v_hat) + self.epsilon)
    
    def get_keras_compatible_config(self) -> Dict[str, any]:
        """
        Get configuration compatible with tf.keras.optimizers.Adam.
        
        Returns:
            Configuration dictionary for Keras Adam
        """
        return {
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        }
    
    def transplant_to_keras_slots(self,
                                 keras_optimizer: tf.keras.optimizers.Optimizer,
                                 variables: List[tf.Variable]) -> None:
        """
        Transplant shadow state into a Keras Adam optimizer.
        
        Args:
            keras_optimizer: Target Keras Adam optimizer
            variables: Model variables
        """
        # Check optimizer type
        valid_types = (tf.keras.optimizers.Adam, tf.keras.optimizers.AdamW)
        if hasattr(tf.keras.optimizers, 'legacy'):
            valid_types += (tf.keras.optimizers.legacy.Adam,)
        
        if not isinstance(keras_optimizer, valid_types):
            raise ValueError(f"Expected Adam/AdamW optimizer, got {type(keras_optimizer)}")
        
        print(f"\n🔄 Transplanting {self.name} state to Keras Adam")
        
        # Transplant iteration count
        if hasattr(keras_optimizer, 'iterations'):
            keras_optimizer.iterations.assign(self.iterations)
            print(f"   • Iterations: {self.iterations.numpy()}")
        
        # Transplant m and v buffers
        transplanted_m = 0
        transplanted_v = 0
        transplanted_vhat = 0
        
        for var in variables:
            var_key = var.ref()
            
            if var_key not in self.slots:
                continue
            
            shadow_slots = self.slots[var_key]
            
            # Transplant m (first moment)
            if 'm' in shadow_slots:
                shadow_m = shadow_slots['m']
                keras_m = keras_optimizer.get_slot(var, 'm')
                
                if keras_m is None:
                    # Force slot creation
                    self._force_keras_slot_creation(keras_optimizer, var)
                    keras_m = keras_optimizer.get_slot(var, 'm')
                
                if keras_m is not None:
                    keras_m.assign(shadow_m)
                    transplanted_m += 1
            
            # Transplant v (second moment)
            if 'v' in shadow_slots:
                shadow_v = shadow_slots['v']
                keras_v = keras_optimizer.get_slot(var, 'v')
                
                if keras_v is None:
                    self._force_keras_slot_creation(keras_optimizer, var)
                    keras_v = keras_optimizer.get_slot(var, 'v')
                
                if keras_v is not None:
                    keras_v.assign(shadow_v)
                    transplanted_v += 1
            
            # Transplant vhat for AMSGrad
            if self.amsgrad and 'vhat' in shadow_slots:
                shadow_vhat = shadow_slots['vhat']
                keras_vhat = keras_optimizer.get_slot(var, 'vhat')
                
                if keras_vhat is not None:
                    keras_vhat.assign(shadow_vhat)
                    transplanted_vhat += 1
        
        print(f"   • Transplanted {transplanted_m} m buffers")
        print(f"   • Transplanted {transplanted_v} v buffers")
        if self.amsgrad:
            print(f"   • Transplanted {transplanted_vhat} vhat buffers")
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