"""
Base Class for Shadow Optimizers

Shadow optimizers track optimizer state (slots) without updating model weights.
They accumulate the same history a real optimizer would, but never modify parameters.
"""

import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


class ShadowOptimizerBase(tf.Module, ABC):
    """
    Abstract base class for shadow optimizers.
    
    Shadow optimizers maintain all the state (momentum, variance, etc.) that
    a real optimizer would maintain, but they never update the actual model weights.
    This allows us to track what state each optimizer would have built up during
    training, then transplant that state into a real optimizer later.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 name: str = "ShadowOptimizer",
                 **kwargs):
        """
        Initialize the shadow optimizer.
        
        Args:
            learning_rate: Learning rate (for reference, not used in updates)
            name: Name of this shadow optimizer
            **kwargs: Additional optimizer-specific parameters
        """
        super().__init__(name=name)
        
        self.learning_rate = learning_rate
        self.iterations = None
        self.slots = {}  # Maps var.ref() to dict of slot variables
        self._built = False
        self._config = {
            'learning_rate': learning_rate,
            'name': name,
            **kwargs
        }
    
    @abstractmethod
    def _create_slots(self, var: tf.Variable) -> Dict[str, tf.Variable]:
        """
        Create slot variables for a given model variable.
        
        This method should be implemented by each specific optimizer to create
        the appropriate slots (e.g., momentum for SGD, m and v for Adam).
        
        Args:
            var: Model variable to create slots for
            
        Returns:
            Dictionary mapping slot names to tf.Variables
        """
        pass
    
    @abstractmethod
    def _update_slots(self, 
                     grad: tf.Tensor, 
                     var: tf.Variable,
                     slots: Dict[str, tf.Variable]) -> None:
        """
        Update slot variables based on gradients.
        
        This method implements the optimizer's update rule for its slots,
        but does NOT update the model variable itself.
        
        Args:
            grad: Gradient for this variable
            var: Model variable (for shape/dtype reference only)
            slots: Dictionary of slot variables for this variable
        """
        pass
    
    def build(self, variables: List[tf.Variable]) -> None:
        """
        Initialize slots for all model variables.
        
        Args:
            variables: List of model variables to track
        """
        if self._built:
            print(f"⚠️ {self.name} already built, skipping rebuild")
            return
        
        print(f"\n🔨 Building {self.name} shadow optimizer")
        print(f"   • Variables to track: {len(variables)}")
        
        # Create iteration counter
        with tf.name_scope(self.name):
            self.iterations = tf.Variable(
                0, dtype=tf.int64, name='iterations', trainable=False
            )
        
        # Create slots for each variable
        total_params = 0
        for var in variables:
            var_key = var.ref()
            
            # Create slots for this variable
            with tf.name_scope(f"{self.name}/{var.name}"):
                var_slots = self._create_slots(var)
            
            self.slots[var_key] = var_slots
            
            # Count parameters
            var_params = tf.reduce_prod(var.shape).numpy()
            total_params += var_params
        
        self._built = True
        
        # Log statistics
        print(f"   • Total parameters tracked: {total_params:,}")
        print(f"   • Slots created per variable: {list(next(iter(self.slots.values())).keys())}")
        print(f"   ✓ {self.name} built successfully")
    
    def update_from_grads(self, 
                         grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]]) -> None:
        """
        Update shadow optimizer state from gradients.
        
        This is the main method that processes gradients and updates the optimizer's
        internal state (slots) without touching the model weights.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs
        """
        if not self._built:
            raise RuntimeError(f"{self.name} not built. Call build() first.")
        
        # Increment iteration counter
        self.iterations.assign_add(1)
        
        # Update slots for each variable
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            
            var_key = var.ref()
            if var_key not in self.slots:
                print(f"⚠️ Variable {var.name} not in slots, skipping")
                continue
            
            # Get slots for this variable
            var_slots = self.slots[var_key]
            
            # Update the slots (but NOT the variable itself!)
            self._update_slots(grad, var, var_slots)
    
    def update_from_grads_dict(self,
                              grads: Dict[str, tf.Tensor],
                              vars_dict: Dict[str, tf.Variable]) -> None:
        """
        Alternative update method using dictionaries.
        
        Args:
            grads: Dictionary mapping variable names to gradients
            vars_dict: Dictionary mapping variable names to variables
        """
        grads_and_vars = []
        for name, grad in grads.items():
            if name in vars_dict:
                grads_and_vars.append((grad, vars_dict[name]))
        
        self.update_from_grads(grads_and_vars)
    
    def get_slot(self, var: tf.Variable, slot_name: str) -> Optional[tf.Variable]:
        """
        Get a specific slot for a variable.
        
        Args:
            var: Model variable
            slot_name: Name of the slot (e.g., 'momentum', 'm', 'v')
            
        Returns:
            Slot variable or None if not found
        """
        var_key = var.ref()
        if var_key in self.slots:
            return self.slots[var_key].get(slot_name)
        return None
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get complete state dictionary for checkpointing.
        
        Returns:
            Dictionary containing all optimizer state
        """
        state = {
            'config': self._config,
            'iterations': self.iterations.numpy() if self.iterations else 0,
            'slots': {}
        }
        
        # Convert slot variables to numpy arrays for serialization
        for var_ref, var_slots in self.slots.items():
            # Use string representation of var_ref for JSON serialization
            var_key = str(var_ref)
            state['slots'][var_key] = {}
            
            for slot_name, slot_var in var_slots.items():
                state['slots'][var_key][slot_name] = slot_var.numpy()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load state from dictionary.
        
        Args:
            state: State dictionary from state_dict()
        """
        if not self._built:
            raise RuntimeError(f"{self.name} must be built before loading state")
        
        # Restore iteration count
        if self.iterations and 'iterations' in state:
            self.iterations.assign(state['iterations'])
        
        # Restore slots
        if 'slots' in state:
            for var_key_str, var_slots_dict in state['slots'].items():
                # Find the matching var_ref
                for var_ref in self.slots.keys():
                    if str(var_ref) == var_key_str:
                        for slot_name, slot_value in var_slots_dict.items():
                            if slot_name in self.slots[var_ref]:
                                self.slots[var_ref][slot_name].assign(slot_value)
                        break
        
        print(f"✓ Loaded state into {self.name}")
    
    def get_slot_names(self) -> List[str]:
        """
        Get list of slot names this optimizer uses.
        
        Returns:
            List of slot names (e.g., ['momentum'] for SGD, ['m', 'v'] for Adam)
        """
        if not self.slots:
            return []
        
        # Get slot names from first variable
        first_var_slots = next(iter(self.slots.values()))
        return list(first_var_slots.keys())
    
    def compute_slot_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics about slot variables.
        
        Returns:
            Dictionary with statistics for each slot type
        """
        stats = {}
        
        for slot_name in self.get_slot_names():
            slot_values = []
            
            # Collect all values for this slot type
            for var_slots in self.slots.values():
                if slot_name in var_slots:
                    slot_var = var_slots[slot_name]
                    slot_values.append(slot_var.numpy().flatten())
            
            if slot_values:
                all_values = np.concatenate(slot_values)
                stats[slot_name] = {
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'norm': float(np.linalg.norm(all_values)),
                }
        
        return stats
    
    def reset(self) -> None:
        """
        Reset all slot variables to their initial values.
        """
        if self.iterations:
            self.iterations.assign(0)
        
        for var_slots in self.slots.values():
            for slot_var in var_slots.values():
                # Reset to zeros (or appropriate initial value)
                slot_var.assign(tf.zeros_like(slot_var))
        
        print(f"✓ Reset {self.name} to initial state")
    
    def __repr__(self) -> str:
        built_str = "built" if self._built else "not built"
        num_vars = len(self.slots) if self._built else 0
        return f"{self.__class__.__name__}(name='{self.name}', {built_str}, tracking {num_vars} variables)"