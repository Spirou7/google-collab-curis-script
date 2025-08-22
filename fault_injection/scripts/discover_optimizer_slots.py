#!/usr/bin/env python3
"""
Discover the exact slot names used by each TensorFlow/Keras optimizer.

This script creates each optimizer and inspects its actual slot names
after performing a dummy training step.
"""

import tensorflow as tf
import numpy as np

def discover_optimizer_slots(optimizer_name: str):
    """Discover actual slot names for a given optimizer."""
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # Create optimizer
    optimizer_map = {
        'sgd': lambda: tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'sgd_vanilla': lambda: tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
        'adam': lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
        'adamw': lambda: tf.keras.optimizers.AdamW(learning_rate=0.01) if hasattr(tf.keras.optimizers, 'AdamW') else tf.keras.optimizers.Adam(learning_rate=0.01),
        'rmsprop': lambda: tf.keras.optimizers.RMSprop(learning_rate=0.01),
        'adagrad': lambda: tf.keras.optimizers.Adagrad(learning_rate=0.01),
        'adadelta': lambda: tf.keras.optimizers.Adadelta(learning_rate=0.01),
        'nadam': lambda: tf.keras.optimizers.Nadam(learning_rate=0.01),
        'ftrl': lambda: tf.keras.optimizers.Ftrl(learning_rate=0.01),
        'adamax': lambda: tf.keras.optimizers.Adamax(learning_rate=0.01),
    }
    
    if optimizer_name.lower() not in optimizer_map:
        print(f"Unknown optimizer: {optimizer_name}")
        return None
    
    optimizer = optimizer_map[optimizer_name.lower()]()
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mse')
    
    # Do a dummy training step to create slots
    dummy_x = np.random.randn(1, 5).astype(np.float32)
    dummy_y = np.random.randn(1, 1).astype(np.float32)
    
    with tf.GradientTape() as tape:
        pred = model(dummy_x, training=True)
        loss = tf.reduce_mean((pred - dummy_y) ** 2)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Now discover the actual slot names
    slot_names = set()
    slot_details = {}
    
    print(f"\n{'='*60}")
    print(f"Optimizer: {optimizer_name.upper()}")
    print(f"{'='*60}")
    
    # Method 1: Using get_slot_names() if available
    if hasattr(optimizer, 'get_slot_names'):
        try:
            names = optimizer.get_slot_names()
            print(f"Slot names from get_slot_names(): {names}")
            slot_names.update(names)
        except:
            pass
    
    # Method 2: Try common slot names and see what exists
    possible_slots = [
        'momentum', 'accumulator', 'rms', 'm', 'v', 
        'velocity', 'vhat', 'delta_accumulator',
        'linear', 'gradient_accumulator', 'slots'
    ]
    
    for var in model.trainable_variables:
        var_slots = []
        for slot_name in possible_slots:
            try:
                slot = optimizer.get_slot(var, slot_name)
                if slot is not None:
                    var_slots.append(slot_name)
                    slot_names.add(slot_name)
            except:
                pass
        
        if var_slots:
            slot_details[var.name] = var_slots
    
    # Method 3: Inspect optimizer._slots if available (internal API)
    if hasattr(optimizer, '_slots'):
        print("\nDirect inspection of _slots:")
        for var_key, slots_dict in optimizer._slots.items():
            for slot_name in slots_dict.keys():
                slot_names.add(slot_name)
                print(f"  Found slot: {slot_name}")
    
    # Method 4: Inspect optimizer variables
    print("\nOptimizer variables:")
    for var in optimizer.variables():
        print(f"  {var.name}: shape={var.shape}, dtype={var.dtype}")
        # Try to infer slot type from variable name
        if 'momentum' in var.name.lower():
            slot_names.add('momentum')
        elif '/m' in var.name or '/m:' in var.name:
            slot_names.add('m')
        elif '/v' in var.name or '/v:' in var.name:
            slot_names.add('v')
        elif 'accumulator' in var.name.lower():
            slot_names.add('accumulator')
        elif 'rms' in var.name.lower():
            slot_names.add('rms')
    
    # Method 5: Check for specific optimizer attributes
    if hasattr(optimizer, '_momentum'):
        print(f"Has _momentum attribute: {optimizer._momentum}")
    if hasattr(optimizer, '_rho'):
        print(f"Has _rho attribute: {optimizer._rho}")
    if hasattr(optimizer, '_beta_1'):
        print(f"Has _beta_1 attribute: {optimizer._beta_1}")
    if hasattr(optimizer, '_beta_2'):
        print(f"Has _beta_2 attribute: {optimizer._beta_2}")
    
    print(f"\nDiscovered slot names: {sorted(slot_names) if slot_names else 'None (stateless)'}")
    
    if slot_details:
        print("\nSlots per variable:")
        for var_name, slots in slot_details.items():
            print(f"  {var_name}: {slots}")
    
    return sorted(list(slot_names))


def get_slot_names_for_optimizer(optimizer_instance, model_variables):
    """
    Dynamically discover slot names for an already-instantiated optimizer.
    
    This is the function you can use in your actual code!
    """
    discovered_slots = {}
    
    for var in model_variables:
        var_slots = {}
        
        # Try all possible slot names
        possible_slots = [
            'momentum', 'accumulator', 'rms', 'm', 'v', 
            'velocity', 'vhat', 'delta_accumulator',
            'linear', 'gradient_accumulator'
        ]
        
        for slot_name in possible_slots:
            try:
                slot = optimizer_instance.get_slot(var, slot_name)
                if slot is not None:
                    var_slots[slot_name] = slot
            except:
                pass
        
        if var_slots:
            discovered_slots[var.name] = var_slots
    
    # Get unique slot names
    all_slot_names = set()
    for var_slots in discovered_slots.values():
        all_slot_names.update(var_slots.keys())
    
    return sorted(list(all_slot_names))


def main():
    print("Discovering TensorFlow Optimizer Slot Names")
    print("=" * 60)
    
    optimizers = [
        'sgd', 'sgd_vanilla', 'adam', 'adamw', 
        'rmsprop', 'adagrad', 'adadelta', 'nadam',
        'ftrl', 'adamax'
    ]
    
    results = {}
    
    for opt_name in optimizers:
        try:
            slots = discover_optimizer_slots(opt_name)
            results[opt_name] = slots
        except Exception as e:
            print(f"Error with {opt_name}: {e}")
            results[opt_name] = None
    
    print("\n" + "=" * 60)
    print("SUMMARY OF OPTIMIZER SLOT NAMES")
    print("=" * 60)
    
    for opt_name, slots in results.items():
        if slots is not None:
            print(f"{opt_name:12s}: {', '.join(slots) if slots else 'No slots (stateless)'}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED SLOT NAME MAPPING")
    print("=" * 60)
    print("""
    slot_name_map = {
        'sgd': ['momentum'],           # Only if momentum > 0
        'sgd_vanilla': [],              # No slots when momentum = 0
        'adam': ['m', 'v'],            # First and second moments
        'adamw': ['m', 'v'],           # Same as Adam
        'rmsprop': ['rms'],            # RMS accumulator (possibly 'momentum' too)
        'adagrad': ['accumulator'],    # Gradient accumulator
        'adadelta': ['accumulator'],   # Gradient accumulator
        'nadam': ['m', 'v'],          # Like Adam
        'adamax': ['m', 'v'],         # Like Adam
        'ftrl': ['accumulator', 'linear']  # FTRL specific slots
    }
    """)


if __name__ == "__main__":
    main()