import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fault_injection.scripts.random_injection import random_fault_injection

def run_specific_injection():
    """
    Runs a fault injection simulation with a specific, hardcoded configuration
    to reproduce a known scenario.
    """
    
    # Parameters derived from the user's request
    target_epoch = 0
    target_step = 15
    target_layer = "basicblock_2_basic_0_downsample"
    
    # Injection position and value from the user's log
    # The log shows a single-point injection. It needs to be wrapped in a list
    # because the injection function can handle multiple injection points.
    inj_pos = [[np.int64(570), np.int64(6), np.int64(10), np.int64(48)]]
    inj_values = [1.7976931348623157e+308] # This is sys.float_info.max
    
    print("="*70)
    print("🚀 STARTING SPECIFIC FAULT INJECTION SIMULATION")
    print(f"   - Target Layer: {target_layer}")
    print(f"   - Target Epoch: {target_epoch}")
    print(f"   - Target Step: {target_step}")
    print(f"   - Injection Position: {inj_pos}")
    print(f"   - Injection Value: {inj_values[0]}")
    print("="*70)
    
    results = random_fault_injection(
        model='resnet18',
        stage='fwrd_inject',
        fmodel='INPUT',
        target_layer=target_layer,
        target_epoch=target_epoch,
        target_step=target_step,
        learning_rate=0.001,
        inj_pos=inj_pos,
        inj_values=inj_values
    )
    
    print("\n" + "="*70)
    print("🏁 SPECIFIC SIMULATION COMPLETE")
    print(f"   - Final train accuracy: {results.get('final_train_accuracy', 'N/A'):.4f}")
    print(f"   - Final train loss: {results.get('final_train_loss', 'N/A'):.4f}")
    print(f"   - Early termination: {results.get('early_terminate', 'N/A')}")
    print("="*70)

def run_conv1_max_value_injection():
    """
    Runs a fault injection into basicblock_1_basic_0_conv1 at a random position
    with the maximum float value.
    """
    target_epoch = 0
    target_step = 2
    target_layer = "basicblock_1_basic_0_conv1"
    max_float_val = sys.float_info.max

    print("="*70)
    print("🚀 STARTING MAX VALUE INJECTION SIMULATION")
    print(f"   - Target Layer: {target_layer}")
    print(f"   - Target Epoch: {target_epoch}")
    print(f"   - Target Step: {target_step}")
    print(f"   - Injection Value: {max_float_val} (at random position)")
    print("="*70)

    results = random_fault_injection(
        model='resnet18',
        stage='fwrd_inject',
        fmodel='INPUT',
        target_layer=target_layer,
        target_epoch=target_epoch,
        target_step=target_step,
        learning_rate=0.001,
        min_val=max_float_val,
        max_val=max_float_val
    )

    print("\n" + "="*70)
    print("🏁 MAX VALUE SIMULATION COMPLETE")
    print(f"   - Final train accuracy: {results.get('final_train_accuracy', 'N/A'):.4f}")
    print(f"   - Final train loss: {results.get('final_train_loss', 'N/A'):.4f}")
    print(f"   - Early termination: {results.get('early_terminate', 'N/A')}")
    print("="*70)

# this was able to INCREASE the accuracy slightly as it was training, which was one of the outcomes !
def run_random_layer_n64_glb_injection(seed=None, max_global_steps=8):
    """
    Injects into a random layer of ResNet18 using the N64_INPUT_GLB fault model
    with hardcoded min/max values.

    Args:
        seed: Optional random seed to control stochasticity for this run
        max_global_steps: Hard limit on total training steps before early stop
    """
    target_epoch = 2
    target_step = 30
    fmodel = 'N16_RD'
    min_val = 3.6E+2
    max_val = 1.2E+8

    print("="*70)
    print("🚀 STARTING RANDOM LAYER N64_INPUT_GLB INJECTION SIMULATION")
    print(f"   - Target Layer: Random")
    print(f"   - Fault Model: {fmodel}")
    print(f"   - Target Epoch: {target_epoch}")
    print(f"   - Target Step: {target_step}")
    print(f"   - Min/Max Value: {min_val}/{max_val}")
    print(f"   - Seed: {seed}")
    print(f"   - Max Global Steps: {max_global_steps}")
    print("="*70)

    results = random_fault_injection(
        model='resnet18',
        stage='bkwd_inject',
        fmodel=fmodel,
        target_layer="basicblock_4_basic_0_downsample_grad_in",  # Omit to select a random layer
        target_epoch=target_epoch,
        target_step=target_step,
        learning_rate=0.001,
        min_val=min_val,
        max_val=max_val,
        seed=seed,
        max_global_steps=max_global_steps
    )

    print("\n" + "="*70)
    print("🏁 N64_INPUT_GLB SIMULATION COMPLETE")
    # Display the randomly chosen layer in the results
    if results and 'injection_params' in results:
        randomly_selected_layer = results['injection_params'].get('target_layer', 'N/A')
        print(f"   - Injected Layer: {randomly_selected_layer}")
        
    print(f"   - Final train accuracy: {results.get('final_train_accuracy', 'N/A'):.4f}")
    print(f"   - Final train loss: {results.get('final_train_loss', 'N/A'):.4f}")
    print(f"   - Early termination: {results.get('early_terminate', 'N/A')}")
    print("="*70)


if __name__ == "__main__":
    # Run the random-layer experiment 20 times with different seeds
    base_seed = 12345
    num_runs = 20
    for i in range(num_runs):
        run_id = i + 1
        seed = base_seed + i
        print("\n" + "#"*80)
        print(f"RUN {run_id}/{num_runs} - seed={seed}")
        print("#"*80 + "\n")
        run_random_layer_n64_glb_injection(seed=seed, max_global_steps=500)