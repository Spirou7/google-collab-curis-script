#!/usr/bin/env python3
"""
Example Integration: Adding Weight NaN Monitoring to Existing Fault Injection

This shows how to integrate weight corruption monitoring into your existing
fault injection experiments without major code changes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from ..models.weight_analyzer import analyze_weight_corruption, check_weights_for_corruption
from ..models.inject_utils import record


def enhanced_training_step_with_weight_monitoring(model, train_recorder, epoch, step):
    """
    Enhanced version of your training step that includes weight monitoring.
    
    This shows how to add weight corruption detection to your existing
    training loops with minimal changes.
    """
    
    # Your existing loss/accuracy tracking code would go here
    # ... training step logic ...
    
    # NEW: Check for weight corruption every 10 steps
    if step % 10 == 0:
        is_corrupted, weight_stats = check_weights_for_corruption(
            model, threshold_percentage=0.1
        )
        
        # Log weight statistics using your existing record function
        record(train_recorder, 
               f"Epoch {epoch}, Step {step} - Weight corruption: "
               f"{weight_stats.corrupted_percentage:.4f}% "
               f"({weight_stats.nan_parameters} NaN, {weight_stats.inf_parameters} Inf)\n")
        
        # Check if corruption exceeds threshold
        if is_corrupted:
            record(train_recorder, "WARNING: Weight corruption above threshold!\n")
            
            # Optional: Get detailed layer breakdown for research
            detailed_stats = analyze_weight_corruption(model, include_layer_details=True)
            record(train_recorder, "Layer-wise corruption breakdown:\n")
            
            for layer_name, layer_data in detailed_stats.layer_stats.items():
                if layer_data['nan'] > 0 or layer_data['inf'] > 0:
                    layer_corruption_pct = ((layer_data['nan'] + layer_data['inf']) / layer_data['total']) * 100
                    record(train_recorder, 
                           f"  {layer_name}: {layer_corruption_pct:.4f}% corrupted "
                           f"({layer_data['nan']} NaN, {layer_data['inf']} Inf)\n")
            
            # Return True to indicate training should terminate
            return True
    
    return False  # Continue training


def enhanced_db_stats_with_weight_tracking():
    """
    Example of how to extend your DBStats class to track weight corruption.
    
    Add these fields to your existing DBStats class:
    """
    
    # NEW FIELDS to add to your DBStats.__init__():
    additional_fields = """
    # Weight corruption tracking
    self.first_weight_corruption_epoch = -1
    self.first_weight_corruption_step = -1
    self.max_weight_corruption_percentage = 0.0
    self.weight_corruption_history = []
    """
    
    # NEW METHOD to add to your DBStats class:
    def update_weight_corruption(self, epoch, step, corruption_percentage, nan_count, inf_count):
        """Add this method to your DBStats class"""
        
        # Track first occurrence of weight corruption
        if corruption_percentage > 0 and self.first_weight_corruption_epoch == -1:
            self.first_weight_corruption_epoch = epoch
            self.first_weight_corruption_step = step
        
        # Track maximum corruption
        if corruption_percentage > self.max_weight_corruption_percentage:
            self.max_weight_corruption_percentage = corruption_percentage
        
        # Record corruption history
        self.weight_corruption_history.append({
            'epoch': epoch,
            'step': step,
            'corruption_percentage': corruption_percentage,
            'nan_count': nan_count,
            'inf_count': inf_count
        })
    
    print("Add these fields and methods to your existing DBStats class:")
    print(additional_fields)
    print("\nAnd add the update_weight_corruption method shown above.")


def example_fault_injection_with_weight_monitoring():
    """
    Example showing how to add weight monitoring to your fault injection experiments.
    
    This mimics the pattern in your reproduce_injections.py and random_injection.py files.
    """
    
    print("Example: Enhanced Fault Injection with Weight Monitoring")
    print("=" * 60)
    
    # This would be your actual model loading code
    print("1. Load your model (ResNet18, DenseNet, etc.)")
    print("   model = load_your_model()")
    
    print("\n2. Set up weight monitoring")
    print("   from models.weight_analyzer import create_weight_monitoring_hook")
    print("   weight_monitor = create_weight_monitoring_hook(")
    print("       model, check_frequency=10, corruption_threshold=0.5,")
    print("       train_recorder=train_recorder)")
    
    print("\n3. Enhanced training loop:")
    print("""
    for epoch in range(config.EPOCHS):
        for step in range(steps_per_epoch):
            
            # Your existing training step
            train_step(train_iterator)
            
            # Your existing fault injection logic
            if epoch == inject_epoch and step == inject_step:
                # Perform injection using your existing injection code
                inj_args, inj_flag = get_inj_args(...)
                # ... injection logic ...
                
                # NEW: Immediately check weight corruption after injection
                is_corrupted, stats = check_weights_for_corruption(model, 0.01)
                record(train_recorder, 
                       f"Post-injection weight corruption: {stats.corrupted_percentage:.4f}%\\n")
            
            # NEW: Regular weight monitoring
            should_terminate = weight_monitor(step, epoch)
            
            # Your existing NaN detection for loss
            if not np.isfinite(train_loss.result()):
                record(train_recorder, "Encounter NaN in loss! Terminate training!\\n")
                break
                
            # NEW: Check if weight corruption requires termination
            if should_terminate:
                record(train_recorder, "Training terminated due to weight corruption!\\n")
                break
    """)


def research_questions_you_can_answer():
    """
    Research questions that weight corruption monitoring can help answer.
    """
    
    print("\n" + "=" * 60)
    print("RESEARCH QUESTIONS YOU CAN NOW ANSWER")
    print("=" * 60)
    
    questions = [
        "1. What percentage of weights become NaN after fault injection?",
        "2. Which layers are most susceptible to corruption from your injection strategies?",
        "3. How does weight corruption correlate with loss explosion?",
        "4. Do different injection types (INPUT, WEIGHT, OUTPUT) cause different corruption patterns?",
        "5. How many training steps does it take for corruption to spread after injection?",
        "6. What's the minimum corruption percentage that causes training failure?",
        "7. Which model architectures (ResNet vs DenseNet vs EfficientNet) are more resilient?",
        "8. How does the injection position affect corruption spread patterns?",
        "9. Can you detect impending training failure before loss becomes NaN?",
        "10. What's the relationship between gradient explosion and weight corruption?"
    ]
    
    for question in questions:
        print(f"  {question}")
    
    print(f"\nExample analysis you can now perform:")
    print(f"  'After injecting NaN into layer X at step Y, {2.3:.1f}% of weights")
    print(f"   became corrupted within 5 training steps, with corruption")
    print(f"   concentrated in layers Z and W.'")


def minimal_integration_example():
    """
    Minimal code changes needed to add weight monitoring.
    """
    
    print("\n" + "=" * 60)
    print("MINIMAL INTEGRATION (Just 3 lines of code!)")
    print("=" * 60)
    
    print("Add to your existing training loop:")
    print("""
# Add this import at the top
from ..models.weight_analyzer import check_weights_for_corruption

# Add these 3 lines in your training loop:
is_corrupted, stats = check_weights_for_corruption(model, threshold_percentage=0.1)
record(train_recorder, f"Weight corruption: {stats.corrupted_percentage:.3f}%\\n")
if is_corrupted: 
    record(train_recorder, "WARNING: High weight corruption detected!\\n")
""")


if __name__ == "__main__":
    print("Integration Example for Fault Injection Research")
    print("This shows how to add weight NaN monitoring to your existing experiments")
    
    enhanced_db_stats_with_weight_tracking()
    example_fault_injection_with_weight_monitoring()
    research_questions_you_can_answer()
    minimal_integration_example()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Try the minimal integration first (just 3 lines)")
    print("2. Run your existing fault injection experiments")
    print("3. Observe weight corruption patterns in your logs")
    print("4. Add more detailed monitoring as needed")
    print("5. Use the insights to refine your injection strategies") 