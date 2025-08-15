import numpy as np
from typing import Dict, List, Tuple


def calculate_recovery_metrics(recovery_history: Dict) -> Dict:
    """
    Calculate recovery metrics from training history.
    
    Args:
        recovery_history: Dictionary containing training history
    
    Returns:
        Dictionary with calculated metrics
    """
    print(f"\n📊 CALCULATING RECOVERY METRICS...")
    
    # Basic metrics
    recovery_history['final_accuracy'] = recovery_history['accuracy'][-1] if recovery_history['accuracy'] else 0
    recovery_history['final_loss'] = recovery_history['loss'][-1] if recovery_history['loss'] else float('inf')
    recovery_history['accuracy_change'] = recovery_history['final_accuracy'] - recovery_history['starting_accuracy']
    
    print(f"   • Final accuracy: {recovery_history['final_accuracy']:.4f}")
    print(f"   • Final loss: {recovery_history['final_loss']:.4f}")
    print(f"   • Total accuracy change: {recovery_history['accuracy_change']:+.4f}")
    
    # Calculate degradation rate
    if len(recovery_history['accuracy']) > 10:
        print(f"\n📈 Calculating degradation/recovery rate...")
        # Linear fit to accuracy over last 100 steps
        recent_steps = recovery_history['steps'][-100:] if len(recovery_history['steps']) > 100 else recovery_history['steps']
        recent_acc = recovery_history['accuracy'][-100:] if len(recovery_history['accuracy']) > 100 else recovery_history['accuracy']
        
        if len(recent_steps) > 1:
            z = np.polyfit(recent_steps, recent_acc, 1)
            recovery_history['degradation_rate'] = float(z[0])  # Slope of accuracy change
            print(f"   • Degradation rate: {z[0]*1000:.4f} (×1000 acc/step)")
            
            if z[0] > 0:
                print(f"   ✅ RECOVERING: Accuracy improving over time")
            elif z[0] < -0.0001:
                print(f"   ⚠️ DEGRADING: Accuracy declining over time")
            else:
                print(f"   📊 STABLE: Accuracy relatively stable")
        else:
            recovery_history['degradation_rate'] = 0
            print(f"   • Insufficient data for rate calculation")
    else:
        recovery_history['degradation_rate'] = 0
        print(f"   • Insufficient data for rate calculation")
    
    return recovery_history


def analyze_recovery_performance(recovery_results: Dict[str, Dict]) -> Dict:
    """
    Analyze and compare recovery performance across optimizers.
    
    Args:
        recovery_results: Dictionary mapping optimizer names to recovery histories
    
    Returns:
        Dictionary with performance analysis
    """
    analysis = {}
    
    print(f"\n📊 Recovery Performance Analysis:")
    print(f"   {'Optimizer':<15} {'Initial Acc':<12} {'Final Acc':<12} {'Change':<12} {'Status'}")
    print(f"   {'-'*70}")
    
    for opt_name, recovery in recovery_results.items():
        initial = recovery['starting_accuracy']
        final = recovery['final_accuracy']
        change = recovery['accuracy_change']
        
        if change > 0.01:
            status = "✅ Recovered"
        elif change < -0.01:
            status = "❌ Degraded"
        else:
            status = "📊 Stable"
        
        print(f"   {opt_name:<15} {initial:<12.4f} {final:<12.4f} {change:<+12.4f} {status}")
        
        analysis[opt_name] = {
            'status': status,
            'change': change,
            'final_accuracy': final,
            'degradation_rate': recovery.get('degradation_rate', 0)
        }
    
    # Determine best performer
    best_optimizer = max(recovery_results.items(), 
                        key=lambda x: x[1]['accuracy_change'])
    print(f"\n🏆 Best performer: {best_optimizer[0]} (change: {best_optimizer[1]['accuracy_change']:+.4f})")
    
    analysis['best_optimizer'] = best_optimizer[0]
    analysis['best_change'] = best_optimizer[1]['accuracy_change']
    
    return analysis


def calculate_aggregate_metrics(all_results: List[Dict]) -> Dict:
    """
    Calculate aggregate metrics across all experiments.
    
    Args:
        all_results: List of experiment results
    
    Returns:
        Dictionary with aggregate statistics
    """
    aggregate = {}
    
    # Collect metrics for each optimizer
    optimizer_metrics = {}
    
    for result in all_results:
        if 'recovery_results' not in result:
            continue
            
        for optimizer, recovery in result['recovery_results'].items():
            if optimizer not in optimizer_metrics:
                optimizer_metrics[optimizer] = {
                    'final_accs': [],
                    'acc_changes': [],
                    'degradation_rates': []
                }
            
            optimizer_metrics[optimizer]['final_accs'].append(recovery['final_accuracy'])
            optimizer_metrics[optimizer]['acc_changes'].append(recovery['accuracy_change'])
            optimizer_metrics[optimizer]['degradation_rates'].append(recovery.get('degradation_rate', 0))
    
    # Calculate statistics for each optimizer
    for optimizer, metrics in optimizer_metrics.items():
        if metrics['final_accs']:
            aggregate[optimizer] = {
                'mean_final_accuracy': float(np.mean(metrics['final_accs'])),
                'std_final_accuracy': float(np.std(metrics['final_accs'])),
                'mean_accuracy_change': float(np.mean(metrics['acc_changes'])),
                'std_accuracy_change': float(np.std(metrics['acc_changes'])),
                'mean_degradation_rate': float(np.mean(metrics['degradation_rates'])),
                'positive_recovery_rate': float(sum(1 for x in metrics['acc_changes'] if x > 0) / len(metrics['acc_changes'])),
                'num_experiments': len(metrics['final_accs'])
            }
    
    return aggregate


def check_for_divergence(loss: float, step: int) -> Tuple[bool, str]:
    """
    Check if training has diverged.
    
    Args:
        loss: Current loss value
        step: Current training step
    
    Returns:
        Tuple of (diverged, message)
    """
    import tensorflow as tf
    
    if not tf.math.is_finite(loss):
        message = f"\n   ⚠️ DIVERGENCE DETECTED at step {step}! Loss: {loss}"
        return True, message
    return False, ""