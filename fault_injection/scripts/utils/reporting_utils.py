import json
import csv
import os
import datetime
import numpy as np
from typing import Dict, List


def save_experiment_results(results: Dict, experiment_dir: str):
    """Save all experiment results with detailed logging."""
    print(f"   • Saving to: {experiment_dir}")
    
    results_path = os.path.join(experiment_dir, 'results.json')
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Deep convert all numpy types
    results_json = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"   • Main results saved to: results.json")
    
    # Also save recovery data as CSV for each optimizer
    for optimizer_name, recovery_data in results['recovery_results'].items():
        csv_path = os.path.join(experiment_dir, f'recovery_{optimizer_name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'accuracy', 'loss'])
            for i in range(len(recovery_data['steps'])):
                writer.writerow([
                    recovery_data['steps'][i],
                    recovery_data['accuracy'][i],
                    recovery_data['loss'][i]
                ])
        print(f"   • {optimizer_name} recovery data saved to: recovery_{optimizer_name}.csv")


def save_intermediate_summary(results: List[Dict], results_dir: str, 
                             baseline_optimizer: str, test_optimizers: List[str]):
    """Save intermediate summary of results."""
    print(f"   Calculating aggregate metrics...")
    
    summary = {
        'completed_experiments': len(results),
        'baseline_optimizer': baseline_optimizer,
        'test_optimizers': test_optimizers,
        'aggregate_metrics': {}
    }
    
    # Calculate aggregate metrics for each optimizer
    for optimizer in [baseline_optimizer] + test_optimizers:
        final_accs = []
        acc_changes = []
        degradation_rates = []
        
        for result in results:
            if result and 'recovery_results' in result and optimizer in result['recovery_results']:
                recovery = result['recovery_results'][optimizer]
                final_accs.append(recovery['final_accuracy'])
                acc_changes.append(recovery['accuracy_change'])
                degradation_rates.append(recovery.get('degradation_rate', 0))
        
        if final_accs:
            summary['aggregate_metrics'][optimizer] = {
                'mean_final_accuracy': float(np.mean(final_accs)),
                'std_final_accuracy': float(np.std(final_accs)),
                'mean_accuracy_change': float(np.mean(acc_changes)),
                'std_accuracy_change': float(np.std(acc_changes)),
                'mean_degradation_rate': float(np.mean(degradation_rates)),
                'positive_recovery_rate': float(sum(1 for x in acc_changes if x > 0) / len(acc_changes))
            }
    
    summary_path = os.path.join(results_dir, 'intermediate_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Summary saved to: intermediate_summary.json")


def generate_final_report(results: List[Dict], results_dir: str, config):
    """Generate comprehensive final report in Markdown format."""
    if not results:
        print("   ⚠️ No results to report")
        return
    
    print(f"   Generating markdown report...")
    
    report_path = os.path.join(results_dir, 'final_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Optimizer Mitigation Experiment - Final Report\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Experiments Completed**: {len(results)}/{config.num_experiments}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Baseline Optimizer**: {config.baseline_optimizer}\n")
        f.write(f"- **Test Optimizers**: {', '.join(config.test_optimizers)}\n")
        f.write(f"- **Fault Model**: {config.fmodel}\n")
        f.write(f"- **Injection Value Range**: [{config.min_val:.2e}, {config.max_val:.2e}]\n")
        f.write(f"- **Max Target Epoch**: {config.max_target_epoch}\n")
        f.write(f"- **Max Target Step**: {config.max_target_step}\n")
        f.write(f"- **Steps After Injection**: {config.max_steps_after_injection}\n\n")
        
        # Aggregate statistics
        f.write("## Aggregate Results\n\n")
        
        # Table header
        f.write("| Optimizer | Mean Final Acc | Std Final Acc | Mean Acc Change | Positive Recovery % | Mean Degrad. Rate |\n")
        f.write("|-----------|---------------|---------------|-----------------|-------------------|------------------|\n")
        
        optimizer_stats = {}
        
        for optimizer in [config.baseline_optimizer] + config.test_optimizers:
            final_accs = []
            acc_changes = []
            degradation_rates = []
            
            for result in results:
                if result and 'recovery_results' in result and optimizer in result['recovery_results']:
                    recovery = result['recovery_results'][optimizer]
                    final_accs.append(recovery['final_accuracy'])
                    acc_changes.append(recovery['accuracy_change'])
                    degradation_rates.append(recovery.get('degradation_rate', 0))
            
            if final_accs:
                mean_final = np.mean(final_accs)
                std_final = np.std(final_accs)
                mean_change = np.mean(acc_changes)
                positive_rate = sum(1 for x in acc_changes if x > 0) / len(acc_changes) * 100
                mean_degrad = np.mean(degradation_rates) * 1000
                
                optimizer_stats[optimizer] = {
                    'mean_change': mean_change,
                    'positive_rate': positive_rate
                }
                
                f.write(f"| {optimizer:11} | {mean_final:13.4f} | {std_final:13.4f} | "
                       f"{mean_change:15.4f} | {positive_rate:17.1f} | {mean_degrad:16.4f} |\n")
                
                print(f"   • {optimizer}: mean change={mean_change:.4f}, positive recovery={positive_rate:.1f}%")
        
        # Winner analysis
        f.write("\n## Analysis\n\n")
        
        # Find best performer
        if optimizer_stats:
            best_optimizer = max(optimizer_stats, key=lambda x: optimizer_stats[x]['mean_change'])
            best_change = optimizer_stats[best_optimizer]['mean_change']
            baseline_change = optimizer_stats.get(config.baseline_optimizer, {}).get('mean_change', 0)
            
            f.write(f"### Best Performing Optimizer: **{best_optimizer}**\n\n")
            
            if best_change > baseline_change and best_optimizer != config.baseline_optimizer:
                improvement = best_change - baseline_change
                f.write(f"✅ **{best_optimizer}** outperforms baseline by {improvement:.4f} points\n\n")
                print(f"\n   🏆 WINNER: {best_optimizer} outperforms baseline by {improvement:.4f}")
            else:
                f.write(f"⚠️ Baseline optimizer **{config.baseline_optimizer}** performs best\n\n")
                print(f"\n   📊 Baseline optimizer {config.baseline_optimizer} performs best")
        
        # Conclusions
        f.write("\n## Conclusions\n\n")
        
        # Check hypothesis
        hypothesis_supported = False
        for optimizer in config.test_optimizers:
            if optimizer in optimizer_stats:
                if optimizer_stats[optimizer]['positive_rate'] > 50:
                    hypothesis_supported = True
                    f.write(f"✅ **Hypothesis SUPPORTED**: {optimizer} shows better recovery "
                           f"than baseline in {optimizer_stats[optimizer]['positive_rate']:.1f}% of cases\n\n")
                    print(f"   ✅ {optimizer} supports hypothesis ({optimizer_stats[optimizer]['positive_rate']:.1f}% success)")
        
        if not hypothesis_supported:
            f.write("❌ **Hypothesis NOT SUPPORTED**: No alternative optimizer consistently "
                   "outperforms the baseline in recovering from slowdegrade effects\n\n")
            print(f"   ❌ Hypothesis not supported by data")
        
        # Add detailed experiment results section
        f.write("\n## Detailed Experiment Results\n\n")
        f.write("### Individual Experiment Performance\n\n")
        
        for i, result in enumerate(results):
            if result and 'recovery_results' in result:
                f.write(f"#### Experiment {i+1}\n")
                f.write(f"- **Injection**: Epoch {result['injection_config']['target_epoch']}, "
                       f"Step {result['injection_config']['target_step']}\n")
                f.write(f"- **Layer**: {result['injection_config']['target_layer']}\n")
                f.write(f"- **Corruption**: {result['corruption_info']['corruption_percentage']:.2f}%\n")
                
                # Find best performer for this experiment
                best_opt = max(result['recovery_results'].items(), 
                             key=lambda x: x[1]['accuracy_change'])
                f.write(f"- **Best Optimizer**: {best_opt[0]} "
                       f"(change: {best_opt[1]['accuracy_change']:+.4f})\n\n")
    
    print(f"   ✓ Report saved to: {report_path}")


def save_error_info(experiment_id: int, error: Exception, results_dir: str):
    """Save error information for failed experiments."""
    import traceback
    
    error_info = {
        'experiment_id': experiment_id,
        'error': str(error),
        'traceback': traceback.format_exc()
    }
    
    error_path = os.path.join(results_dir, f'error_exp_{experiment_id:03d}.json')
    with open(error_path, 'w') as f:
        json.dump(error_info, f, indent=2)
    
    print(f"   Error details saved to: {error_path}")