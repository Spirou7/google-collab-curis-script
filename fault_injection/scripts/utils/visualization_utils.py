import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List


def create_recovery_comparison_plot(results: Dict, experiment_dir: str):
    """Create comparison plots for a single experiment."""
    print(f"   • Creating comparison plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different optimizers
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['recovery_results'])))
    
    # Plot 1: Accuracy over time
    injection_step = results['corruption_info']['injection_step']
    
    # Plot pre-injection phase
    pre_history = results['pre_injection_history']
    if pre_history['steps']:
        ax1.plot(pre_history['steps'], pre_history['accuracy'], 
                'k-', label='Pre-injection', linewidth=1, alpha=0.7)
    
    # Plot recovery for each optimizer
    for i, (opt_name, recovery) in enumerate(results['recovery_results'].items()):
        ax1.plot(recovery['steps'], recovery['accuracy'], 
                color=colors[i], label=f'{opt_name} (final: {recovery["final_accuracy"]:.3f})',
                linewidth=2)
    
    ax1.axvline(x=injection_step, color='red', linestyle='--', 
               label='Injection', alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Accuracy Recovery After Fault Injection')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparative recovery metrics
    optimizer_names = list(results['recovery_results'].keys())
    final_accuracies = [r['final_accuracy'] for r in results['recovery_results'].values()]
    accuracy_changes = [r['accuracy_change'] for r in results['recovery_results'].values()]
    
    x = np.arange(len(optimizer_names))
    width = 0.35
    
    ax2.bar(x - width/2, final_accuracies, width, label='Final Accuracy', alpha=0.8)
    ax2.bar(x + width/2, accuracy_changes, width, label='Accuracy Change', alpha=0.8)
    
    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Recovery Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(optimizer_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add zero line for reference
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(experiment_dir, 'recovery_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   • Recovery comparison saved to: recovery_comparison.png")


def create_degradation_rate_plot(results: Dict, experiment_dir: str):
    """Create plot focusing on degradation rates."""
    print(f"   • Creating degradation analysis plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate and plot degradation slopes
    optimizer_names = []
    degradation_rates = []
    colors = []
    
    for opt_name, recovery in results['recovery_results'].items():
        optimizer_names.append(opt_name)
        degradation_rates.append(recovery.get('degradation_rate', 0) * 1000)  # Scale for visibility
        
        # Color based on performance
        if recovery.get('degradation_rate', 0) > 0:
            colors.append('green')  # Improving
        elif recovery.get('degradation_rate', 0) < -0.0001:
            colors.append('red')    # Degrading significantly
        else:
            colors.append('yellow') # Stable
    
    bars = ax.bar(optimizer_names, degradation_rates, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, rate in zip(bars, degradation_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.3f}',
               ha='center', va='bottom' if height > 0 else 'top')
    
    ax.set_ylabel('Degradation Rate (×1000 accuracy/step)')
    ax.set_title('Accuracy Degradation Rate by Optimizer\n(Positive = Improving, Negative = Degrading)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = os.path.join(experiment_dir, 'degradation_rates.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   • Degradation rates saved to: degradation_rates.png")


def create_experiment_visualizations(results: Dict, experiment_dir: str):
    """Create all visualizations for a single experiment."""
    create_recovery_comparison_plot(results, experiment_dir)
    create_degradation_rate_plot(results, experiment_dir)


def create_summary_visualization(all_results: List[Dict], results_dir: str):
    """Create comprehensive summary visualization across all experiments."""
    if not all_results:
        return
    
    print(f"   • Creating summary visualization...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Collect data across all experiments
    optimizer_data = {}
    
    for result in all_results:
        if 'recovery_results' not in result:
            continue
        
        for opt_name, recovery in result['recovery_results'].items():
            if opt_name not in optimizer_data:
                optimizer_data[opt_name] = {
                    'final_accs': [],
                    'acc_changes': [],
                    'degradation_rates': []
                }
            
            optimizer_data[opt_name]['final_accs'].append(recovery['final_accuracy'])
            optimizer_data[opt_name]['acc_changes'].append(recovery['accuracy_change'])
            optimizer_data[opt_name]['degradation_rates'].append(recovery.get('degradation_rate', 0))
    
    optimizer_names = list(optimizer_data.keys())
    
    # Plot 1: Final accuracy distribution
    ax = axes[0]
    data = [optimizer_data[opt]['final_accs'] for opt in optimizer_names]
    bp = ax.boxplot(data, labels=optimizer_names, patch_artist=True)
    ax.set_ylabel('Final Accuracy')
    ax.set_title('Final Accuracy Distribution')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Accuracy change distribution
    ax = axes[1]
    data = [optimizer_data[opt]['acc_changes'] for opt in optimizer_names]
    bp = ax.boxplot(data, labels=optimizer_names, patch_artist=True)
    ax.set_ylabel('Accuracy Change')
    ax.set_title('Accuracy Change Distribution')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Mean performance comparison
    ax = axes[2]
    means = [np.mean(optimizer_data[opt]['acc_changes']) for opt in optimizer_names]
    stds = [np.std(optimizer_data[opt]['acc_changes']) for opt in optimizer_names]
    x = np.arange(len(optimizer_names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Accuracy Change')
    ax.set_title('Mean Recovery Performance')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Recovery success rate
    ax = axes[3]
    success_rates = [sum(1 for x in optimizer_data[opt]['acc_changes'] if x > 0) / 
                    len(optimizer_data[opt]['acc_changes']) * 100 
                    for opt in optimizer_names]
    ax.bar(optimizer_names, success_rates, alpha=0.7, color='green')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Recovery Success Rate (% with positive change)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Degradation rate distribution
    ax = axes[4]
    data = [np.array(optimizer_data[opt]['degradation_rates']) * 1000 for opt in optimizer_names]
    bp = ax.boxplot(data, labels=optimizer_names, patch_artist=True)
    ax.set_ylabel('Degradation Rate (×1000)')
    ax.set_title('Degradation Rate Distribution')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 6: Summary statistics table
    ax = axes[5]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Optimizer', 'Mean Δ Acc', 'Success %', 'Best', 'Worst']
    
    for opt in optimizer_names:
        changes = optimizer_data[opt]['acc_changes']
        mean_change = np.mean(changes)
        success_rate = sum(1 for x in changes if x > 0) / len(changes) * 100
        best = max(changes)
        worst = min(changes)
        
        table_data.append([
            opt,
            f'{mean_change:+.4f}',
            f'{success_rate:.1f}%',
            f'{best:+.4f}',
            f'{worst:+.4f}'
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Summary Statistics', pad=20)
    
    plt.suptitle('Optimizer Mitigation Experiment - Summary Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'summary_visualizations.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   • Summary visualizations saved to: summary_visualizations.png")