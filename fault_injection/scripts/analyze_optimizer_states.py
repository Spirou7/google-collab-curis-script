#!/usr/bin/env python3
"""
Detailed Optimizer State Analysis Tool

This script provides detailed visualizations of optimizer internal states,
showing how different variables and slot types evolve during training.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional
import argparse
from pathlib import Path


def load_experiment_results(results_dir: str) -> Dict:
    """Load experiment results from directory."""
    results = {}
    
    # Find the latest experiment directory
    base_path = Path(results_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for optimizer result files
    for opt_file in base_path.glob("*_result.json"):
        opt_name = opt_file.stem.replace("_result", "")
        with open(opt_file, 'r') as f:
            results[opt_name] = json.load(f)
    
    # Also try pickle files if JSON files not found
    if not results:
        for opt_file in base_path.glob("*_result.pkl"):
            opt_name = opt_file.stem.replace("_result", "")
            with open(opt_file, 'rb') as f:
                results[opt_name] = pickle.load(f)
    
    return results


def analyze_detailed_states(results: Dict) -> Dict:
    """Analyze detailed optimizer states to understand structure."""
    analysis = {
        'slot_types': set(),
        'variable_names': set(),
        'variable_groups': {},
        'optimizer_slots': {}
    }
    
    for opt_name, result in results.items():
        if 'history' not in result:
            continue
            
        history = result['history']
        if 'detailed_optimizer_states' not in history:
            continue
        
        detailed_states = history['detailed_optimizer_states']
        if not detailed_states:
            continue
        
        # Analyze first non-empty state
        for state in detailed_states:
            if state:
                analysis['optimizer_slots'][opt_name] = set()
                
                for var_name, slots in state.items():
                    if var_name == '_iteration':
                        continue
                    
                    if isinstance(slots, dict):
                        analysis['variable_names'].add(var_name)
                        
                        # Categorize variables
                        if 'conv' in var_name.lower():
                            group = 'Convolutional'
                        elif 'dense' in var_name.lower() or 'fc' in var_name.lower():
                            group = 'Dense/FC'
                        elif 'batch' in var_name.lower() or 'bn' in var_name.lower():
                            group = 'BatchNorm'
                        elif 'bias' in var_name.lower():
                            group = 'Bias'
                        elif 'kernel' in var_name.lower() or 'weight' in var_name.lower():
                            group = 'Weights'
                        else:
                            group = 'Other'
                        
                        if group not in analysis['variable_groups']:
                            analysis['variable_groups'][group] = set()
                        analysis['variable_groups'][group].add(var_name)
                        
                        # Track slot types
                        for slot_name in slots.keys():
                            analysis['slot_types'].add(slot_name)
                            analysis['optimizer_slots'][opt_name].add(slot_name)
                
                break  # Only need first state for structure
    
    return analysis


def plot_detailed_state_evolution(results: Dict, analysis: Dict, output_dir: str):
    """Create detailed plots showing state evolution per variable type and slot."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get color maps
    optimizers = list(results.keys())
    opt_colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))
    
    # Find injection step if available
    injection_step = None
    for result in results.values():
        if 'actual_injection_step' in result:
            injection_step = result['actual_injection_step']
            break
    
    # Plot 1: Comparison of slot types across optimizers
    if analysis['slot_types']:
        fig, axes = plt.subplots(
            len(analysis['slot_types']), 1, 
            figsize=(12, 4 * len(analysis['slot_types'])),
            squeeze=False
        )
        
        for slot_idx, slot_type in enumerate(sorted(analysis['slot_types'])):
            ax = axes[slot_idx, 0]
            
            for opt_idx, (opt_name, result) in enumerate(results.items()):
                if 'history' not in result or 'detailed_optimizer_states' not in result['history']:
                    continue
                
                history = result['history']
                detailed_states = history['detailed_optimizer_states']
                steps = history['steps'][:len(detailed_states)]
                
                # Calculate total magnitude for this slot type
                slot_magnitudes = []
                for state in detailed_states:
                    total_mag = 0.0
                    count = 0
                    for var_name, slots in state.items():
                        if var_name != '_iteration' and isinstance(slots, dict):
                            if slot_type in slots:
                                total_mag += slots[slot_type]
                                count += 1
                    
                    if count > 0:
                        slot_magnitudes.append(total_mag / count)
                    else:
                        slot_magnitudes.append(0.0)
                
                if any(m > 0 for m in slot_magnitudes):
                    ax.plot(steps[:len(slot_magnitudes)], slot_magnitudes,
                           label=opt_name, color=opt_colors[opt_idx],
                           linewidth=2, alpha=0.8)
            
            if injection_step:
                ax.axvline(x=injection_step, color='red', linestyle='--', 
                          label='Injection', alpha=0.7)
            
            ax.set_title(f'Slot Type: {slot_type}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Average Magnitude')
            ax.set_yscale('log')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Optimizer State Evolution by Slot Type', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_by_slot_type.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: State evolution by variable group
    if analysis['variable_groups']:
        n_groups = len(analysis['variable_groups'])
        fig, axes = plt.subplots(
            n_groups, len(analysis['slot_types']) if analysis['slot_types'] else 1,
            figsize=(5 * max(len(analysis['slot_types']), 1), 4 * n_groups),
            squeeze=False
        )
        
        for group_idx, (group_name, var_set) in enumerate(sorted(analysis['variable_groups'].items())):
            for slot_idx, slot_type in enumerate(sorted(analysis['slot_types']) if analysis['slot_types'] else ['']):
                ax = axes[group_idx, slot_idx] if analysis['slot_types'] else axes[group_idx, 0]
                
                for opt_idx, (opt_name, result) in enumerate(results.items()):
                    if 'history' not in result or 'detailed_optimizer_states' not in result['history']:
                        continue
                    
                    history = result['history']
                    detailed_states = history['detailed_optimizer_states']
                    steps = history['steps'][:len(detailed_states)]
                    
                    # Calculate magnitude for this group and slot
                    group_slot_mags = []
                    for state in detailed_states:
                        total_mag = 0.0
                        count = 0
                        
                        for var_name in var_set:
                            if var_name in state and isinstance(state[var_name], dict):
                                if slot_type in state[var_name]:
                                    total_mag += state[var_name][slot_type]
                                    count += 1
                        
                        if count > 0:
                            group_slot_mags.append(total_mag / count)
                        else:
                            group_slot_mags.append(0.0)
                    
                    if any(m > 0 for m in group_slot_mags):
                        ax.plot(steps[:len(group_slot_mags)], group_slot_mags,
                               label=opt_name, color=opt_colors[opt_idx],
                               linewidth=2, alpha=0.8)
                
                if injection_step:
                    ax.axvline(x=injection_step, color='red', linestyle='--', alpha=0.7)
                
                ax.set_title(f'{group_name} - {slot_type}' if slot_type else group_name)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Average Magnitude')
                ax.set_yscale('log')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('State Evolution by Variable Group and Slot Type', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_by_variable_group.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Heatmap showing which optimizers have which state types
    if analysis['optimizer_slots']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_slots = sorted(analysis['slot_types']) if analysis['slot_types'] else []
        all_optimizers = sorted(analysis['optimizer_slots'].keys())
        
        # Create matrix
        matrix = np.zeros((len(all_optimizers), len(all_slots)))
        for i, opt in enumerate(all_optimizers):
            for j, slot in enumerate(all_slots):
                if slot in analysis['optimizer_slots'].get(opt, set()):
                    matrix[i, j] = 1
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(all_slots)))
        ax.set_yticks(np.arange(len(all_optimizers)))
        ax.set_xticklabels(all_slots)
        ax.set_yticklabels(all_optimizers)
        
        # Add text annotations
        for i in range(len(all_optimizers)):
            for j in range(len(all_slots)):
                text = ax.text(j, i, '✓' if matrix[i, j] else '',
                             ha='center', va='center', color='black' if matrix[i, j] else 'gray')
        
        ax.set_title('Optimizer State Types Matrix')
        ax.set_xlabel('Slot Type')
        ax.set_ylabel('Optimizer')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimizer_state_matrix.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 4: State magnitude comparison at injection point
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8 / len(optimizers)
    x_positions = np.arange(len(analysis['slot_types']) if analysis['slot_types'] else 1)
    
    for opt_idx, (opt_name, result) in enumerate(results.items()):
        if 'history' not in result or 'detailed_optimizer_states' not in result['history']:
            continue
        
        history = result['history']
        detailed_states = history['detailed_optimizer_states']
        
        if not detailed_states:
            continue
        
        # Get state at injection point (or closest)
        injection_idx = -1
        if injection_step and 'steps' in history:
            for idx, step in enumerate(history['steps']):
                if step >= injection_step:
                    injection_idx = idx
                    break
        
        if injection_idx >= 0 and injection_idx < len(detailed_states):
            state = detailed_states[injection_idx]
            
            slot_mags = []
            for slot_type in sorted(analysis['slot_types']) if analysis['slot_types'] else ['']:
                total_mag = 0.0
                count = 0
                
                for var_name, slots in state.items():
                    if var_name != '_iteration' and isinstance(slots, dict):
                        if slot_type in slots:
                            total_mag += slots[slot_type]
                            count += 1
                
                slot_mags.append(total_mag / count if count > 0 else 0)
            
            x_offset = x_positions + opt_idx * bar_width
            ax.bar(x_offset, slot_mags, bar_width, label=opt_name,
                  color=opt_colors[opt_idx], alpha=0.8)
    
    ax.set_xlabel('Slot Type')
    ax.set_ylabel('Average Magnitude at Injection')
    ax.set_title('Optimizer State Comparison at Injection Point')
    ax.set_xticks(x_positions + bar_width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels(sorted(analysis['slot_types']) if analysis['slot_types'] else [''])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_at_injection.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZER STATE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Optimizers analyzed: {', '.join(optimizers)}")
    print(f"Slot types found: {', '.join(sorted(analysis['slot_types'])) if analysis['slot_types'] else 'None'}")
    print(f"Variable groups: {', '.join(sorted(analysis['variable_groups'].keys()))}")
    print("\nOptimizer-specific slots:")
    for opt, slots in analysis['optimizer_slots'].items():
        print(f"  {opt}: {', '.join(sorted(slots)) if slots else 'No internal state'}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze detailed optimizer states')
    parser.add_argument('results_dir', help='Path to experiment results directory')
    parser.add_argument('--output', '-o', default='state_analysis',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Analyze structure
    print("Analyzing optimizer state structure...")
    analysis = analyze_detailed_states(results)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_detailed_state_evolution(results, analysis, args.output)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()