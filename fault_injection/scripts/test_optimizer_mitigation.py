#!/usr/bin/env python3
"""
Optimizer Mitigation Experiment - Main Script
Tests whether changing optimizers after fault injection can mitigate slowdegrade effects.
"""

import tensorflow as tf
import time
import argparse
from typing import List, Dict

# Configure TensorFlow
print("="*80)
print("INITIALIZING TENSORFLOW CONFIGURATION")
print("="*80)
tf.config.set_visible_devices([], 'GPU')
tf.config.set_soft_device_placement(True)
print("✓ TensorFlow configured for CPU execution")
print("✓ Soft device placement enabled")
print()

# Import utilities
from utils import (
    ExperimentConfig,
    PhaseRunner,
    analyze_recovery_performance,
    create_experiment_visualizations,
    save_experiment_results,
    save_intermediate_summary,
    generate_final_report,
    save_error_info,
    create_summary_visualization
)


class OptimizerMitigationExperiment:
    """
    Main experiment class that orchestrates the optimizer mitigation tests.
    Uses modular utilities for cleaner code organization.
    """
    
    def __init__(self, 
                 baseline_optimizer: str = 'adam',
                 test_optimizers: List[str] = None,
                 num_experiments: int = 100,
                 base_seed: int = 42,
                 learning_rate: float = 0.001,
                 max_steps_after_injection: int = 200):
        """Initialize the experiment with configuration."""
        
        # Setup configuration
        self.config = ExperimentConfig(
            baseline_optimizer=baseline_optimizer,
            test_optimizers=test_optimizers,
            num_experiments=num_experiments,
            base_seed=base_seed,
            learning_rate=learning_rate,
            max_steps_after_injection=max_steps_after_injection
        )
        
        # Log configuration
        self.config.log_configuration()
        
        # Initialize phase runner
        self.phase_runner = PhaseRunner()
    
    def run_single_experiment(self, experiment_id: int) -> Dict:
        """
        Run complete experiment: create corruption, then test all optimizers.
        
        Args:
            experiment_id: ID of the experiment to run
            
        Returns:
            Dictionary with experiment results or None if failed
        """
        print(f"\n" + "="*80)
        print(f"=" * 80)
        print(f"STARTING EXPERIMENT {experiment_id + 1}/{self.config.num_experiments}")
        print(f"=" * 80)
        print(f"=" * 80)
        
        # Get pre-generated injection config
        injection_config = self.config.get_injection_config(experiment_id)
        injection_config['baseline_optimizer'] = self.config.baseline_optimizer
        
        print(f"\n📋 EXPERIMENT CONFIGURATION:")
        print(f"   • Experiment ID: {experiment_id:03d}")
        print(f"   • Seed: {injection_config['seed']}")
        print(f"   • Target: Epoch {injection_config['target_epoch']}, Step {injection_config['target_step']}")
        print(f"   • Layer: {injection_config['target_layer']}")
        print(f"   • Injection value: {injection_config['injection_value']:.2e}")
        
        try:
            # Phase 1: Create corrupted checkpoint
            print(f"\n" + "▶"*40)
            print(f"PHASE 1: CREATING CORRUPTED CHECKPOINT")
            print("▶"*40)
            
            checkpoint_info = self.phase_runner.create_corrupted_checkpoint(
                injection_config, self.config.results_base_dir
            )
            
            # Phase 2: Test each optimizer's recovery
            print(f"\n" + "▶"*40)
            print(f"PHASE 2: TESTING OPTIMIZER RECOVERY")
            print("▶"*40)
            
            results = {
                'experiment_id': experiment_id,
                'injection_config': injection_config,
                'corruption_info': checkpoint_info['corruption_info'],
                'pre_injection_history': checkpoint_info['pre_injection_history'],
                'recovery_results': {}
            }
            
            # Test baseline optimizer
            print(f"\n🔄 Testing baseline optimizer: {self.config.baseline_optimizer}")
            baseline_recovery = self.phase_runner.test_optimizer_recovery(
                checkpoint_info, self.config.baseline_optimizer, 
                injection_config, self.config.max_steps_after_injection
            )
            results['recovery_results'][self.config.baseline_optimizer] = baseline_recovery
            
            # Test alternative optimizers
            for i, optimizer_name in enumerate(self.config.test_optimizers, 1):
                print(f"\n🔄 Testing alternative optimizer {i}/{len(self.config.test_optimizers)}: {optimizer_name}")
                recovery = self.phase_runner.test_optimizer_recovery(
                    checkpoint_info, optimizer_name, 
                    injection_config, self.config.max_steps_after_injection
                )
                results['recovery_results'][optimizer_name] = recovery
            
            # Save results
            print(f"\n💾 Saving experiment results...")
            save_experiment_results(results, checkpoint_info['experiment_dir'])
            
            # Create visualizations
            print(f"\n📊 Creating experiment visualizations...")
            create_experiment_visualizations(results, checkpoint_info['experiment_dir'])
            
            # Analyze performance
            print(f"\n" + "="*80)
            print(f"EXPERIMENT {experiment_id + 1} COMPLETE - SUMMARY")
            print("="*80)
            
            analysis = analyze_recovery_performance(results['recovery_results'])
            
            print("="*80 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR in experiment {experiment_id}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save error info
            save_error_info(experiment_id, e, self.config.results_base_dir)
            
            return None
    
    def run(self) -> List[Dict]:
        """
        Run all experiments.
        
        Returns:
            List of experiment results
        """
        print(f"\n" + "="*80)
        print("="*80)
        print("STARTING FULL EXPERIMENT SUITE")
        print("="*80)
        print("="*80)
        
        print(f"\n📊 EXPERIMENT OVERVIEW:")
        print(f"   • Total experiments: {self.config.num_experiments}")
        print(f"   • Baseline optimizer: {self.config.baseline_optimizer}")
        print(f"   • Test optimizers: {', '.join(self.config.test_optimizers)}")
        print(f"   • Results directory: {self.config.results_base_dir}")
        
        all_results = []
        successful_experiments = 0
        failed_experiments = []
        
        overall_start_time = time.time()
        
        for exp_id in range(self.config.num_experiments):
            exp_start_time = time.time()
            
            print(f"\n" + "▶"*80)
            print(f"EXPERIMENT {exp_id + 1}/{self.config.num_experiments}")
            print("▶"*80)
            
            results = self.run_single_experiment(exp_id)
            
            if results:
                all_results.append(results)
                successful_experiments += 1
                exp_duration = time.time() - exp_start_time
                print(f"✅ Experiment {exp_id + 1} completed in {exp_duration:.1f} seconds")
            else:
                failed_experiments.append(exp_id)
                print(f"❌ Experiment {exp_id + 1} failed")
            
            # Save intermediate summary every 10 experiments
            if (exp_id + 1) % 10 == 0:
                print(f"\n📊 INTERMEDIATE CHECKPOINT (after {exp_id + 1} experiments)")
                print(f"   • Successful: {successful_experiments}")
                print(f"   • Failed: {len(failed_experiments)}")
                print(f"   • Elapsed time: {(time.time() - overall_start_time)/60:.1f} minutes")
                save_intermediate_summary(
                    all_results, self.config.results_base_dir,
                    self.config.baseline_optimizer, self.config.test_optimizers
                )
                print(f"   ✓ Intermediate summary saved")
        
        # Generate final report
        print(f"\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)
        
        generate_final_report(all_results, self.config.results_base_dir, self.config)
        
        # Create summary visualizations
        print(f"\nCreating summary visualizations...")
        create_summary_visualization(all_results, self.config.results_base_dir)
        
        total_duration = time.time() - overall_start_time
        
        print(f"\n" + "="*80)
        print("="*80)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print("="*80)
        
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   • Total experiments: {self.config.num_experiments}")
        print(f"   • Successful: {successful_experiments}")
        print(f"   • Failed: {len(failed_experiments)}")
        if failed_experiments:
            print(f"   • Failed experiment IDs: {failed_experiments}")
        print(f"   • Total duration: {total_duration/60:.1f} minutes")
        print(f"   • Average time per experiment: {total_duration/self.config.num_experiments:.1f} seconds")
        print(f"\n📁 Results saved to: {self.config.results_base_dir}")
        print("="*80)
        
        return all_results


def main():
    """Main function to run the experiment."""
    
    print("\n" + "="*80)
    print("OPTIMIZER MITIGATION EXPERIMENT")
    print("="*80)
    
    parser = argparse.ArgumentParser(
        description='Test optimizer mitigation for slowdegrade effects'
    )
    parser.add_argument('--baseline', type=str, default='adam',
                       help='Baseline optimizer (default: adam)')
    parser.add_argument('--test-optimizers', type=str, nargs='+',
                       default=['sgd', 'rmsprop', 'adamw'],
                       help='Test optimizers for mitigation')
    parser.add_argument('--num-experiments', type=int, default=100,
                       help='Number of experiments to run (default: 100)')
    parser.add_argument('--steps-after-injection', type=int, default=200,
                       help='Steps to train after injection (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("\n📋 COMMAND LINE ARGUMENTS:")
    print(f"   • Baseline: {args.baseline}")
    print(f"   • Test optimizers: {args.test_optimizers}")
    print(f"   • Number of experiments: {args.num_experiments}")
    print(f"   • Steps after injection: {args.steps_after_injection}")
    print(f"   • Base seed: {args.seed}")
    print(f"   • Learning rate: {args.learning_rate}")
    
    print("\n🚀 KEY FEATURES:")
    print("   ✓ Modular code organization with utility functions")
    print("   ✓ Full visibility into execution")
    print("   ✓ Comprehensive error reporting")
    print("   ✓ Real-time metrics and visualizations")
    print("   ✓ Intermediate checkpointing")
    
    # Create and run experiment
    experiment = OptimizerMitigationExperiment(
        baseline_optimizer=args.baseline,
        test_optimizers=args.test_optimizers,
        num_experiments=args.num_experiments,
        base_seed=args.seed,
        learning_rate=args.learning_rate,
        max_steps_after_injection=args.steps_after_injection
    )
    
    print("\n🎬 STARTING EXPERIMENTS...")
    print("="*80)
    
    results = experiment.run()
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"📁 Full results available at: {experiment.config.results_base_dir}")
    print("="*80)


if __name__ == "__main__":
    main()