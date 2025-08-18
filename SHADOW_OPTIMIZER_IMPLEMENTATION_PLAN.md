# Shadow Optimizer Framework Implementation Plan

## Overview
Implement a reproducible, context-equalized training framework with shadow optimizers, full-state checkpointing, and step-aligned fault injection for TensorFlow/CPU environments.

## Phase 1: Core Infrastructure (Foundation)

### Files to Create:
1. **`fault_injection/core/determinism.py`**
   - Implement global seed control
   - Dataset determinism options
   - Augmentation seeding utilities

2. **`fault_injection/core/manifest.py`**
   - Run manifest schema definition
   - Manifest I/O utilities
   - Versioning support

### Files to Modify:
1. **`fault_injection/scripts/utils/experiment_config.py`**
   - Add manifest generation
   - Integrate determinism controls
   - Add schema versioning

## Phase 2: Shadow Optimizer Engine

### Files to Create:
1. **`fault_injection/shadows/shadow_base.py`**
   - Abstract base class for shadow optimizers
   - Common slot management methods
   - State dict interface

2. **`fault_injection/shadows/shadow_sgd.py`**
   - ShadowSGD implementation with momentum tracking
   - Slot: momentum buffer

3. **`fault_injection/shadows/shadow_adam.py`**
   - ShadowAdam implementation
   - Slots: m (mean), v (variance), iterations

4. **`fault_injection/shadows/shadow_rmsprop.py`**
   - ShadowRMSProp implementation
   - Slots: rms, momentum (if centered)

5. **`fault_injection/shadows/shadow_adagrad.py`**
   - ShadowAdagrad implementation
   - Slot: accumulator

6. **`fault_injection/shadows/__init__.py`**
   - Export all shadow classes
   - Factory function for shadow creation

## Phase 3: Enhanced Training Loop

### Files to Create:
1. **`fault_injection/core/trainer.py`**
   - Primary training loop with shadow updates
   - Step accounting and synchronization
   - Gradient collection and distribution

### Files to Modify:
1. **`fault_injection/scripts/utils/training_utils.py`**
   - Integrate shadow optimizer updates
   - Add gradient collection hooks
   - Support dual-path training

2. **`fault_injection/scripts/utils/experiment_runner.py`**
   - Add shadow initialization
   - Integrate with new trainer

## Phase 4: Advanced Checkpoint System

### Files to Create:
1. **`fault_injection/core/ckpt.py`**
   - Comprehensive checkpoint manager
   - Save: weights, buffers, shadows, scheduler, loss-scale
   - Schema versioned checkpoints
   - Atomic save operations

2. **`fault_injection/core/transplant.py`**
   - Force Keras slot creation
   - Shadow-to-optimizer state transfer
   - State validation utilities

### Files to Modify:
1. **`fault_injection/scripts/utils/checkpoint_utils.py`**
   - Integrate new checkpoint system
   - Add backward compatibility layer
   - Enhanced restoration logic

## Phase 5: Observability & Metrics

### Files to Create:
1. **`fault_injection/core/metrics.py`**
   - TensorBoard integration
   - Per-layer gradient/activation variance tracking
   - BN stats trajectories
   - Slot norm monitoring (‖u‖, ‖m‖, √v)
   - Recovery metrics

### Files to Modify:
1. **`fault_injection/scripts/utils/metrics_utils.py`**
   - Integrate new metrics system
   - Add drift monitors
   - Enhanced recovery analysis

## Phase 6: CLI & Configuration

### Files to Create:
1. **`fault_injection/cli.py`**
   - Main CLI entry point
   - Commands: prefix-run, resume-with, run-all
   - Configuration loading

2. **`fault_injection/configs/experiment_template.yaml`**
   - Template configuration file
   - Example configurations
   - Documentation

### Files to Modify:
1. **`fault_injection/scripts/test_optimizer_mitigation.py`**
   - Integrate with new framework
   - Use shadow optimizers
   - Support transplant operations

## Key Design Decisions

1. **Shadow Storage**: Use `tf.Variable` keyed by `var.ref()` for efficient lookups
2. **Slot Names**: Map to Keras conventions (SGD:"momentum", Adam:"m"/"v", etc.)
3. **Determinism**: Use `tf.data.Options.experimental_deterministic=True`
4. **Checkpoints**: Atomic saves with schema versioning for forward compatibility
5. **Memory Management**: Feature flags to enable/disable specific shadows
6. **Integration**: Work with existing injection system, no replacement

## Testing Strategy

1. **Unit Tests**: Each module with >90% coverage
2. **Integration Tests**: End-to-end training scenarios
3. **Determinism Tests**: Verify identical runs with same seeds
4. **Performance Tests**: Ensure <10% overhead with 2 shadows
5. **Recovery Tests**: Validate optimizer resilience comparisons

## Risk Mitigations

1. **TF Version Compatibility**: Abstract optimizer APIs, test on TF 2.15+
2. **Memory Constraints**: Selective shadow enabling, FP16 support
3. **Checkpoint Size**: Compression options, selective saving
4. **API Changes**: Versioned interfaces, deprecation warnings

## Success Criteria

1. ✓ Identical prefix runs yield same loss/accuracy/BN stats
2. ✓ Transplanted optimizers resume with exact historical state
3. ✓ Existing injection system continues to work seamlessly
4. ✓ Performance overhead <10% with 2 shadows on CPU
5. ✓ Complete observability of all training metrics

## Implementation Status

### Phase 1: Core Infrastructure ✅
- [x] determinism.py - Complete determinism control with seed management, dataset options, and verification utilities
- [x] manifest.py - Comprehensive manifest system for experiment tracking with schema versioning
- [x] Update experiment_config.py - Integrated determinism controller and manifest system with backward compatibility

### Phase 2: Shadow Optimizer Engine ✅
- [x] shadow_base.py - Abstract base class with slot management and state dict interface
- [x] shadow_sgd.py - SGD with momentum tracking
- [x] shadow_adam.py - Adam with m/v moment tracking and AMSGrad support
- [x] shadow_rmsprop.py - RMSProp with RMS accumulator and optional momentum/centering
- [x] shadow_adagrad.py - Adagrad with accumulator tracking
- [x] shadows/__init__.py - Factory functions and exports

### Phase 3: Enhanced Training Loop ⏳
- [ ] trainer.py
- [ ] Update training_utils.py
- [ ] Update experiment_runner.py

### Phase 4: Advanced Checkpoint System ⏳
- [ ] ckpt.py
- [ ] transplant.py
- [ ] Update checkpoint_utils.py

### Phase 5: Observability & Metrics ⏳
- [ ] metrics.py
- [ ] Update metrics_utils.py

### Phase 6: CLI & Configuration ⏳
- [ ] cli.py
- [ ] experiment_template.yaml
- [ ] Update test_optimizer_mitigation.py

---
*This document will be updated as implementation progresses. Each completed item will be marked with ✅ and include notes on any deviations from the original plan.*

## Progress Notes

### Phase 1 Completed (✅)
- **Determinism System**: Comprehensive control over all random seeds (Python, NumPy, TensorFlow)
- **Manifest System**: Full experiment tracking with environment capture, git info, and schema versioning
- **Integration**: Backward-compatible integration with existing experiment configuration

### Phase 2 Completed (✅)
- **Shadow Base Class**: Robust abstract base with slot management and state serialization
- **Shadow Optimizers**: Complete implementations for SGD, Adam, RMSProp, and Adagrad
- **Key Features**: 
  - Track optimizer state without weight updates
  - Support for all major optimizer variants (momentum, AMSGrad, centered RMSProp, etc.)
  - Transplantation methods to copy state into real Keras optimizers
  - Statistics and analysis methods for debugging