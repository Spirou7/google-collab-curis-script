# System Spec: Refactor Training Stack for Shadow Optimizers, Full-State Checkpointing, and Step-Aligned Fault Injection (TensorFlow/CPU)

## 0) Goal (why we’re doing this)

Establish a **reproducible, context-equalized** training framework that:

- Tracks and persists **all temporal state** (model buffers + optimizer slots + schedulers + loss-scale) at a chosen step **S**.
- Builds **shadow optimizers** that accumulate their *own* history from real gradients **without touching weights**.
- Allows **state transplant** so any optimizer can resume at **S** with the *exact* history it would have had.
- Applies **identical, step-indexed injections** to compare optimizer resilience to SlowDegrade under the **same circumstances**.

## 1) Scope

- TensorFlow 2.19 (Keras OptimizerV2) on CPU (single device execution).
- Model families with BatchNorm (BN) or EMA/teacher models.
- Built-in optimizers: SGD(+momentum), Adam/AdamW, RMSProp, Adagrad. (Extensible to Lion/Shampoo/AdaFactor later.)

## 2) Non-Goals

- No framework hop (stay in TF/Keras).
- No attempt to equalize *data* or *compute* noise across hardware beyond strict determinism controls.
- No automatic hyperparameter retuning.

## 3) Definitions (crisp)

- **Buffers (model-side):** non-trainable, step-updated state (e.g., BN `moving_mean/variance`, EMA/teacher weights).
- **Optimizer slots (optimizer-side):** per-variable accumulators (momentum `u`, Adam `m/v`, Adagrad `sum_sq`), step counters, loss-scale.
- **Shadow optimizer:** a module that updates **only** slot tensors from real gradients; **never** updates weights.
- **Injection:** a step-aligned perturbation to grads/activations/optimizer slots/model buffers.

## 4) Functional Requirements

1. **Determinism Controller**
    - Fix global seeds; deterministic `tf.data` order; deterministic augmentation.
2. **Shadow State Engine**
    - Implement `ShadowSGD`, `ShadowAdam`, `ShadowRMSProp`, `ShadowAdagrad` as `tf.Module`s mapping `var.ref() -> slots`.
    - API must explicitly support the following three methods:
    - `build(vars)`: initialize slot tensors (momentum buffers, m/v accumulators, etc.) for each model variable.
    - `update_from_grads(grads, vars)`: given current gradients, update those slot tensors without touching weights.
    - `state_dict()`: return a dictionary view of all slot tensors so they can be checkpointed, restored, or transplanted into a real optimizer later.
3. **Primary Training Loop**
    - One “primary” optimizer updates weights; shadows consume same gradients per step and advance their slots.
4. **Checkpoint Manager**
    - At step **S**, persist **(a)** model weights **θ(S)**, **(b)** all model buffers (BN stats, EMA/teacher), **(c)** **each** shadow state, **(d)** primary optimizer state, **(e)** scheduler phase + warmup state, **(f)** loss-scale state, **(g)** a **run manifest** (seeds, dataset hash, commit, LR schedule state).
5. **Transplant & Resume**
    - Given a chosen optimizer `O_k`, recreate the model and then explicitly create its slot variables (by running a dummy zero-gradient update). After these slot tensors exist, copy the corresponding values from the shadow optimizer into them (for example, copy momentum buffers into `"momentum"`, Adam’s `m` and `v` into `"m"` and `"v"`, Adagrad’s accumulator into `"accumulator"`). This ensures the real optimizer resumes with the exact historical state it would have built if it had been training from the start.
6. **Injection Engine**
    - Must integrate cleanly with the existing injection engine you have already built.
7. **Metrics/Observability (CRITICAL)**
    - This component is **essential** for validating experiments. It must log loss/accuracy; per-layer grad/activation variances; BN mean/var trajectories; slot norms (‖u‖, ‖m‖, √v); gradient norms; and recovery time after injection. Without these metrics, results on SlowDegrade resilience cannot be trusted.
8. **CLI/Config**
    - YAML/JSON config for model/opt/seed/step S/injection schedule; single command to run prefix, checkpoint, and N continuation runs per optimizer.

## 5) Non-Functional Requirements

- **Overhead:** ≤ **5–10%** step-time overhead with two shadows; memory headroom configurable (enable/disable specific shadows).
- **Compatibility:** CPU execution; no TPU/XLA-specific constructs.
- **Robustness:** graceful skip if a slot isn’t supported; schema-versioned checkpoints; explicit failure on determinism violations.

## 6) Architecture

**Modules**

- `determinism.py` — seeds, dataset options, augmentation seeding.
- `shadows/` — `shadow_base.py`, `shadow_sgd.py`, `shadow_adam.py`, `shadow_rmsprop.py`, `shadow_adagrad.py`.
- `trainer.py` — primary loop (weights update) + shadows update; step accounting.
- `ckpt.py` — checkpoint/save/restore; schema versioning; run-manifest I/O.
- `transplant.py` — create Keras slots and copy shadow → slot.
- `injector/` — injection spec parser; hook registration; built-in injectors.
- `metrics.py` — TB summaries, scalars, histograms; drift monitors.
- `cli.py` — `prefix-run`, `resume-with`, `run-all`.

**Control Flow (prefix)**

```
seed_all(); ds = make_dataset(deterministic=True)
model = build_model(); materialize()
primary_opt = AdamW(...)
shadows = [ShadowAdam(...), ShadowSGD(...), ...]; shadows.build(model.vars)

for step in range(1, S+1):
  loss, grads = forward_backward(model, batch)
  primary_opt.apply_gradients(zip(grads, vars))       # weights move
  for sh in shadows: sh.update_from_grads(grads, vars) # slots only
save_checkpoint(S, model, primary_opt, shadows, sched_state, loss_scale, manifest)

```

**Control Flow (resume + inject)**

```
model2 = build_model(); restore_model_and_buffers(S)
test_opt = AdamW(...)  # or SGD, etc.
force_create_keras_slots(test_opt, model2.vars)
copy_shadow_to_slots(test_opt, shadow_state_for(test_opt.name))
injector = build_injector(spec)  # step-aligned

for t in S+1..S+H:
  loss, grads = forward_backward(model2, batch, injector.pre_forward_hooks)
  grads = injector.pre_apply_gradients(grads)
  test_opt.apply_gradients(zip(grads, vars))
  injector.post_apply(model2, test_opt)  # e.g., BN var drift
  log_metrics(...)

```

## 7) Data Model & Checkpoint Contents

**Trackables**

- `model/` — **weights θ(S)** + **non-trainables** (BN `moving_mean`, `moving_variance`, any EMA/teacher weights).
- `optim/primary/` — primary optimizer slots, step counter, hyper state.
- `optim/shadows/<name>/` — each shadow’s slot tensors (`u`, `m`, `v`, `sum_sq`), iterations.
- `sched/` — LR scheduler internal step/phase, warmup progress.
- `loss_scale/` — mixed precision scaler.
- `manifest.json` — seeds, git SHA, dataset fingerprint, LR schedule cfg, optimizer hyperparams, TF version.

**Manifest example**

```json
{
  "schema_version": 2,
  "run_id": "exp_2025_08_15_S20000",
  "tf_version": "2.15.0",
  "seeds": {"global":1337, "data":1337, "aug":1337},
  "dataset": {"name":"CIFAR-10", "split":"train", "order_hash":"..."},
  "schedule": {"type":"cosine", "warmup_steps":1000, "step":20000},
  "primary_opt": {"name":"AdamW", "lr":0.0003, "betas":[0.9,0.999], "wd":0.01},
  "shadows": ["sgd_mom","adam","rmsprop","adagrad"]
}

```

## 8) Public APIs (Python)

```python
# Determinism
seed_all(seed:int); ds = make_dataset(cfg, deterministic=True)

# Shadows
sh = ShadowAdam(beta1=0.9, beta2=0.999); sh.build(vars)
sh.update_from_grads(grads, vars)  # slots only
state = sh.state_dict()            # for inspection

# Checkpoint
save_checkpoint(step=S, model, primary_opt, shadows:list, sched_state, loss_scale, manifest)
restore_checkpoint(dir, into_model, shadows_to_restore:list)

# Transplant
force_create_keras_slots(optimizer, vars)  # zero-grad apply inside
copy_shadow_to_keras(optimizer, shadow_state)

# Injection
inj = Injector.from_yaml("spec.yaml")
inj.register_hooks(hooks={"pre_forward":..., "pre_apply":..., "post_apply":...})

```

## 9) Injection Spec (DSL)

**YAML**

```yaml
injections:
  - name: bn_var_drift
    target: "model.bn.running_var"
    where: "post_forward"
    start_step: 20001
    duration: 5000
    op: "mul"
    magnitude: 1.00001        # multiplicative drift per step
  - name: adam_v_leak
    target: "optimizer.Adam.v"
    where: "post_apply"
    start_step: 20001
    duration: 5000
    op: "mul"
    magnitude: 1.00002
  - name: grad_scale
    target: "gradients"
    where: "pre_apply"
    start_step: 20001
    duration: 5000
    op: "mul"
    magnitude: 1.0001
    mask: "per_layer:conv.*"
seed: 4242

```

**Built-ins**

- `model.bn.running_var|running_mean`
- `optimizer.<Name>.momentum|m|v|accumulator`
- `gradients` (global or per-layer masks)
- Extensibility via Python plugin registry.

## 11) Implementation Notes (TF/CPU)

- Shadows store slot tensors as `tf.Variable` keyed by `var.ref()`.
- Keras slot names: SGD `"momentum"`, Adam `"m"` & `"v"`, RMSProp `"rms"`/`"momentum"` (centered variant), Adagrad `"accumulator"`. Guard for version drift with a name map.
- Force slot creation via one zero-grad `apply_gradients` before `assign`ing shadow tensors.
- Dataset: `options.experimental_deterministic=True`; fixed `shuffle(..., reshuffle_each_iteration=False)`.
- EMA/teacher: store as separate `tf.Module` trackable and include in checkpoints.

## 12) Observability

- TensorBoard: scalars (loss, acc), histograms (per-layer grad/act), custom scalars (‖m‖, ‖v‖, ‖u‖, BN σ² mean/std).
- Drift monitors with alert thresholds (e.g., % change per 1k steps).
- Export a compact **run report** (JSON) per continuation: final metrics, time-to-divergence, recovery slope.

## 13) Migration Plan (phased, no time estimates)

1. Wire **determinism** + **manifest** (no behavioral change).
2. Add **shadow engine** and dual-path training (feature-flagged off by default).
3. Completely overhaul the checkpoint system to hold all of the new information (model weights, buffers, shadows, scheduler, loss-scale, etc.), without maintaining backwards compatibility with previous versions.
4. Add **transplant** utility + validation tests.
5. Integrate **injection engine** + metrics.
6. Roll out in experiments; deprecate v1 checkpoints after bake-in.

## 14) Risks & Mitigations

- **Slot name/API drift across TF/Keras:** centralize a slot-name map; assert presence; fail fast with diagnostics.
- **Memory blow-up with many shadows:** per-optimizer gating; FP16/FP32 control; selective layer lists.
- **Determinism illusions (hidden nondeterminism):** dataset hash in manifest; assert equal hashes pre/post; CI determinism test.

## 15) Acceptance Criteria (must all pass)

- Replaying prefix to **S** twice yields identical: loss curve, BN stats, slot norms, manifest hash.
- After transplant, “no-injection resume” matches continuous run within tolerance on weights and metrics.
- Injection schedules apply identically across optimizers; metrics collected; run reports generated.
- Perf overhead within agreed budget with two shadows on CPU.
