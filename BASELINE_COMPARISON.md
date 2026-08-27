# Matched M3GNet Baseline

This repository branch defines the controlled baseline for evaluating the
Transformer contribution.

## Architectures

- Baseline: `M3GNet -> Linear atomic-energy head -> atomic sum`
- Target: `M3GNet -> Transformer -> Linear atomic-energy head -> atomic sum`

The baseline intentionally retains the same M3GNet graph construction,
three-body interactions, message-passing blocks, linear atomic-energy head,
energy/force/stress loss, data split, and optimisation settings used by the
target model. Only Transformer-based feature exchange is omitted.

The gated operations inside the original M3GNet interaction blocks are not
removed. `linear_atomic_sum` replaces only the final gated atomic-energy
readout, preventing the readout design from confounding the Transformer
comparison.

## Matched training settings

- Dataset: MatPES-PBE-2025.2
- Split: 90/5/5 with random seed 42
- Epochs: 200
- Batch size: 32
- Gradient accumulation: 4 batches
- Energy/force/stress loss weights: 1.0/1.0/0.1
- Loss: Huber
- Learning rate: 1e-3
- Gradient clipping: 2.0
- Checkpoint selection: lowest validation total loss

## Gadi execution

The PBS script reuses the existing MatPES data and graph cache. Submit staged
jobs with increasing target epochs:

```bash
qsub -v TARGET_EPOCHS=50 train_m3gnet_linear_gpu.pbs
qsub -v TARGET_EPOCHS=100 train_m3gnet_linear_gpu.pbs
qsub -v TARGET_EPOCHS=150 train_m3gnet_linear_gpu.pbs
qsub -v TARGET_EPOCHS=200 train_m3gnet_linear_gpu.pbs
```

Submit each later stage after the previous job has completed. The script
automatically resumes from `checkpoints/last.ckpt` when it exists.

Final comparisons should use the best-validation checkpoint from each model,
not the final epoch or intermediate fixed checkpoints.
