# Implementation Plan: E6 YOLO26s + SCConv + DySample

## Overview

Create a new Colab-ready notebook `Train3_YOLO26s_SCConv_DySample.ipynb` for the E6 ablation. The notebook must start from `yolo26s.pt`, preserve the DataSet2 class order and training protocol, insert exactly two runtime-selected DySample modules and exactly two neck `C3k2_SCConv` modules, preserve compatible pretrained weights, validate the custom graph before training, train with a custom `DetectionTrainer`, and report validation metrics, confusion-matrix errors, plots, complexity, and comparison with E0/E4/E5.

## Architecture Decisions

- Build all E6 models from `YOLO('yolo26s.pt')`; do not load E1-E5 checkpoints as initialization.
- Detect the two `nn.Upsample(scale_factor=2)` neck nodes and two neck `C3k2` nodes from the live YOLO26s graph.
- Reuse the checked DySample implementation (`style='lp'`, `scale=2`, `dyscope=True`) and the checked SCConv `SRU -> CRU` implementation/configuration.
- Replace the second convolution inside each target bottleneck with `ScConv`, keeping the original C3k2 CSP path and channel dimensions.
- Transfer only shape-compatible state tensors from the baseline model; report transferred and new parameter counts.
- Use validation only for model selection and comparison; do not invoke the test split.

## Task List

### Phase 1: Foundation

- [x] Inspect E0-E5 notebooks and identify the verified DySample and SCConv implementations.
- [x] Write the E6 notebook structure and immutable dataset/training configuration.

### Phase 2: Custom architecture

- [x] Implement runtime graph inspection and insertion of exactly two DySample nodes.
- [x] Implement SCConv, `Bottleneck_SCConv`, and `C3k2_SCConv`, then replace exactly two neck nodes.
- [x] Add parameter-transfer reporting, dummy forward hooks, and backward gradient checks.

### Checkpoint: Architecture

- [ ] Notebook JSON parses successfully.
- [ ] Static audit confirms the notebook contains the required counts, assertions, protocol, and no test-split training.
- [ ] Colab execution remains the runtime verification point because the local workspace does not contain the Colab GPU/data environment.

### Phase 3: Training and evaluation

- [x] Add custom trainer, output-folder guard, exact E6 training overrides, and custom checkpoint reconstruction.
- [x] Add overall/per-class validation metrics, no-helmet error analysis, plots, and complexity benchmark.
- [x] Add direct E0/E4/E5/E6 validation comparison and measured KEEP/REJECT decision.
- [x] Add final audit for module counts, gradients, trainer preservation, checkpoint preservation, protocol, dataset, and forbidden modules.

### Checkpoint: Complete

- [ ] Final notebook is saved as `Train3_YOLO26s_SCConv_DySample.ipynb` without modifying E0-E5 notebooks.
- [ ] Static validation passes and the notebook is ready to run in the specified Colab environment.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Runtime graph differs from an earlier YOLO release | High | Probe module classes, forward shapes, Detect inputs, and strides at runtime; fail loudly if the expected two nodes are not found. |
| Custom checkpoint deserialization drops modules | High | Try direct load, then rebuild YOLO26s and load compatible state tensors; assert both module counts afterward. |
| Ultralytics trainer rebuilds the standard graph | High | Override `get_model`/`setup_model`, assign the custom PyTorch graph, and assert counts before and inside training. |
| SCConv channel constraints differ at a target node | Medium | Keep the requested E5 configuration and fail with the target details instead of silently changing the architecture. |
| E4/E5 checkpoints are unavailable during comparison | Medium | Resolve the expected Drive paths and raise an actionable `FileNotFoundError`. |

## Open Questions

- None required for notebook construction; the Colab runtime supplies the GPU, Drive mount, dataset, and prior E0/E4/E5 checkpoints.
