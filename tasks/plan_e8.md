# Implementation Plan: E8 YOLO26s + SCConv at 768 px

## Overview

Create `Train3_YOLO26s_SCConv_768.ipynb` as an independent Colab notebook for E8. It starts from `yolo26s.pt`, applies the verified E5 SCConv replacement to exactly two neck `C3k2` nodes at P3/8 and P4/16, and changes only the training input resolution from 640 to 768.

## Architecture decisions

- Keep Ultralytics pinned to `8.4.155` and use the exact DataSet2 YAML/class order from E5.
- Discover the two neck targets from the live YOLO26s graph instead of relying on hard-coded layer indices.
- Reuse the E5 `GroupBatchnorm2d -> SRU -> CRU -> ScConv` implementation and configuration.
- Build from `YOLO('yolo26s.pt')`; transfer only shape-compatible pretrained tensors into the custom graph.
- Use a custom `DetectionTrainer` only to preserve the custom PyTorch graph; leave its dataloader standard.
- Evaluate on `val` only and write E8 results to a fresh output directory.

## Ordered tasks

### Phase 1: Notebook foundation

- [x] Inspect E5 and existing experiment conventions.
- [x] Define the E8-specific paths, protocol, augmentation, and audit constants.

### Phase 2: Architecture and preflight

- [x] Add runtime neck discovery and exact SCConv implementation.
- [x] Replace exactly two target `C3k2` modules and transfer compatible YOLO26s weights.
- [x] Add 768 dummy forward and SCConv backward-gradient checks.

### Checkpoint: Architecture

- [ ] Notebook JSON parses.
- [ ] Static audit finds `C3k2_SCConv == 2`, `imgsz == 768`, `batch == 16`, and no hard-mining/sampler code.
- [ ] Colab runtime confirms GPU, dataset, module counts, forward, and backward checks.

### Phase 3: Training and evaluation

- [x] Add output-directory guard and standard-dataloader custom trainer.
- [x] Add train-only validation protocol and robust custom-checkpoint loading.
- [x] Add overall/per-class metrics, no-helmet confusion statistics, plots, and complexity benchmark.
- [x] Add measured E5@640 versus E8@768 comparison and keep/reject guidance without hard-coding outcomes.

### Checkpoint: Complete

- [ ] Final audit cell passes in the target Colab runtime.
- [ ] Notebook is ready to run without modifying E5 or using the test split.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| Runtime graph differs from the checked E5 graph | High | Probe module classes, Detect inputs, strides, and shapes; fail loudly unless exactly one target exists at each stride. |
| 768 batch of 16 exceeds GPU memory | High | Run the required preflight before training and stop with an explicit message; never silently change the batch. |
| Custom checkpoint deserialization loses SCConv modules | High | Try direct load, then rebuild the custom graph and load compatible tensors; assert both custom blocks remain. |
| E5 checkpoint is unavailable for comparison | Medium | Resolve its expected Drive path and raise an actionable error rather than substituting another model. |
| Higher resolution increases latency/VRAM | Low | Report parameters, GFLOPs, latency, FPS, and peak VRAM separately; do not treat this as an architecture failure. |

## Open questions

None for notebook construction. The actual GPU/data run remains a Colab-side verification step.
