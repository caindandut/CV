# Implementation Plan: E7 SCConv + Hard Example Mining

## Scope

Create `Train3_YOLO26s_SCConv_HardMining.ipynb` as a standalone E7 experiment. Keep the E5 architecture and exact training protocol; add only train-split hard-example mining and weighted sampling.

## Ordered slices

1. Reuse and verify the E5 runtime graph, SCConv implementation, and two neck replacements.
2. Load E5 only for train-image mining and produce `hard_samples_train.csv`.
3. Implement a version-8.4.155-compatible train-only `WeightedRandomSampler` while keeping validation loading standard.
4. Build E7 from fresh `yolo26s.pt`, train with the unchanged protocol, and preserve the custom graph.
5. Validate E7, report no-helmet confusion counts/diagonal, compare E5 vs E7 on the same validation split, and run the final audit.

## Verification gates

- Notebook JSON and all Python cells compile.
- Mining report uses only `train/images` and `train/labels`.
- E7 starts from `yolo26s.pt`, not E5/E6.
- Exactly two `C3k2_SCConv` modules, no DySample/P2/attention modules.
- Train sampler is weighted with replacement and `num_samples == len(train_dataset)`.
- Validation sampler is not weighted.
- Training protocol remains epochs 50, imgsz 640, batch 16, device 0, workers 4, optimizer auto, seed 0, deterministic true, patience 15.
- Test split is never used.

## Runtime limitation

The local workspace has no PyTorch/Ultralytics GPU runtime, so Colab execution remains the runtime verification gate. The notebook contains assertions and sampler diagnostics for that environment.
