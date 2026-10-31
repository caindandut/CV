# E7 Checklist

- [x] New notebook name and independent output directory
- [x] Reuse E5 SCConv architecture with two neck replacements
- [x] Load E5 checkpoint only for mining/evaluation, never E7 initialization
- [x] Train-only dataset statistics
- [x] Hard mining: missed, misclassified, low-confidence, false-positive no-helmet
- [x] Hard score and capped sample weights
- [x] WeightedRandomSampler with replacement for train only
- [x] Standard validation loader
- [x] Fresh YOLO26s initialization and exact E5 protocol
- [x] Validation metrics, confusion matrix, curves, E5 vs E7 comparison
- [x] Final audit and no test-set usage
