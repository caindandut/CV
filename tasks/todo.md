# E6 Notebook Checklist

- [x] New notebook name and independent output directory
- [x] Pin `ultralytics==8.4.155`
- [x] Validate DataSet2 and class order
- [x] Runtime-select P5->P4 and P4->P3 upsampling nodes
- [x] Insert exactly two official DySample modules
- [x] Runtime-select exactly two neck C3k2 nodes at strides 16 and 8
- [x] Insert SCConv through Bottleneck_SCConv inside C3k2_SCConv
- [x] Transfer compatible YOLO26s pretrained weights and report counts
- [x] Dummy forward, shape hooks, and gradient checks
- [x] Custom DetectionTrainer with exact protocol
- [x] Guard against pre-existing experiment directory
- [x] Rebuild/load best checkpoint without losing custom modules
- [x] Validation-only metrics, plots, confusion errors, and no-helmet diagonal
- [x] E0/E4/E5/E6 comparison and complexity benchmark
- [x] Final audit and no test-set usage
