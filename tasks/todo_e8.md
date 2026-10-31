# E8 Notebook Checklist

- [x] New notebook name and independent output directory
- [x] Pin `ultralytics==8.4.155`
- [x] Validate DataSet2 path and class order
- [x] Start from `yolo26s.pt`
- [x] Reuse exact E5 SCConv configuration and implementation
- [x] Replace exactly two neck `C3k2` modules at P3/8 and P4/16
- [x] Use standard Ultralytics dataloader without hard-example sampling
- [x] Run 768 dummy forward and backward gradient checks
- [x] Enforce batch 16 without silent fallback
- [x] Guard the fresh experiment directory
- [x] Train with E5 protocol and only change `imgsz` to 768
- [x] Load/rebuild the best custom checkpoint
- [x] Report overall, per-class, no-helmet, confusion, curves, and complexity metrics
- [x] Compare E5@640 with E8@768 using validation only
- [x] Final audit and explicit paper interpretation note
