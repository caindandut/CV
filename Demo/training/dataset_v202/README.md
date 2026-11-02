# Dataset snapshot v202

This snapshot inherits earlier versioned image assets and records the full manifest. It adds two manually reviewed train crops from user-provided DataSet2 videos. The full source frames remain pending and do not count toward the goal.

- Accepted clean images: 757 / 5,000
- Remaining: 4,243
- Pending review: 1,492 (not counted)
- Split counts: train 603, validation 86, locked test 68
- Added crops: one helmeted driver (`IMG_5280_mp4-3`) and one motorcycle carrying a helmeted driver plus unhelmeted passenger (`IMG_5030-2_MOV-2`).
- Class 0 motorcycle; class 1 seated/driving rider wearing a helmet; class 2 seated/driving rider without a helmet.
- The DataSet2 user confirmed ownership or permission on 2026-11-01. Standard license terms were not supplied; no open license is asserted.

See `manifest.json`, `manual_review_batch37.json`, `source_metadata`, `review_evidence/manual_batch_37`, and `../verification/dataset_v202.json`.
