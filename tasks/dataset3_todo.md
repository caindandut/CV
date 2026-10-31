# Dataset3 checklist

- [x] Implement discovery and validation for the two approved source YAMLs
- [x] Implement train/valid/test resolution from YAML
- [x] Implement label, image/label-pair, and source-inventory validation
- [x] Implement exact duplicate analysis and removal decisions
- [x] Implement pHash near-duplicate analysis with threshold <= 4
- [x] Implement guarded copy into Dataset3 with deterministic prefixes
- [x] Implement data.yaml, manifest, class/bbox reports, and summary
- [x] Implement all five contact sheets with bounding boxes and legend
- [x] Implement leakage, split-invariance, structure, and no-training assertions
- [x] Run the full build and report final counts

## Completion note

Vietnam `valid/images` and `valid/labels` were supplied. The YAML paths were resolved within the nested approved source root, and the full Dataset3 build completed with leakage and split-invariance assertions passing.
