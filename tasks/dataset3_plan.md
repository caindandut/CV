# Implementation Plan: Dataset3 construction

## Overview

Build `create_dataset3.py` at the project root and run it to create `Dataset3/` from only the approved Vietnam Dataset2 and Luna v5 YOLO detection datasets. The workflow validates source metadata, labels, image/label pairs, exact and perceptual duplicates, copies only accepted files, produces the requested reports/contact sheets, and performs post-merge leakage and integrity assertions without training a model.

## Architecture decisions

- Discover the approved dataset roots recursively by matching `data.yaml` metadata; fail if either source is absent or ambiguous.
- Resolve `train`, `val`, and `test` paths from each YAML rather than assuming a fixed directory layout.
- Keep all source files read-only; use `shutil.copy2` with deterministic `vn__` and `luna__` destination prefixes.
- Validate class order and every YOLO label before any output dataset is created.
- Use deterministic ordering and a fixed random seed for duplicate decisions, manifests, and contact sheets.
- Use SHA-256 for exact matching and pHash/BK-tree lookup for Hamming-distance near-duplicate checks.
- Treat Vietnam valid/test as immutable target-domain splits and allow Luna to contribute only to train.

## Task list

### Phase 1: Foundation

- [ ] Implement source discovery, YAML metadata validation, split resolution, and dependency checks.
- [ ] Implement label/image-pair validation and source inventory records.

### Phase 2: Deduplication and build

- [ ] Implement exact and pHash near-duplicate filtering with deterministic priority rules.
- [ ] Implement guarded Dataset3 creation, prefixed copies, `data.yaml`, manifest, and reports.

### Checkpoint: Core workflow

- [ ] Focused unit tests pass for YAML resolution, label validation, duplicate priority, and destination naming.
- [ ] Script runs against the approved sources and does not modify source files.

### Phase 3: Analysis and verification

- [ ] Add class/bbox distributions, contact sheets, domain ratio, summary, and final counts.
- [ ] Run post-merge leakage, valid/test invariance, structure, and no-training assertions.

### Checkpoint: Complete

- [ ] `Dataset3/` contains the requested structure and reports.
- [ ] Final console summary reports `DATASET3 CREATED SUCCESSFULLY`, counts, domain ratio, leakage PASS, and `MODEL TRAINING: NOT PERFORMED`.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---:|---|
| Nested or ambiguous source roots | High | Recursive metadata matching and fail-fast validation. |
| Invalid source labels or missing pairs | High | Report every issue; exclude unsafe Luna/train records and stop if Vietnam valid/test cannot remain unchanged. |
| Near-duplicate leakage | High | Deterministic SHA-256 plus pHash BK-tree checks, then repeat checks on Dataset3. |
| Partial overwrite | High | Refuse to run when `Dataset3/` exists and validate sources before creating the output directory. |
| Large contact sheets or hash workload | Medium | Deterministic bounded samples and indexed Hamming search. |

## Open questions

None; the request specifies the source datasets, class order, split policy, reports, and output location.
