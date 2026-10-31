"""Fixed-operating-point diagnostics, not COCO AP or a model-selection evaluator."""


def iou(a, b):
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_image(ground_truth, predictions, threshold=0.5):
    counts = [{"tp": 0, "fp": 0, "fn": 0, "gt": 0} for _ in range(3)]
    matched_gt, matched_predictions = set(), set()
    order = sorted(
        range(len(predictions)),
        key=lambda n: predictions[n]["confidence"],
        reverse=True,
    )
    for gt in ground_truth:
        counts[gt["class_id"]]["gt"] += 1
    for pi in order:
        pred = predictions[pi]
        candidates = [
            (iou(pred["xyxy"], gt["xyxy"]), gi)
            for gi, gt in enumerate(ground_truth)
            if gi not in matched_gt and gt["class_id"] == pred["class_id"]
        ]
        overlap, gi = max(candidates, default=(0, -1))
        if overlap >= threshold and gi >= 0:
            matched_gt.add(gi)
            matched_predictions.add(pi)
            counts[pred["class_id"]]["tp"] += 1
        else:
            counts[pred["class_id"]]["fp"] += 1
    missed = [gi for gi in range(len(ground_truth)) if gi not in matched_gt]
    for gi in missed:
        counts[ground_truth[gi]["class_id"]]["fn"] += 1
    false_positives = [pi for pi in order if pi not in matched_predictions]
    # Only unmatched predictions/GT: diagnostic candidates, not confirmed label errors.
    confused_gt, confusion = set(), []
    for pi in false_positives:
        pred = predictions[pi]
        candidates = [
            (iou(pred["xyxy"], ground_truth[gi]["xyxy"]), gi)
            for gi in missed
            if gi not in confused_gt
            and ground_truth[gi]["class_id"] != pred["class_id"]
        ]
        overlap, gi = max(candidates, default=(0, -1))
        if overlap >= threshold and gi >= 0:
            confused_gt.add(gi)
            confusion.append(
                {
                    "gt_index": gi,
                    "prediction_index": pi,
                    "gt_class": ground_truth[gi]["class_id"],
                    "predicted_class": pred["class_id"],
                }
            )
    return {
        "counts": counts,
        "matched_gt": sorted(matched_gt),
        "missed_gt": missed,
        "false_positives": false_positives,
        "class_confusion_candidates": confusion,
    }


def summarize(counts):
    result = []
    for cls, row in enumerate(counts):
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        result.append(
            {"class_id": cls, **row, "precision": precision, "recall": recall, "f1": f1}
        )
    return result
