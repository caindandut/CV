from evaluation_metrics import iou, evaluate_image


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    assert iou([0, 0, 10, 10], [0, 0, 5, 10]) == 0.5


def test_duplicate_prediction_counts_once():
    gt = [{"class_id": 2, "xyxy": [0, 0, 10, 10]}]
    pred = [
        {"class_id": 2, "xyxy": [0, 0, 10, 10], "confidence": 0.9},
        {"class_id": 2, "xyxy": [0, 0, 10, 10], "confidence": 0.8},
    ]
    result = evaluate_image(gt, pred, 0.5)
    assert result["counts"][2] == {"tp": 1, "fp": 1, "fn": 0, "gt": 1}
    assert result["matched_gt"] == [0]


def test_wrong_helmet_class_is_fp_and_fn():
    gt = [{"class_id": 2, "xyxy": [0, 0, 10, 10]}]
    pred = [{"class_id": 1, "xyxy": [0, 0, 10, 10], "confidence": 0.9}]
    result = evaluate_image(gt, pred, 0.5)
    assert result["counts"][2]["fn"] == 1
    assert result["counts"][1]["fp"] == 1
    assert result["class_confusion_candidates"] == [
        {"gt_index": 0, "prediction_index": 0, "gt_class": 2, "predicted_class": 1}
    ]


def test_iou_threshold_and_empty_cases():
    gt = [{"class_id": 1, "xyxy": [0, 0, 10, 10]}]
    pred = [{"class_id": 1, "xyxy": [0, 0, 5, 10], "confidence": 0.9}]
    assert evaluate_image(gt, pred, 0.5)["counts"][1]["tp"] == 1
    assert evaluate_image(gt, pred, 0.75)["counts"][1]["tp"] == 0
    assert evaluate_image([], [], 0.5)["counts"][0]["gt"] == 0
