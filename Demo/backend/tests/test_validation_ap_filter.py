import torch
from evaluate_validation_ap import ThreeClassValidator
from ultralytics.models.yolo.detect.val import DetectionValidator


def test_ap_validator_does_not_score_unmapped_head_channels(monkeypatch):
    raw = {
        "cls": torch.tensor([0.0, 1.0, 2.0, 3.0, 79.0]),
        "conf": torch.ones(5),
        "bboxes": torch.zeros(5, 4),
        "extra": torch.zeros(5, 0),
    }
    monkeypatch.setattr(
        DetectionValidator, "postprocess", lambda self, predictions: [raw]
    )
    validator = ThreeClassValidator.__new__(ThreeClassValidator)
    result = validator.postprocess(None)[0]
    assert result["cls"].tolist() == [0.0, 1.0, 2.0]
    assert result["bboxes"].shape == (3, 4)
    assert result["extra"].shape == (3, 0)
