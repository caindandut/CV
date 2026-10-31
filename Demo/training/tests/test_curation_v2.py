import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curate_v2 import normalize_boxes, validate_external


def test_boxes_use_review_canvas_not_original_resolution():
    assert normalize_boxes([[2, 100, 50, 300, 150]], [400, 200]) == [[2, .5, .5, .5, .5]]


@pytest.mark.parametrize('box', [[0, -1, 0, 20, 20], [1, 0, 0, 500, 20], [2, 10, 0, 10, 20]])
def test_invalid_manual_boxes_fail_instead_of_clipping(box):
    with pytest.raises(ValueError):
        normalize_boxes([box], [400, 200])


def test_foreign_or_unverified_source_cannot_enter_dataset():
    record = dict(country='VN', location_evidence='File description: Vietnam',
                  visual_reviewed=True, boxes=[[0, 0, 0, 100, 100]])
    validate_external(record)
    for field, value in [('country', 'IN'), ('location_evidence', ''), ('visual_reviewed', False)]:
        with pytest.raises(ValueError):
            validate_external({**record, field: value})
