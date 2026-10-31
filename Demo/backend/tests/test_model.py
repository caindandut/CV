import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.detector import Detector, validate_class_contract
from app.config import NAMES


def test_three_names_do_not_hide_eighty_class_head():
    model = SimpleNamespace(
        names=dict(enumerate(NAMES)),
        model=SimpleNamespace(model=[SimpleNamespace(nc=80)]),
    )
    with pytest.raises(ValueError, match="80 lớp"):
        validate_class_contract(model)


def test_matching_head_and_names_required():
    model = SimpleNamespace(
        names=dict(enumerate(NAMES)),
        model=SimpleNamespace(model=[SimpleNamespace(nc=3)]),
    )
    validate_class_contract(model)
    model.names = dict(enumerate(reversed(NAMES)))
    with pytest.raises(ValueError, match="Thứ tự"):
        validate_class_contract(model)


def test_current_checkpoint_can_use_eighty_channel_compatibility():
    model = SimpleNamespace(
        names=dict(enumerate(NAMES)),
        model=SimpleNamespace(model=[SimpleNamespace(nc=80)]),
    )
    validate_class_contract(
        model,
        checkpoint_sha256="f6b6b0ab4bfe794f52b446107cf2b79d548070641bb805440c3f4a976cbc8f39",
    )
    with pytest.raises(ValueError):
        validate_class_contract(model, checkpoint_sha256="different-checkpoint")


def test_prediction_requests_three_ids_and_never_labels_other_channels():
    calls = []

    def box(cls):
        return SimpleNamespace(
            cls=SimpleNamespace(item=lambda: cls),
            conf=SimpleNamespace(item=lambda: 0.9),
            xyxy=[SimpleNamespace(tolist=lambda: [1.0, 2.0, 30.0, 40.0])],
        )

    def predict(frame, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=[box(1), box(7)])]

    detector = Detector()
    detector.model = SimpleNamespace(predict=predict)
    detector.device = "cpu"
    result = detector.predict(None, 768, 0.25)
    assert calls[0]["classes"] == [0, 1, 2]
    assert [item["class_id"] for item in result] == [1]
    assert result[0]["class_name"] == "rider_with_helmet"


def test_custom_definitions_match_training_notebook():
    root = Path(__file__).resolve().parents[3]
    nb = json.loads(
        (root / "Train3_YOLO26s_SCConv_768.ipynb").read_text(encoding="utf-8")
    )
    source_nodes = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                source_nodes[node.name] = ast.dump(node, include_attributes=False)
    custom = ast.parse(
        (root / "Demo/backend/app/scconv.py").read_text(encoding="utf-8")
    )
    nodes = [n for n in custom.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert len(nodes) == 10
    for node in nodes:
        assert ast.dump(node, include_attributes=False) == source_nodes[node.name]
