import csv
import tempfile
import unittest
from pathlib import Path


from create_dataset3 import (
    DatasetError,
    destination_filename,
    load_yaml_mapping,
    parse_label_file,
    resolve_split_path,
    select_luna_action,
    write_dataset_yaml,
)


class Dataset3UnitTests(unittest.TestCase):
    def test_resolve_split_path_supports_relative_paths_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "download" / "Dataset2"
            yaml_path = root / "data.yaml"
            yaml_path.parent.mkdir(parents=True)
            (root / "train" / "images").mkdir(parents=True)
            yaml_path.write_text("train: ../Dataset2/train/images\n", encoding="utf-8")

            resolved = resolve_split_path(yaml_path, "../Dataset2/train/images")

            self.assertEqual(resolved, (root / "train" / "images").resolve())

    def test_resolve_split_path_finds_nested_dataset_root_when_yaml_path_is_one_level_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp) / "download"
            root = outer / "DataSet2"
            yaml_path = root / "data.yaml"
            yaml_path.parent.mkdir(parents=True)
            (root / "train" / "images").mkdir(parents=True)
            yaml_path.write_text("train: ../train/images\n", encoding="utf-8")

            resolved = resolve_split_path(yaml_path, "../train/images")

            self.assertEqual(resolved, (root / "train" / "images").resolve())

    def test_parse_label_file_returns_class_counts_and_rejects_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "sample.txt"
            label_path.write_text(
                "0 0.5 0.5 0.2 0.3\n"
                "2 0.25 0.75 0.1 0.2\n",
                encoding="utf-8",
            )

            parsed = parse_label_file(label_path)

            self.assertEqual(parsed.num_objects, 2)
            self.assertEqual(parsed.class_counts, {0: 1, 1: 0, 2: 1})
            self.assertEqual(parsed.invalid_rows, [])

            label_path.write_text("1 0.5 0.5 1.2 0.3\n", encoding="utf-8")
            invalid = parse_label_file(label_path)
            self.assertEqual(len(invalid.invalid_rows), 1)
            self.assertIn("width", invalid.invalid_rows[0]["reason"])

    def test_destination_filename_uses_dataset_prefix_and_keeps_extension(self):
        self.assertEqual(destination_filename("vn", Path("frame_01.JPG")), "vn__frame_01.JPG")
        self.assertEqual(destination_filename("luna", Path("frame_01.jpg")), "luna__frame_01.jpg")

    def test_select_luna_action_prioritizes_vietnam_leakage_over_internal_duplicate(self):
        action = select_luna_action(
            exact_vietnam_splits={"valid"},
            exact_luna=False,
            near_vietnam_splits={"test"},
            near_luna=False,
        )
        self.assertEqual(action, "drop_vietnam_leakage")

        action = select_luna_action(
            exact_vietnam_splits=set(),
            exact_luna=False,
            near_vietnam_splits=set(),
            near_luna=True,
        )
        self.assertEqual(action, "drop_luna_duplicate")

    def test_generated_data_yaml_round_trips_with_numeric_class_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "Dataset3"
            output_root.mkdir()
            write_dataset_yaml(output_root)

            metadata = load_yaml_mapping(output_root / "data.yaml")

            self.assertEqual(metadata["nc"], 3)
            self.assertEqual(metadata["names"], {0: "motorcycle", 1: "rider_with_helmet", 2: "rider_without_helmet"})


if __name__ == "__main__":
    unittest.main()
