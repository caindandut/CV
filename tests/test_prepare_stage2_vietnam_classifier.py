import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_stage2_vietnam_classifier import (
    REQUIRED_NAMES,
    crop_rider_region,
    find_source_image_leakage,
    load_dataset_metadata,
    parse_yolo_label_file,
    run_preparation,
)


class Stage2ClassifierPreparationTests(unittest.TestCase):
    def test_crop_rider_region_uses_upper_half_and_padding_without_leaving_image(self):
        crop = crop_rider_region(
            (20, 20, 60, 100),
            image_width=100,
            image_height=120,
        )

        self.assertEqual(crop, (14, 12, 66, 68))

    def test_parse_yolo_label_file_keeps_valid_rows_and_reports_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "frame.txt"
            label_path.write_text(
                "1 0.5 0.5 0.4 0.5\n"
                "2 0.25 0.75 1.2 0.2\n"
                "broken row\n",
                encoding="utf-8",
            )

            parsed = parse_yolo_label_file(label_path)

            self.assertEqual(len(parsed.valid_rows), 1)
            self.assertEqual(parsed.valid_rows[0].class_id, 1)
            self.assertEqual(len(parsed.invalid_rows), 2)
            self.assertIn("width", parsed.invalid_rows[0]["reason"])

    def test_load_dataset_metadata_requires_exact_vietnam_dataset_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "data.yaml"
            yaml_path.write_text(
                "train: ../train/images\n"
                "val: ../valid/images\n"
                "test: ../test/images\n"
                "nc: 3\n"
                "names: ['motorcycle', 'rider_with_helmet', 'rider_without_helmet']\n"
                "roboflow:\n"
                "  workspace: nckh-2023\n"
                "  project: helmet-detection-project\n"
                "  version: 19\n",
                encoding="utf-8",
            )

            metadata = load_dataset_metadata(yaml_path)

            self.assertTrue(metadata.is_required_dataset)
            self.assertEqual(metadata.names, REQUIRED_NAMES)

    def test_load_dataset_metadata_accepts_numeric_names_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "data.yaml"
            yaml_path.write_text(
                "train: train/images\n"
                "val: valid/images\n"
                "test: test/images\n"
                "nc: 3\n"
                "names:\n"
                "  0: motorcycle\n"
                "  1: rider_with_helmet\n"
                "  2: rider_without_helmet\n"
                "roboflow:\n"
                "  workspace: nckh-2023\n"
                "  project: helmet-detection-project\n"
                "  version: 19\n",
                encoding="utf-8",
            )

            metadata = load_dataset_metadata(yaml_path)

            self.assertEqual(metadata.names, REQUIRED_NAMES)
            self.assertTrue(metadata.is_required_dataset)

    def test_source_image_leakage_is_reported_between_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jpg"
            valid = root / "valid.jpg"
            train.write_bytes(b"same image bytes")
            valid.write_bytes(b"same image bytes")

            leakage = find_source_image_leakage(
                {
                    "train": {"img-a": train},
                    "valid": {"img-b": valid},
                    "test": {},
                }
            )

            self.assertEqual(leakage[0]["split_a"], "train")
            self.assertEqual(leakage[0]["split_b"], "valid")

    def test_run_preparation_creates_stage2_layout_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Dataset2"
            dataset.mkdir()
            yaml_path = dataset / "data.yaml"
            yaml_path.write_text(
                "train: train/images\n"
                "val: valid/images\n"
                "test: test/images\n"
                "nc: 3\n"
                "names: ['motorcycle', 'rider_with_helmet', 'rider_without_helmet']\n"
                "roboflow:\n"
                "  workspace: nckh-2023\n"
                "  project: helmet-detection-project\n"
                "  version: 19\n",
                encoding="utf-8",
            )
            for split, class_id, color in (
                ("train", 1, (255, 0, 0)),
                ("valid", 2, (0, 255, 0)),
                ("test", 1, (0, 0, 255)),
            ):
                image_dir = dataset / split / "images"
                label_dir = dataset / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                image_path = image_dir / f"{split}_frame.jpg"
                Image.new("RGB", (100, 100), color).save(image_path, quality=95)
                (label_dir / f"{image_path.stem}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.6 0.6\n",
                    encoding="utf-8",
                )

            output_root = run_preparation(root, seed=7)

            self.assertTrue((output_root / "manifest.csv").exists())
            self.assertTrue((output_root / "stage2_summary.txt").exists())
            self.assertTrue((output_root / "reports" / "visual_audit.csv").exists())
            self.assertEqual(len(list((output_root / "train" / "helmet").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output_root / "valid" / "no_helmet").glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
