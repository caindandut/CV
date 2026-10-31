import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from clean_stage2_vietnam import (
    CleaningError,
    CropRecord,
    PRIORITY,
    canonical_source_name,
    find_exact_crop_duplicates,
    find_near_crop_duplicates,
    find_source_group_conflicts,
    find_source_image_duplicates,
    hamming_distance,
    phash_image,
    run_cleaning,
    source_group,
)


class CleanStage2Tests(unittest.TestCase):
    def _record(
        self,
        uid,
        split,
        class_name="helmet",
        sha256=None,
        source_sha256=None,
        phash="0000000000000000",
        group="MOV-1",
    ):
        return CropRecord(
            uid=uid,
            split=split,
            class_name=class_name,
            class_id="1" if class_name == "helmet" else "2",
            source_image=f"{split}_{uid}.jpg",
            source_label=f"{split}_{uid}.txt",
            object_index="1",
            bbox=("0", "0", "10", "10"),
            crop_box=("0", "0", "10", "10"),
            source_width="10",
            source_height="10",
            crop_width=10,
            crop_height=10,
            original_crop=Path(f"{split}_{uid}.jpg"),
            original_crop_relative=f"{split}/{class_name}/{uid}.jpg",
            canonical_source_name=f"{split}_{uid}",
            source_group=group,
            sha256=sha256 or f"crop-{uid}",
            source_sha256=source_sha256 or f"source-{uid}",
            phash=phash,
        )

    def test_canonical_source_name_strips_roboflow_hash_and_image_suffix(self):
        self.assertEqual(
            canonical_source_name("IMG_5136_MOV-17_jpg.rf.abc123.jpg"),
            "IMG_5136_MOV-17",
        )
        self.assertEqual(
            canonical_source_name("image_16_jpg.rf.hash.png"),
            "image_16",
        )

    def test_source_group_normalizes_mov_and_mp4_variants(self):
        self.assertEqual(source_group("IMG_5136_MOV-17_jpg.rf.x.jpg"), "MOV-17")
        self.assertEqual(source_group("IMG_5136_mov_17_jpg.rf.x.jpg"), "MOV-17")
        self.assertEqual(source_group("IMG_5136_mp47_jpg.rf.x.jpg"), "MP4-7")
        self.assertEqual(source_group("one_off_image_jpg.rf.x.jpg"), "one_off_image")

    def test_priority_keeps_test_over_valid_over_train(self):
        self.assertGreater(PRIORITY["test"], PRIORITY["valid"])
        self.assertGreater(PRIORITY["valid"], PRIORITY["train"])

    def test_hamming_distance_is_count_of_different_bits(self):
        self.assertEqual(hamming_distance("0" * 16, "f" * 16), 64)
        self.assertEqual(hamming_distance("0" * 16, "0" * 16), 0)

    def test_phash_is_deterministic_for_identical_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "crop.jpg"
            Image.new("RGB", (64, 64), (120, 80, 40)).save(image_path, quality=95)

            first = phash_image(image_path)
            second = phash_image(image_path)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 16)

    def test_exact_duplicate_keeps_valid_and_drops_train(self):
        train = self._record(1, "train", sha256="same")
        valid = self._record(2, "valid", sha256="same")

        rows, _ = find_exact_crop_duplicates([train, valid])

        self.assertEqual(rows[0]["action"], "KEEP_valid_DROP_train")
        self.assertEqual(train.drop_reasons, ["EXACT_CROP_DUPLICATE"])
        self.assertEqual(valid.drop_reasons, [])

    def test_exact_duplicate_with_different_labels_is_a_conflict(self):
        train = self._record(1, "train", class_name="helmet", sha256="same")
        valid = self._record(2, "valid", class_name="no_helmet", sha256="same")

        rows, _ = find_exact_crop_duplicates([train, valid])

        self.assertEqual(rows[0]["action"], "LABEL_CONFLICT")
        self.assertEqual(train.drop_reasons, [])
        self.assertEqual(valid.drop_reasons, [])

    def test_source_group_conflict_keeps_test_and_drops_lower_priority_splits(self):
        train = self._record(1, "train", group="MOV-7")
        valid = self._record(2, "valid", group="MOV-7")
        test = self._record(3, "test", group="MOV-7")

        rows = find_source_group_conflicts([train, valid, test])

        self.assertEqual(rows[0]["kept_split"], "test")
        self.assertIn("SOURCE_SEQUENCE_LEAKAGE", train.drop_reasons)
        self.assertIn("SOURCE_SEQUENCE_LEAKAGE", valid.drop_reasons)
        self.assertEqual(test.drop_reasons, [])

    def test_source_image_duplicate_marks_all_lower_priority_crops(self):
        train = self._record(1, "train", source_sha256="same-source")
        valid = self._record(2, "valid", source_sha256="same-source")

        rows = find_source_image_duplicates([train, valid])

        self.assertEqual(rows[0]["action"], "KEEP_valid_DROP_LOWER_PRIORITY")
        self.assertIn("SOURCE_IMAGE_DUPLICATE", train.drop_reasons)
        self.assertEqual(valid.drop_reasons, [])

    def test_near_duplicate_uses_priority_and_marks_lower_split(self):
        train = self._record(1, "train", phash="0000000000000000", group="train-only")
        valid = self._record(2, "valid", phash="0000000000000001", group="valid-only")

        rows, _ = find_near_crop_duplicates([train, valid])

        self.assertEqual(rows[0]["distance"], 1)
        self.assertEqual(rows[0]["action"], "KEEP_valid_DROP_train")
        self.assertIn("NEAR_DUPLICATE", train.drop_reasons)

    def test_run_cleaning_copies_only_priority_preserved_samples_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "Stage2_Vietnam"
            manifest_rows = []
            for split, class_name, filename, group, color in (
                ("train", "helmet", "train.jpg", "MOV-1", (255, 0, 0)),
                ("valid", "helmet", "valid.jpg", "MOV-1", (0, 255, 0)),
                ("test", "no_helmet", "test.jpg", "MOV-2", (0, 0, 255)),
            ):
                crop_dir = input_root / split / class_name
                source_dir = root / "sources"
                crop_dir.mkdir(parents=True, exist_ok=True)
                source_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / filename
                source_path = source_dir / f"IMG_000_{group}_jpg.rf.{split}.jpg"
                image = Image.new("RGB", (64, 64), color)
                image.save(crop_path, quality=95)
                image.save(source_path, quality=95)
                manifest_rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "class_id": "1" if class_name == "helmet" else "2",
                        "source_image": str(source_path),
                        "source_label": str(source_path.with_suffix(".txt")),
                        "object_index": "1",
                        "bbox_x1": "0",
                        "bbox_y1": "0",
                        "bbox_x2": "64",
                        "bbox_y2": "64",
                        "crop_x1": "0",
                        "crop_y1": "0",
                        "crop_x2": "64",
                        "crop_y2": "64",
                        "source_width": "64",
                        "source_height": "64",
                        "crop_width": "64",
                        "crop_height": "64",
                        "bbox_area_normalized": "1.0",
                        "output_crop": f"{split}/{class_name}/{filename}",
                    }
                )
            with (input_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
                writer.writeheader()
                writer.writerows(manifest_rows)

            output_root = run_cleaning(input_root, root / "Stage2_Vietnam_Clean", seed=3)

            with (output_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                cleaned = list(csv.DictReader(handle))
            self.assertEqual(len(cleaned), 2)
            self.assertEqual({row["split"] for row in cleaned}, {"valid", "test"})
            self.assertEqual(
                (output_root / "reports" / "source_group_conflicts.csv").read_text(encoding="utf-8").count("MOV-1"),
                1,
            )
            with self.assertRaises(CleaningError):
                run_cleaning(input_root, output_root, seed=3)


if __name__ == "__main__":
    unittest.main()
