"""Regression tests for UTF-8 and current counts in Colab hand-off files."""

import json
import sys
import unittest
from pathlib import Path


TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from build_current_bundle import render_notebook, render_readme  # noqa: E402


class BundleRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        progress = json.loads(
            (TRAINING / "acquisition_2000/progress.json").read_text(encoding="utf-8")
        )
        cls.dataset_name = progress["frozen_dataset"]
        cls.version = cls.dataset_name.removeprefix("dataset_")
        cls.rows = json.loads(
            (TRAINING / cls.dataset_name / "manifest.json").read_text(encoding="utf-8")
        )
        cls.active_count = sum(row["new_split"] != "pending" for row in cls.rows)
        cls.missing_count = max(0, 2000 - cls.active_count)

    def test_notebook_summary_uses_utf8_and_current_count(self):
        archive_name = f"training_colab_70_{self.version}.zip"
        notebook_name = f"Train_SCConv_Clean_70_{self.version}_Colab.ipynb"
        raw = render_notebook(
            TRAINING,
            self.dataset_name,
            archive_name,
            notebook_name,
            self.rows,
        )
        notebook = json.loads(raw)
        intro = "".join(notebook["cells"][0]["source"])

        self.assertIn(f"hiện có **{self.active_count}/2.000 ảnh**", intro)
        self.assertIn("Mục tiêu 2.000 ảnh chưa hoàn thành", intro)
        self.assertNotIn("\ufffd", intro)
        self.assertNotIn("Ã", intro)

    def test_readme_summary_uses_utf8_and_current_count(self):
        readme = render_readme(self.dataset_name, self.rows)
        missing_text = f"{self.missing_count:,}".replace(",", ".")

        self.assertIn(f"{self.active_count}/2.000 ảnh sạch", readme)
        self.assertIn(f"còn thiếu **{missing_text} ảnh**", readme)
        self.assertIn("giấy phép", readme)
        self.assertNotIn("\ufffd", readme)
        self.assertNotIn("Ã", readme)


if __name__ == "__main__":
    unittest.main()
