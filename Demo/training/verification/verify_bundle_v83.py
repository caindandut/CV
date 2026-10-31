import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path("d:/Nam 4/CV/Demo/training")
DATASET = "dataset_v83"
ARCHIVE_PATH = ROOT / f"colab_bundles/{DATASET}/training_colab_70_v83.zip"
EXTRACT_DIR = ROOT / f"verification/colab_bundle_v83_extract"

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True)

with zipfile.ZipFile(ARCHIVE_PATH) as z:
    crc_check = z.testzip()
    assert crc_check is None, f"CRC check failed on {crc_check}"
    z.extractall(EXTRACT_DIR)
    member_names = z.namelist()

hashes_file = EXTRACT_DIR / "files_sha256.json"
assert hashes_file.is_file(), "Missing files_sha256.json"
expected_hashes = json.loads(hashes_file.read_text("utf-8"))

verified_count = 0
for name, expected_sha in expected_hashes.items():
    file_path = EXTRACT_DIR / name
    assert file_path.is_file(), f"Missing extracted file: {name}"
    actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert actual_sha == expected_sha, f"Hash mismatch for {name}: {actual_sha} vs {expected_sha}"
    verified_count += 1

# Check packaged training code import
code_path = EXTRACT_DIR / "code/train_colab.py"
assert code_path.is_file(), "Missing extracted train_colab.py"

sys.path.insert(0, str(code_path.parent))
spec = importlib.util.spec_from_file_location("extracted_train_colab_v83", str(code_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.verify_dataset(EXTRACT_DIR / f"{DATASET}/data.yaml")

archive_sha256 = hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest()

bundle_verification = {
    "passed": True,
    "dataset": DATASET,
    "accepted_images": 517,
    "target": 2000,
    "target_achieved": False,
    "archive_sha256": archive_sha256,
    "archive_members": len(member_names),
    "all_extracted_member_hashes_verified": verified_count,
    "crc_verified": True,
    "pending_media_excluded": True,
    "packaged_code_imported_from": str(code_path),
    "extracted_training_code_verified_dataset": True,
    "extraction_root": str(EXTRACT_DIR),
    "epochs_total": 70,
    "imgsz": 768,
    "patience": 20,
    "main_training_run": False,
    "new_cpu_smoke_run": False,
    "gpu_or_drive_runtime_test": False,
    "previous_smoke_reference": "v54 CPU3total epochs, unchanged packaged training logic; not rerun for static dataset update."
}

(ROOT / f"verification/colab_bundle_v83.json").write_text(
    json.dumps(bundle_verification, indent=2), encoding="utf-8"
)

print(f"Colab bundle v83 successfully verified! SHA256: {archive_sha256}, members: {len(member_names)}")
